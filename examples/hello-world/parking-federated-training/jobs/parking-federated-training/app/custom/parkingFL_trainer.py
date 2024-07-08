# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union
import copy
import numpy as np
import os.path

import torch
from pt_constants import PTConstants
from Resnet import ResnetFasterRCNN
from PkLotDataLoader import PklotDataSet, collate_fn
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import GTSRB
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torchvision, pickle
import torch.utils.data

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.apis.fl_constant import FLMetaKey, ReturnCode

class_id_to_name_dict = {1: "Space-empty", 2: "Space-occupied"}


class ParkingFL_Trainer(ModelLearner):
    def __init__(
        self,
        data_path,
        lr,
        epochs,
        num_classes, 
        batch_size,
        valid_detection_threshold,
        fedproxloss_mu: float = 0.0,
        shuffle_training_data_enable=True,
        num_workers_dl=4,
    ):
        """Init function for the ParkingFL_Trainer class.

        Args:
            data_path: data (train/valid/test) root path
            lr: learning rate
            epochs: number of local epochs before aggregating with the server
            num_classes: number of classes in the dataset
            batch_size: batch size
            valid_detection_threshold: threshold for detection. Used to filter out low scoring detections during the validation phase.
            fedproxloss_mu: FedProx loss parameter. Default is 0.0 (FedAvg). If its value is greater than zero, then it becomes FedProx.
            shuffle_training_data_enable: whether to shuffle the training data or not. The default value is True.
            num_workers_dl: number of workers for the DataLoader. The default value is 4.
        """
        super().__init__()

        # AB: Parameters
        self.data_path = data_path
        self._lr = lr
        self._epochs = epochs
        self.num_classes = num_classes
        self.batch_size = batch_size
        self._valid_detection_threshold = valid_detection_threshold
        self.fedproxloss_mu = fedproxloss_mu
        self.shuffle_training_data_enable = shuffle_training_data_enable
        self.num_workers_dl = num_workers_dl

    def initialize(self):
        """
        Initialization function for the ParkingFL_Trainer class. This function is automatically called from `nvflare/app_common/executors/model_learner_executor.py`
        """
        # when the run starts, this is where the actual settings get initialized for trainer
        self.info(
            f"Client {self.site_name} initializing at \n {self.app_root} \n with args: {self.args}",
        )

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_path = os.path.abspath(os.path.join(self.dir_path, self.data_path)) # AB: This is to make sure that the path is correct.
        self.info(f"Number of classes: {self.num_classes}, Learning rate: {self._lr}, Number of epochs: {self._epochs}, Batch size: {self.batch_size}, data path: {self.data_path}, valid_detection_threshold: {self._valid_detection_threshold}")

        # Training setup
        self.model = ResnetFasterRCNN.get_pretrained_model(self.num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=self._lr, momentum=0.9)

        train_data_path = self.data_path # AB: This is the path to the data for this client.
        train_data_dir = os.path.join(train_data_path, "train")
        val_data_dir = os.path.join(train_data_path, "valid")
        train_coco = os.path.join(train_data_dir, "_annotations.coco.json")
        val_coco = os.path.join(val_data_dir, "_annotations.coco.json")

        self._train_dataset = PklotDataSet(
            root_path=train_data_dir, annotation_path=train_coco, transforms=ResnetFasterRCNN.get_transform()
        )

        self._val_dataset = PklotDataSet(
            root_path=val_data_dir, annotation_path=val_coco, transforms=ResnetFasterRCNN.get_transform()
        )

         # own DataLoader
        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training_data_enable,
            num_workers=self.num_workers_dl,
            collate_fn=collate_fn,
        )

        self._validate_loader = torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_dl,
            collate_fn=collate_fn,
        )

        # Split the dataset
        self.info(f"The initialization is running from this folder: {os.path.abspath(__file__)}")

        self._n_iterations = len(self._train_loader)
        self.info(f"Number of iterations: {self._n_iterations}")
        # print(f"Shape of the whole dataset: {self._train_dataset.dataset.data.shape}")
        self.info(f"Number of samples from the whole data: {len(self._train_dataset)} = Number of iterations ({self._n_iterations}) * batch size ({self.batch_size})")

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf
        )

        if self.fedproxloss_mu > 0:
            print(f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)

        self.outputs_dir = os.path.abspath(os.path.join(self.dir_path, '../outputs'))
        self.models_dir = os.path.abspath(os.path.join(self.outputs_dir, 'models'))
        
        # If models folder exist, remove it, then create another one
        if os.path.exists(self.models_dir):
            os.system(f"rm -rf {self.models_dir}")
        os.makedirs(self.models_dir)

        self.local_epoch = None
        self.global_epoch = -1 # Because it is incremented by one before the beginning of the first epoch. This is a hack to make it start from 0.
        self.overall_trackers = {"train_loss": [], "val_acc": []}
        self.best_mAP = 0.0
        self.best_local_model_file = None # Its value is set inside the save_model function.

        # AB: Note that the data is downloaded on my local machine in the path: "~/data", and it is shared between all the clients.
        print(f"ParkingFL_Trainer initialized: This is the path of the data: {self.data_path} for client: {self.site_name}") # AB: This was just to make sure that print statements will be displayed in the output. It is displayed in the CMD, but not in the log files, which is expected.

    def train(self, model: FLModel) -> Union[str, FLModel]:
        """
        This function trains the model. It is called from the `nvflare/app_common/executors/model_learner_executor.py` file.
        """
        # get round information
        self.info(f"Enter train function for Client: {self.site_name}. Current/Total Round: {self.current_round + 1}/{self.total_rounds}")
        self.info(f"Client identity: {self.site_name}")

        # update local model weights with received weights
        global_weights = model.params

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self._train_loader)
        self.info(f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for potential FedProx loss or SCAFFOLD
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # local train
        self._local_train(
            self.fl_ctx,
            model_global=model_global
        )
        self.save_model(is_best=False)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = np.subtract(local_weights[name].cpu().numpy(), global_weights[name], dtype=np.float32)
            if np.any(np.isnan(model_diff[name])):
                self.stop_task(f"{name} weights became NaN...")
                return ReturnCode.EXECUTION_EXCEPTION

        # return an FLModel containing the model differences
        fl_model = FLModel(params_type=ParamsType.DIFF, params=model_diff)

        FLModelUtils.set_meta_prop(fl_model, FLMetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)
        self.info("Local epochs finished. Returning FLModel")
        return fl_model

    def validate(self, model: FLModel) -> Union[str, FLModel]:
        """
        This function validates the model after each epoch. It is called from the `nvflare/app_common/executors/model_learner_executor.py` file.
        """
        # get validation information
        self.info(f"Started validation function for Client: {self.site_name}")

        # update local model weights with received weights
        global_weights = model.params

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        # self.info(f'AB: global_weights keys: {model_keys}')
        # self.info(f'AB: local_var_dict keys: {local_var_dict.keys()}')
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        # get validation meta info
        validate_type = FLModelUtils.get_meta_prop(
            model, FLMetaKey.VALIDATE_TYPE, ValidateType.MODEL_VALIDATE
        )  # TODO: enable model.get_meta_prop(...)
        model_owner = self.get_shareable_header(AppConstants.MODEL_OWNER)

        # perform valid
        # AB: No need to validate the model on the training data. We will only validate the model on the validation data.
        # train_acc = self._validate(self._train_loader)
        # self.info(f"training acc ({model_owner}): {train_acc:.4f}")
        if self.global_epoch == len(self.overall_trackers['val_acc']):
            # I did this because the evaluation is called 6 times that I do not want. First, using the initial model before the start of the training. Then, at the end 5 times at the last epoch. It validates on the same model for the same client. I only need one reading from them. This is a bug that I need to fix. # TODO: AB: Replace the final validation by validating different model from the users.
            metrics = self._validate(self._validate_loader, self.global_epoch, self.fl_ctx, self._valid_detection_threshold)
            self.info(f"Validation completed for round: {self.global_epoch}, global_epoch: {self.global_epoch}. mAP: {self.overall_trackers['val_acc'][self.global_epoch]['mAP']}, AP: {self.overall_trackers['val_acc'][self.global_epoch]['ap']}, log_avg_miss_rate: {self.overall_trackers['val_acc'][self.global_epoch]['log_avg_miss_rate']}")

            self.overall_trackers['val_acc'].append(metrics)
            mAP = metrics['mAP']
            
            self.info("Evaluation finished. Returning result")

            pickle_file_path = os.path.abspath(os.path.join(self.outputs_dir, 'overall_trackers.pkl'))
            pickle.dump(self.overall_trackers, open(pickle_file_path, 'wb')) # AB: Save the training trackers to the disk after each epoch

            if mAP > self.best_mAP:
                self.best_mAP = mAP
                self.save_model(is_best=True)

            # val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc} # AB: Commented because we are not validating the model on the training data.
            val_results = {"mAP": mAP}
        else:
            val_results = {"mAP": -1} # AB: This is a hack default value.
        return FLModel(metrics=val_results)

    def save_model(self, is_best=False):
        """
        This function saves the model in the self.models_dir directory. Note: If is_best is True, then the model will be saved as the best model and all other models in the same folder will be removed.
        """
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.global_epoch}
        if is_best:
            model_path = os.path.abspath(os.path.join(self.models_dir, f"best_local_model.pt"))
            save_dict.update({"best_mAP": self.best_mAP})
            # Remove the previous saved model
            os.system(f"rm {self.models_dir}/*.pth")
            # Save the new best model
            torch.save(save_dict, model_path)
            self.best_local_model_file = model_path
        else:
            model_path = os.path.abspath(os.path.join(self.models_dir, f"local_model.pt"))
            torch.save(save_dict, model_path)
        self.info(f"Client {self.site_name} Model saved to {model_path}")

    def _local_train(self, fl_ctx, model_global):
        len_dataloader = len(self._train_loader)

        # Basic training
        for local_epoch in range(self._epochs):
            self.local_epoch = local_epoch
            self.global_epoch += 1 # It is initialized to -1 in the initialize function.
            running_loss = 0.0
            import time
            start_time = time.time()
            self.model.train()
            for i, (imgs, annotations) in enumerate(self._train_loader):
                if self.abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                imgs = list(img.to(self.device) for img in imgs)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
                if len(annotations) == 0:
                    continue

                loss_dict = self.model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    losses += fed_prox_loss
                
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    self.log_info(fl_ctx, f"AB: Round: {self.current_round}, Epoch: {local_epoch}/{self._epochs}, Global epoch: {self.global_epoch}, Iteration: {i}/{len_dataloader}, Loss: {losses}")

                running_loss += losses.cpu().detach().numpy() / imgs[0].size()[0]

            epoch_loss = running_loss / len(self._train_loader)
            self.log_info(
                        fl_ctx, f"AB: Round: {self.global_epoch}, Epoch: {local_epoch}/{self._epochs}, global epoch: {self.global_epoch}, Iteration: {i}, " f"Loss: {epoch_loss}")
            self.overall_trackers['train_loss'].append(epoch_loss)

            # Display the time taken for the local_epoch
            epoch_end_time = time.time()
            from datetime import timedelta
            td = timedelta(seconds=epoch_end_time - start_time)
            self.log_info(fl_ctx, f"Epoch {self.global_epoch} took: {td}")

    def _validate(self, val_loader, epoch, fl_ctx, detection_threshold=0.5):
        """
        Validate the model on the given loader (validate or test).
        """
        mAP_val_prediction_directory = os.path.abspath(os.path.join(self.outputs_dir, 'mapInput/detection-results'))
        mAP_val_gt_directory = os.path.abspath(os.path.join(self.outputs_dir, "mapInput/ground-truth"))

        # If the directories exist, remove them
        if os.path.exists(mAP_val_prediction_directory):
            os.system(f"rm -rf {mAP_val_prediction_directory}")
        if os.path.exists(mAP_val_gt_directory):
            os.system(f"rm -rf {mAP_val_gt_directory}")

        # Create the directories
        os.makedirs(mAP_val_prediction_directory)
        os.makedirs(mAP_val_gt_directory)

        self.model.eval()  # Set the model to evaluation mode
        device = self.device
        len_val_loader = len(val_loader)
        with torch.no_grad():  # No need to track gradients
            for batch_id, (imgs, annotations) in enumerate(val_loader):
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                predictions = self.model(imgs)  # Get model predictions
                
                for i, prediction in enumerate(predictions):
                    # Post-process the predictions to remove low scoring parts
                    pred_scores = prediction['scores']
                    pred_boxes = prediction['boxes']
                    pred_labels = prediction['labels']

                    # Filter out predictions based on detection threshold
                    keep = pred_scores > detection_threshold
                    pred_boxes = pred_boxes[keep].cpu().numpy().tolist()
                    pred_labels = pred_labels[keep].cpu().numpy().tolist()
                    pred_scores = pred_scores[keep].cpu().numpy().tolist()

                    # Ground truth
                    gt_boxes = list(annotations[i]['boxes'].cpu().numpy())
                    gt_labels = list(annotations[i]['labels'].cpu().numpy())

                    unique_image_id = batch_id * len(imgs) + i
                    file_name = f'{unique_image_id}.txt'
                    prediction_file_path = os.path.join(mAP_val_prediction_directory, file_name)
                    with open(prediction_file_path, 'w') as f:
                        for box, label_id, score in zip(pred_boxes, pred_labels, pred_scores):
                            label_name = class_id_to_name_dict[label_id]
                            f.write(f'{label_name} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')

                    gt_file_path = os.path.join(mAP_val_gt_directory, file_name)
                    with open(gt_file_path, 'w') as f:
                        for box, label_id in zip(gt_boxes, gt_labels):
                            label_name = class_id_to_name_dict[label_id]
                            f.write(f'{label_name} {box[0]} {box[1]} {box[2]} {box[3]}\n')

                # if batch_id % 10 == 0:
                #     print(f"Validating batch: {batch_id} / {len_val_loader}", end="\r")
        # print("\n")
        from mAP import calculate_mAP
        mAP_val_input_dir = os.path.abspath(os.path.join(mAP_val_prediction_directory, '..'))
        mAP_val_output_dir = os.path.abspath(os.path.join(self.outputs_dir, f'mapOutputs/{epoch}'))

        if os.path.exists(mAP_val_output_dir):
            os.system(f"rm -rf {mAP_val_output_dir}")
        os.makedirs(mAP_val_output_dir)
        
        self.log_info(fl_ctx, f"mAP_val_input_dir: {mAP_val_input_dir}, mAP_val_output_dir: {mAP_val_output_dir}")
        metric = calculate_mAP(mAP_val_input_dir, mAP_val_output_dir)
        self.log_info(fl_ctx,"Validation complete.")
        return metric

    def _load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml
    
    def get_model(self, model_name: str) -> Union[str, FLModel]:
        """
        This function is called from nvflare/app_common/executors/model_learner_executor.py
        """
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except Exception as e:
                raise ValueError("Unable to load best model") from e

            # Create FLModel from model data.
            if model_data:
                # convert weights to numpy to support FOBS
                model_weights = model_data["model_weights"]
                for k, v in model_weights.items():
                    model_weights[k] = v.numpy()
                return FLModel(params_type=ParamsType.FULL, params=model_weights)
            else:
                # Set return code.
                self.error(f"best local model not found at {self.best_local_model_file}.")
                return ReturnCode.EXECUTION_RESULT_ERROR
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.
    
    def finalize(self):
        """
        This function is called at the end of the training. It is called from the `nvflare/app_common/executors/model_learner_executor.py` file.
        """
        # collect threads, close files here
        # TODO: I will perform the test here only if the training is successful.
        pass
