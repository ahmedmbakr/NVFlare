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

import os.path

import torch
from pt_constants import PTConstants
from alex_net_network import AlexnetTS
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import GTSRB
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torchvision, pickle

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class Gtsrb43Trainer(Executor):
    def __init__(
        self,
        data_path="../../../data",
        lr=0.01,
        epochs=1,
        num_classes = 43,
        batch_size = 32,
        train_val_split = 0.8,
        random_seed = 42,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None,
        pre_train_task_name=AppConstants.TASK_GET_WEIGHTS,
    ):
        """Cifar10 Trainer handles train and submit_model tasks. During train_task, it trains a
        simple network on CIFAR10 dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
            pre_train_task_name: Task name for pre train task, i.e., sending initial model weights.
            train_val_split: This is the percentage of the data that will be used for the training. The rest will be used for the validation.
        """
        super().__init__()

        # AB: Parameters
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.abspath(os.path.join(dir_path, data_path)) # AB: This is to make sure that the path is correct.
        
        torch.manual_seed(random_seed) # AB: This is to make sure that the training and the validation data are split in the same way for all the clients.

        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._pre_train_task_name = pre_train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        print(f"Number of classes: {num_classes}, Learning rate: {lr}, Number of epochs: {epochs}, Batch size: {batch_size}, Train validation split: {train_val_split}, data path: {data_path}")

        # Training setup
        self.model = AlexnetTS(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # Create Cifar10 dataset for training.
        transforms = Compose(
            [
                Resize([112, 112]),
                ToTensor()
            ]
        )
        train_data_path = data_path # AB: This is the path to the data for this client.
        self._dataset = torchvision.datasets.ImageFolder(root = train_data_path, transform = transforms)

        n_train_examples = int(len(self._dataset) * train_val_split)
        n_val_examples = len(self._dataset) - n_train_examples

        self._train_dataset, self._val_dataset = random_split(self._dataset, [n_train_examples, n_val_examples]) 


        # Split the dataset
        print(f"The initialization is running from this folder: {os.path.abspath(__file__)}")

        self._train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)
        self._validate_loader = DataLoader(self._val_dataset, batch_size=batch_size, shuffle=False)
        self._n_iterations = len(self._train_loader)
        print(f"Number of iterations: {self._n_iterations}")
        # print(f"Shape of the whole dataset: {self._train_dataset.dataset.data.shape}")
        print(f"Number of samples from the whole data: {len(self._train_dataset.indices)} = Number of iterations ({self._n_iterations}) * batch size ({batch_size})")

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf
        )

        # AB: Note that the data is downloaded on my local machine in the path: "~/data", and it is shared between all the clients.
        print(f"Gtsrb43Trainer initialized: This is the path of the data: {data_path}") # AB: This was just to make sure that print statements will be displayed in the output. It is displayed in the CMD, but not in the log files, which is expected.

        self.__num_times_to_call_trainer = 0 # AB: This variable is incremented every time the train function is called.
        self.overall_trackers = {"train_loss": [], "val_acc": []}


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Print the path of the executing file.
        # AB: This will be printed in the CMD and the log files for 
        self.log_info(fl_ctx, f"Executing file: {os.path.abspath(__file__)}")
        try:
            if task_name == self._pre_train_task_name:
                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self._local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self._save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self._load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _get_model_weights(self) -> Shareable:
        # Get the new state dict and send as weights
        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )
        return outgoing_dxo.to_shareable()

    def _local_train(self, fl_ctx, weights, abort_signal, validate_enabled=True):
        self.__num_times_to_call_trainer += 1

        # Set the model weights
        self.model.load_state_dict(state_dict=weights)

        trackers = {"train_loss": [], "val_acc": []}

        # Basic training
        self.model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            for i, batch in enumerate(self._train_loader):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                self.optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                # if i % 3000 == 0:
                #     self.log_info(
                #         fl_ctx, f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, " f"Loss: {running_loss/3000}"
                #     )
                #     running_loss = 0.0
            epoch_loss = running_loss / len(self._train_loader)
            self.log_info(
                        fl_ctx, f"AB: Epoch: {epoch}/{self._epochs}, Iteration: {i}, " f"Loss: {epoch_loss}")
            trackers['train_loss'].append(epoch_loss)
            if validate_enabled:
                trackers['val_acc'].append(self._validate(self._validate_loader, fl_ctx))
                self.log_info(fl_ctx, f"AB: Validation accuracy (correct / total validation images): {(trackers['val_acc'][epoch] * 100):.2f}%")
                self.display_train_trackers(trackers, fl_ctx, is_overall=False)
                self.overall_trackers['train_loss'].append(epoch_loss)
                self.overall_trackers['val_acc'].append(trackers['val_acc'][epoch])
                self.display_train_trackers(self.overall_trackers, fl_ctx, is_overall=True)

                dir_path = os.path.dirname(os.path.realpath(__file__))
                outputs_dir = os.path.abspath(os.path.join(dir_path, '../outputs'))
                pickle_file_path = os.path.abspath(os.path.join(outputs_dir, 'overall_trackers.pkl'))
                pickle.dump(self.overall_trackers, open(pickle_file_path, 'wb')) # AB: Save the training trackers to the disk after each epoch

    def _validate(self, loader, fl_ctx):
        """
        Validate the model on the given loader (validate or test).
        """
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)

                _, pred_label = torch.max(output, 1)

                correct += (pred_label == labels).sum().item()
                total += images.size()[0]
            self.log_info(fl_ctx, f"AB: Correct: {correct}, not Correct: {total - correct}, Total: {total}")
            metric = correct / float(total)

        return metric

    def display_train_trackers(self, trackers, fl_ctx, is_overall=False):
        import pickle
        import matplotlib.pyplot as plt
        import os

        dir_path = os.path.dirname(os.path.realpath(__file__))
        outputs_dir = os.path.abspath(os.path.join(dir_path, '../outputs'))
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        
        # print(trackers)
        # Display train loss graph.
        plt.figure() # New graph
        plt.plot(trackers['train_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Train loss')
        plt.title('Train loss vs Epochs')
        # Save the graph to the disk
        save_file_name = 'normal_train_loss_overall.png' if is_overall else f'normal_train_loss_{self.__num_times_to_call_trainer}.png'
        plt.savefig(os.path.abspath(os.path.join(outputs_dir, save_file_name)))

        # Display validation accuracy graph.
        plt.figure() # New graph
        plt.plot(trackers['val_acc'])
        plt.xlabel('Epochs')
        plt.ylabel('Validation accuracy')
        plt.title('Validation accuracy vs Epochs')
        # Save the graph to the disk
        save_file_name = 'normal_val_acc_overall.png' if is_overall else f'normal_val_acc_{self.__num_times_to_call_trainer}.png'
        plt.savefig(os.path.abspath(os.path.join(outputs_dir, save_file_name)))

    def _save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

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
