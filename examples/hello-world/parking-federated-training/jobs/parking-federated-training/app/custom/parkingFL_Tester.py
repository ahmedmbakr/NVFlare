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

import torch
from Resnet import ResnetFasterRCNN
from SSDnet import SSDVGG16
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torchvision
from PkLotDataLoader import PklotDataSet, collate_fn
import torch.utils.data

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

import os

class ParkingFL_Tester(Executor):
    def __init__(self, data_path, num_classes,
        batch_size, num_workers_dl, valid_detection_threshold, model_name, validate_task_name=AppConstants.TASK_VALIDATION):
        super().__init__()

        # AB: Parameters
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.abspath(os.path.join(dir_path, data_path)) # AB: This is to make sure that the path is correct.
        test_data_dir = os.path.join(data_path, "test")
        test_coco = os.path.join(test_data_dir, "_annotations.coco.json")

        self.outputs_dir = os.path.abspath(os.path.join(dir_path, '../outputs'))
        self.test_models_on_test_data_results_file_path = os.path.join(self.outputs_dir, "test_results.txt")
        
        self._validate_task_name = validate_task_name
        self._valid_detection_threshold = valid_detection_threshold

        # Setup the model
        if model_name == "resnet":
            resnetNetwork = ResnetFasterRCNN(num_classes)
            self.model = resnetNetwork.get_model()
        elif model_name == "ssdnet":
            alexNetNetwork = SSDVGG16(num_classes)
            self.model = alexNetNetwork.get_model()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        if model_name == "resnet":
            transforms = resnetNetwork.get_transform()
        elif model_name == "ssdnet":
            transforms = alexNetNetwork.get_transform()
        test_dataset = PklotDataSet(
            root_path=test_data_dir, annotation_path=test_coco, transforms=transforms
        )

        self._test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers_dl,
            collate_fn=collate_fn,
        )

        # Split the dataset
        print(f"The initialization is running from this folder: {os.path.abspath(__file__)}")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                mAP = self._validate(weights, abort_signal, fl_ctx, model_owner)
                # Append the test results to the test_results file
                with open(self.test_models_on_test_data_results_file_path, 'a') as f:
                    f.write(f"{model_owner}: {mAP}\n")

                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"mAP when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {mAP}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"mAP": mAP})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def _validate(self, weights, abort_signal, fl_ctx, model_owner):
        """
        Validate the model on the given test loader.
        """
        from parkingFL_trainer import class_id_to_name_dict
        self.model.load_state_dict(weights)

        mAP = ParkingFL_Tester.validate_model_on_test_data(self.model, model_owner, self.outputs_dir, self.device, self._test_loader, self._valid_detection_threshold)
        return mAP
    
    @staticmethod
    def validate_model_on_test_data(model, model_owner, outputs_dir, device, test_loader, valid_detection_threshold):
        from parkingFL_trainer import class_id_to_name_dict

        mAP_val_prediction_directory = os.path.abspath(os.path.join(outputs_dir, 'mapInput/detection-results'))
        mAP_val_gt_directory = os.path.abspath(os.path.join(outputs_dir, "mapInput/ground-truth"))

        # If the directories exist, remove them
        if os.path.exists(mAP_val_prediction_directory):
            os.system(f"rm -rf {mAP_val_prediction_directory}")
        if os.path.exists(mAP_val_gt_directory):
            os.system(f"rm -rf {mAP_val_gt_directory}")

        # Create the directories
        os.makedirs(mAP_val_prediction_directory)
        os.makedirs(mAP_val_gt_directory)

        model.eval()  # Set the model to evaluation mode
        len_val_loader = len(test_loader)
        with torch.no_grad():  # No need to track gradients
            for batch_id, (imgs, annotations) in enumerate(test_loader):
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                predictions = model(imgs)  # Get model predictions
                
                for i, prediction in enumerate(predictions):
                    # Post-process the predictions to remove low scoring parts
                    pred_scores = prediction['scores']
                    pred_boxes = prediction['boxes']
                    pred_labels = prediction['labels']

                    # Filter out predictions based on detection threshold
                    keep = pred_scores > valid_detection_threshold
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

                if batch_id % 10 == 0:
                    print(f"Validating batch: {batch_id} / {len_val_loader}", end="\r")
        print("\n")
        from mAP import calculate_mAP
        mAP_val_input_dir = os.path.abspath(os.path.join(mAP_val_prediction_directory, '..'))
        mAP_val_output_dir = os.path.abspath(os.path.join(outputs_dir, f'mapOutputs/testOn_{model_owner}'))

        if os.path.exists(mAP_val_output_dir):
            os.system(f"rm -rf {mAP_val_output_dir}")
        os.makedirs(mAP_val_output_dir)
        
        print(f"mAP_val_input_dir: {mAP_val_input_dir}, mAP_val_output_dir: {mAP_val_output_dir}")
        metric = calculate_mAP(mAP_val_input_dir, mAP_val_output_dir)
        print(f"Validation complete on the model from {model_owner} with the results mAP: {metric['mAP']}, ap: {metric['ap']}, log_avg_miss_rate: {metric['log_avg_miss_rate']}")
        return metric['mAP'] # Return only the mAP
