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
        batch_size, num_workers_dl, validate_task_name=AppConstants.TASK_VALIDATION):
        super().__init__()

        # AB: Parameters
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.abspath(os.path.join(dir_path, data_path)) # AB: This is to make sure that the path is correct.
        test_data_dir = os.path.join(data_path, "test")
        test_coco = os.path.join(test_data_dir, "_annotations.coco.json")
        
        self._validate_task_name = validate_task_name

        # Setup the model
        resnetNetwork = ResnetFasterRCNN(num_classes)
        self.model = resnetNetwork.get_model()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        test_dataset = PklotDataSet(
            root_path=test_data_dir, annotation_path=test_coco, transforms=resnetNetwork.get_transform()
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
                val_accuracy = self._validate(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Accuracy when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {val_accuracy}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def _validate(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()
        # TODO: AB: Implement this function
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for i, (images, labels) in enumerate(self._test_loader):
        #         if abort_signal.triggered:
        #             return 0

        #         images, labels = images.to(self.device), labels.to(self.device)
        #         output = self.model(images)

        #         _, pred_label = torch.max(output, 1)

        #         correct += (pred_label == labels).sum().item()
        #         total += images.size()[0]

        #     metric = correct / float(total)
        metric = None # TODO: AB: Remove this line
        return metric
