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

# This file was changed by AB to incorporate the AlexNet model.
# This network was taken from the following source: https://mailto-surajk.medium.com/a-tutorial-on-traffic-sign-classification-using-pytorch-dabc428909d7

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes:int = 43, dropout: float = 0.5):
        super().__init__()
        # AB: The original AlexNet model was designed for ImageNet dataset, which has 1000 classes. I took the architecture from https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py which is part of the original PyTorch library. I modified the output layer to have the number of needed classes.
        # How to use it example: https://pytorch.org/hub/pytorch_vision_alexnet/
        # To understand it, you can refer to this medium link: https://medium.com/@siddheshb008/alexnet-architecture-explained-b6240c528bd5. Its a good explanation of the AlexNet architecture. However, the architecture is slightly different from the one I used here (Original Pytorch implementation).

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = torch.flatten(x, 1)
        x = self.classifier(h)
        return x #, h # AB: I changed this one as I only want the classification output, not the feature extractor layer output
