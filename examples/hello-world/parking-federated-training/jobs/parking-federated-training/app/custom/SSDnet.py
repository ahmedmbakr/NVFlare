import torch
import torchvision
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
import torch.nn as nn

class SSDVGG16(torch.nn.Module):
    def __init__(self, num_classes=3):  # Adjust based on the number of your classes
        super(SSDVGG16, self).__init__()
        self.num_classes = num_classes
        self.model = self.get_pretrained_model(num_classes)

    @staticmethod
    def get_pretrained_model(num_classes):
        # Load SSD model pre-trained on COCO
        model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

        # Fetch the number of anchors (default boxes) per location for each feature map
        num_anchors_per_location = model.anchor_generator.num_anchors_per_location()

        # Fetch the in_channels and num_anchors for each feature map from the current classification head
        in_channels = [layer.in_channels for layer in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()

        # Replace the SSD classification head with one that matches the new number of classes
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,   # List of in_channels for each feature map
            num_anchors=num_anchors,   # List of num_anchors for each feature map
            num_classes=num_classes    # Number of classes to predict
        )

        return model
    
    def forward(self, x):
        return self.model(x)

    # Optionally: Define the transformation pipeline based on the pre-trained weights
    @staticmethod
    def get_transform():
        # Use the default transformations used by the pre-trained model
        return SSD300_VGG16_Weights.DEFAULT.transforms()

if __name__ == "__main__":
    model = SSDVGG16.get_pretrained_model(3)
    print(model)
    # print(SSDVGG16.get_transform())