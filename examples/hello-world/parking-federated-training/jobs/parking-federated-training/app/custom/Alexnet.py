import torch.utils.data
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models import AlexNet_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import alexnet
import torch.nn as nn
import torch.nn.functional as F

class AlexNetFasterRCNN(FasterRCNN):
    def __init__(self, num_classes=3): # 3 classes: space-empty, space-occupied, background
        super(AlexNetFasterRCNN, self).__init__(in_channels=3, num_classes=num_classes)
        self = AlexNetFasterRCNN.get_pretrained_model(num_classes) # AB: This does not yield the required results. I depend on get_pretrained_model function to get the model

    @staticmethod
    def get_pretrained_model(num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT) # TODO: AB: Experiment with this. If you want to use a pre-trained model, set it to True
        # get number of input features for the classifier
        in_features = model.classifier[6].in_features
        # replace the pre-trained head with a new one
        model.classifier[6] = nn.Linear(in_features, num_classes)

        return model
    
    # In my case, just added ToTensor
    @staticmethod
    def get_transform():
        custom_transforms = [] # TODO: AB: If you are going to use a trained model, make sure that the normalization is the same as the one used during training
        custom_transforms.append(torchvision.transforms.ToTensor())
        # Normalization for ImageNet pre-trained weights: mean and std used during training
        custom_transforms.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                   std=[0.229, 0.224, 0.225]))
        return torchvision.transforms.Compose(custom_transforms)
