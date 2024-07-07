import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torch.nn as nn
import torch.nn.functional as F

class ResnetFasterRCNN(FastRCNNPredictor):
    def __init__(self, num_classes=3): # 3 classes: space-empty, space-occupied, background
        super(ResnetFasterRCNN, self).__init__(in_channels=3, num_classes=num_classes)
        self.num_classes = num_classes
        self = ResnetFasterRCNN.__get_model(num_classes)
    
    def get_model(self): # TODO: AB: To be removed
        return self.__get_model(self.num_classes) 

    @staticmethod
    def __get_model(num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT) # TODO: AB: Experiment with this. If you want to use a pre-trained model, set it to True
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model
    
    # In my case, just added ToTensor
    @staticmethod
    def get_transform():
        custom_transforms = [] # TODO: AB: If you are going to use a trained model, make sure that the normalization is the same as the one used during training
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)
