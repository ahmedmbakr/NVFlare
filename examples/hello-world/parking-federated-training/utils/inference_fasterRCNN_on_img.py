# Main resources: https://pytorch.org/vision/0.18/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
import torch
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
sys.path.append(os.path.abspath(os.path.join(dir_path, '../jobs/parking-federated-training/app/custom')))
print(sys.path)
import Resnet
import SSDnet
import Yolov5net
import normal_trainer
import pklot_trainer_config as config

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def get_transforms(model_name, pretrained=True): # TODO: AB: Consider for now that we only use pretrained images
    # Transform the image
    if model_name == "resnet":
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        transforms = weights.transforms()
    elif model_name == "ssdnet":
        from torchvision.models.detection import FasterRCNN_AlexNet_Weights
        weights = FasterRCNN_AlexNet_Weights.DEFAULT
        transforms = weights.transforms()
    elif model_name == "yolov5":
        transforms = Yolov5net.YOLOv5.get_transform()
    
    return transforms

if __name__ == "__main__":
    # Read image:
    # image_path = "/home/bakr/pklot/train/2013-03-06_09_00_03_jpg.rf.e2ebe82b00611d3e7d1710765c640507.jpg"
    image_path = '/home/bakr/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera1/2015-11-12_0909.jpg'
    model_path = "/home/bakr/NVFlare/examples/hello-world/parking-federated-training/models/model_3.pth"
    score_threshold = .2
    model_name = "yolov5" # The model name can be either "resnet" or "ssdnet" or "yolov5"

    # Load the model
    trainer = normal_trainer.ParkingTrainer(config=config, inference=True)
    model = trainer.get_model(config.num_classes, pretrained=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the image
    img = read_image(image_path)

    images_list = [img]

    # Transform the image
    transforms = get_transforms(model_name)
    transformed_images = [transforms(imgx) for imgx in images_list]

    # Make predictions
    model = model.eval()
    outputs = model(transformed_images)

    image_with_boxes = [
        draw_bounding_boxes(dog_int, boxes=output['boxes'][output['scores'] > score_threshold], width=4)
        for dog_int, output in zip(images_list, outputs)
    ]
    show(image_with_boxes)
