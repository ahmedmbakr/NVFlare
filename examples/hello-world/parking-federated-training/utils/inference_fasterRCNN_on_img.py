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
import normal_pklot_trainer
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

def get_transforms(pretrained=True): # TODO: AB: Consider for now that we only use pretrained images
    # Transform the image
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    return transforms

if __name__ == "__main__":
    # Read image:
    image_path = "/home/bakr/pklot/train/2013-03-06_09_00_03_jpg.rf.e2ebe82b00611d3e7d1710765c640507.jpg"
    model_path = "/home/bakr/NVFlare/examples/hello-world/parking-federated-training/models/model_0.pth"
    score_threshold = .8

    # Load the model
    trainer = normal_pklot_trainer.PklotTrainer(inference=True)
    model = trainer.get_model(config.num_classes, pretrained=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the image
    img = read_image(image_path)

    images_list = [img]

    # Transform the image
    transforms = get_transforms()
    transformed_images = [transforms(imgx) for imgx in images_list]

    # Make predictions
    model = model.eval()
    outputs = model(transformed_images)

    dogs_with_boxes = [
        draw_bounding_boxes(dog_int, boxes=output['boxes'][output['scores'] > score_threshold], width=4)
        for dog_int, output in zip(images_list, outputs)
    ]
    show(dogs_with_boxes)