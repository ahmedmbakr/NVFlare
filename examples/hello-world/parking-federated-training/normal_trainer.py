# Main Sources:
# 1. https://github.com/tkshnkmr/frcnn_medium_sample/blob/master/utils.py
# 2. https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
# 3. https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html

import os, sys
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PkLotDataLoader import PklotDataSet, collate_fn
from torchvision.models.detection import SSD300_VGG16_Weights

dir_path = os.path.dirname(os.path.realpath(__file__))
mAP_path = os.path.abspath(os.path.join(dir_path, 'jobs/parking-federated-training/app/custom'))
print("mAP path: ", mAP_path)
sys.path.append(mAP_path)
import mAP
import SSDnet
import Yolov5net

from yolov5.utils.general import non_max_suppression # pip install yolov5

class_id_to_name_dict = {1: "Space-empty", 2: "Space-occupied"}

class ParkingTrainer:

    def __init__(self, config, model_name, inference=False, continue_training=False):
        self.config = config
        self.continue_training = continue_training
        self.model_name = model_name
        self.outputs_dir = os.path.abspath(os.path.join(dir_path, 'outputs'))
        if os.path.exists(self.outputs_dir):
            os.system(f"rm -rf {self.outputs_dir}")
        os.makedirs(self.outputs_dir)
        # create own Dataset
        train_pklot_dataset = PklotDataSet(
            root_path=self.config.train_data_dir, annotation_path=self.config.train_coco, model_name=model_name, transforms=self.get_transform()
        )

        val_pklot_dataset = PklotDataSet(
            root_path=self.config.val_data_dir, annotation_path=self.config.val_coco, model_name=model_name, transforms=self.get_transform()
        )

         # own DataLoader
        self.train_data_loader = torch.utils.data.DataLoader(
            train_pklot_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=self.config.train_shuffle_dl,
            num_workers=self.config.num_workers_dl,
            collate_fn=collate_fn,
        )

        self.val_data_loader = torch.utils.data.DataLoader(
            val_pklot_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers_dl,
            collate_fn=collate_fn,
        )

        # select device (whether GPU or CPU)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = self.get_model(self.config.num_classes, self.config.pre_trained_model_allowed)

        # move model to the right device
        self.model.to(self.device)

        # parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay
        )

        # If models folder exist, remove it, then create another one
        if inference == False:
            if self.continue_training == False and os.path.exists(self.config.models_folder):
                os.system(f"rm -rf {self.config.models_folder}")
            os.makedirs(self.config.models_folder)


    # In my case, just added ToTensor
    def get_transform(self):
        if self.config.pre_trained_model_allowed:
            # From: https://pytorch.org/vision/0.18/auto_examples/others/plot_repurposing_annotations.html#sphx-glr-auto-examples-others-plot-repurposing-annotations-py
            if self.model_name == 'resnet':
                from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                tranforms = weights.transforms()
            elif self.model_name == 'ssdnet':
                tranforms = SSDnet.SSDVGG16.get_transform()
            elif self.model_name == 'yolov5':
                tranforms = Yolov5net.YOLOv5.get_transform()
            return tranforms
        else:
            custom_transforms = [] # TODO: AB: If you are going to use a trained model, make sure that the normalization is the same as the one used during training
            custom_transforms.append(torchvision.transforms.ToTensor())
            return torchvision.transforms.Compose(custom_transforms)


    def get_model(self, num_classes, pretrained):
        # https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
        # load an instance segmentation model pre-trained pre-trained on COCO
        if not pretrained:
            if self.model_name == 'resnet':
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
            elif self.model_name == 'ssdnet':
                model = torchvision.models.detection.ssd300_vgg16()
            elif self.model_name == 'yolov5':
                model = Yolov5net.YOLOv5(num_classes=num_classes, pretrained=False)
        else:
            # Try this method and check the difference TODO: AB: Check the difference between the two methods
            if self.model_name == 'resnet':
                from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
                # get number of input features for the classifier
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                # replace the pre-trained head with a new one
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            elif self.model_name == 'ssdnet':
                model = SSDnet.SSDVGG16(num_classes=num_classes).model # TODO: AB: Add pretrained parameter
            elif self.model_name == 'yolov5':
                model = Yolov5net.YOLOv5(num_classes=num_classes, pretrained=True).model

        return model
    
    def __visualize_images_with_boxes_func(self, imgs, annotations):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import torchvision.transforms.functional as F
        debug_images_directory = os.path.join(self.outputs_dir, "debug_images")
        if os.path.exists(debug_images_directory): # If the folder exists, remove it. Then, create it again
            os.system(f"rm -rf {debug_images_directory}")
        os.makedirs(debug_images_directory)

        
        for i, img in enumerate(imgs):
            fig, axs = plt.subplots()
            img = img.permute(1, 2, 0).cpu().numpy()
            img = F.to_pil_image(img)
            axs.imshow(img)
            for box, label in zip(annotations[i]['boxes'], annotations[i]['labels']):
                x, y, w, h = box
                edge_color = 'g' if label == 1 else 'r'
                rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor=edge_color, facecolor='none')
                axs.add_patch(rect)
            axs.axis('off')
            # save image
            image_path = os.path.join(debug_images_directory, f"image_{i}.png")
            plt.savefig(image_path)
            print(f"save image with boxes in: {image_path}")
        print("Done Saving all the images with boxes in the directory: ", debug_images_directory)
    
    def local_train(self, visualize_images_with_boxes=False):
        if self.model_name == 'yolov5':
            self.yolo_loss = Yolov5net.YoloLoss()
        len_dataloader = len(self.train_data_loader)
        # Training
        trackers = {"train_loss": [], "val_acc": []}
        for epoch in range(self.config.num_epochs):
            running_loss = 0.0
            import time
            start_time = time.time()
            print(f"Epoch: {epoch}/{self.config.num_epochs}")
            self.model.train()
            i = 0
            for imgs, annotations in self.train_data_loader:
                if visualize_images_with_boxes: # AB: Used only for debugging
                    self.__visualize_images_with_boxes_func(imgs, annotations)
                # imgs = list(img.to(self.device) for img in imgs)
                imgs = torch.stack(imgs).to(self.device)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
                if len(annotations) == 0:
                    continue
                
                # If loss_dict is a list or tuple:
                if self.model_name == 'yolov5':
                    predictions = self.model(imgs)
                    # Compute the loss using predictions and annotations
                    losses = self.yolo_loss(predictions, annotations)
                else:
                    loss_dict = self.model(imgs, annotations)
                    # If it’s a dictionary (in case the structure varies), you can use the original approach
                    losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}", end='\r')
                i += 1
                running_loss += losses.cpu().detach().numpy() / imgs[0].size()[0]

            epoch_loss = running_loss / len_dataloader
            print("\n")
            print(f"Epoch {epoch} Loss: {epoch_loss}")
            # Validation
            metric = self.validate(self.val_data_loader, epoch, self.config.mAP_detection_threshold)
            trackers['train_loss'].append(epoch_loss)
            trackers['val_acc'].append(metric)
            import pickle
            pickle.dump(trackers, open(self.config.mAP_metric_file_path, "wb"))
            # Write the metrics as a json file
            import json
            with open(self.config.mAP_metric_file_path.replace(".pkl", ".json"), "w") as f:
                json.dump(trackers, f)
            # Save the model
            model_path = os.path.join(self.config.models_folder, f"model_{epoch}.pth")
            torch.save(self.model.state_dict(), model_path)

            epoch_end_time = time.time()
            from datetime import timedelta
            diff_sec = epoch_end_time - start_time
            num_remaining_epochs = self.config.num_epochs - epoch - 1
            td = timedelta(seconds=diff_sec)
            print(f"Epoch {epoch} took {td}. The remaining time is {td * num_remaining_epochs}")
    
    def validate(self, val_loader, epoch, detection_threshold=0.5):
        """
        This function is used to validate the model on the validation set.
        It calculates the mAP score for the model.
        Inputs:
        - val_loader: PyTorch DataLoader object for the validation set.
        - detection_threshold: Threshold for considering detected objects. If you encountered an error when the validation runs, change this number to a lower value (e.x. 0.3)
        Outputs:
        - metric: Metrics after running mAP calculation. 1) AP per class, 2) precision per class, 3) recall per class, 4) log average miss rate per class, 5) mAP
        """
        # Remove the files in the input directory needed by mAP calculation
        if os.path.exists(self.config.mAP_val_prediction_directory):
            os.system(f"rm -rf {self.config.mAP_val_prediction_directory}")
        os.makedirs(self.config.mAP_val_prediction_directory)

        if os.path.exists(self.config.mAP_val_gt_directory):
            os.system(f"rm -rf {self.config.mAP_val_gt_directory}")
        os.makedirs(self.config.mAP_val_gt_directory)

        self.model.eval()  # Set the model to evaluation mode
        device = self.device
        len_val_loader = len(val_loader)
        with torch.no_grad():  # No need to track gradients
            for batch_id, (imgs, annotations) in enumerate(val_loader):
                # imgs = list(img.to(device) for img in imgs)
                imgs = torch.stack(imgs).to(self.device).contiguous()
                print("imgs shape: ", imgs.shape)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                predictions = self.model(imgs)  # Get model predictions
                if self.model_name == 'yolov5':
                    predictions = self.transform_yolov5_predictions(predictions)
                
                for i, prediction in enumerate(predictions):
                    # Post-process the predictions to remove low scoring parts
                    pred_scores = prediction['scores']
                    pred_boxes = prediction['boxes']
                    pred_labels = prediction['labels']

                    # Filter out predictions based on detection threshold
                    keep = pred_scores > detection_threshold
                    pred_boxes = pred_boxes[keep].cpu().numpy().tolist()
                    pred_labels = pred_labels[keep].cpu().numpy().tolist()
                    # Convert labels to integer values
                    pred_labels = [int(label) for label in pred_labels]
                    pred_scores = pred_scores[keep].cpu().numpy().tolist()

                    # Ground truth
                    gt_boxes = list(annotations[i]['boxes'].cpu().numpy())
                    gt_labels = list(annotations[i]['labels'].cpu().numpy())

                    unique_image_id = batch_id * len(imgs) + i
                    file_name = f'{unique_image_id}.txt'
                    prediction_file_path = os.path.join(self.config.mAP_val_prediction_directory, file_name)
                    with open(prediction_file_path, 'w') as f:
                        for box, label_id, score in zip(pred_boxes, pred_labels, pred_scores):
                            if label_id == 0:
                                continue
                            label_name = class_id_to_name_dict[label_id]
                            f.write(f'{label_name} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')

                    gt_file_path = os.path.join(self.config.mAP_val_gt_directory, file_name)
                    with open(gt_file_path, 'w') as f:
                        for box, label_id in zip(gt_boxes, gt_labels):
                            label_name = class_id_to_name_dict[label_id]
                            f.write(f'{label_name} {box[0]} {box[1]} {box[2]} {box[3]}\n')

                    # Calculate and print some form of validation metric
                    # This is where you might calculate intersection-over-union (IoU) and derive your precision/recall metrics.
                    # For simplicity, here we are just printing out the number of true positives, etc.
                    # In practice, you would use something like COCOeval from the pycocotools library to do this.
                        # print(f"Processed {i+1}/{len(val_loader)} images")
                        # print(f"Predictions: {len(pred_scores)} objects detected")
                    # Example metric (not implemented here): IoU, Precision, Recall, mAP
                if batch_id % 10 == 0:
                    print(f"Validating batch: {batch_id} / {len_val_loader}", end="\r")
        print("\n")

        mAP_val_input_dir = os.path.abspath(os.path.join(self.config.mAP_val_prediction_directory, '..'))
        mAP_val_output_dir = os.path.abspath(os.path.join(self.outputs_dir, f'mAPOutputs/{epoch}'))

        if os.path.exists(mAP_val_output_dir):
            os.system(f"rm -rf {mAP_val_output_dir}")
        os.makedirs(mAP_val_output_dir)
        
        metric = mAP.calculate_mAP(mAP_val_input_dir, mAP_val_output_dir)
        print("Validation complete.")
        return metric
    
    def transform_yolov5_predictions(self, predictions):
        ret_list_predictions = []
        # Run non-max suppression on the predictions
        nms_predictions = non_max_suppression(predictions)

        # Iterate over predictions to extract boxes, scores, and labels
        for pred in nms_predictions:
            if pred is not None:
                # boxes are in the format [x1, y1, x2, y2]
                boxes = pred[:, :4]  # Bounding boxes
                scores = pred[:, 4]  # Confidence scores
                labels = pred[:, 5]  # Class labels

                # Now boxes, scores, and labels are ready to be used
                # print("Boxes:", boxes)
                # print("Scores:", scores)
                print("Labels:", labels)
                ret_list_predictions.append({"boxes": boxes, "scores": scores, "labels": labels})
        return ret_list_predictions
    
    def test_model(self, test_coco_paths_list, test_names, detection_threshold=0.5):
        """
        In this function, we will test the model using all the given test coco files.
        Inputs:
        - test_coco_paths_list: List of paths to the test coco files.
        - test_names: List of names for the test coco files. Used for printing purposes and the same name will be used as the folder name to save the results.
        """
        for test_coco_path, test_name in zip(test_coco_paths_list, test_names):
            test_coco_dir_path = os.path.dirname(test_coco_path)
            test_pklot_dataset = PklotDataSet(
            root_path=test_coco_dir_path, annotation_path=test_coco_path, transforms=self.get_transform()
            )

            test_data_loader = torch.utils.data.DataLoader(
                test_pklot_dataset,
                batch_size=self.config.train_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers_dl,
                collate_fn=collate_fn,
            )

            metric = self.validate(test_data_loader, test_name, detection_threshold)
            print(f"Testing completed on dataset: {test_name}. mAP: {metric['mAP']}, AP: {metric['ap']}, log_avg_miss_rate: {metric['log_avg_miss_rate']}")


if __name__ == "__main__":
    PKLOT_CNR_TRAINING_SELECTOR = 'PKLOT'
    MODEL_NAME='yolov5' # The model name can be either 'resnet', 'ssdnet' or 'yolov5'
    if PKLOT_CNR_TRAINING_SELECTOR == 'PKLOT':
        import pklot_trainer_config as config
    elif PKLOT_CNR_TRAINING_SELECTOR == 'PUCPR':
        import pklot_PUCPR_trainer_config as config
    elif PKLOT_CNR_TRAINING_SELECTOR == 'UFPR04':
        import pklot_UFPR04_trainer_config as config
    elif PKLOT_CNR_TRAINING_SELECTOR == 'UFPR05':
        import pklot_UFPR05_trainer_config as config
    elif PKLOT_CNR_TRAINING_SELECTOR == 'CNR':
        import cnr_trainer_config as config
    trainer = ParkingTrainer(config, model_name=MODEL_NAME)
    trainer.local_train()
    trainer.test_model(config.test_coco_paths_list, config.test_names, config.mAP_detection_threshold)
