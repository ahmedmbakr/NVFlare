# Main Sources:
# 1. https://github.com/tkshnkmr/frcnn_medium_sample/blob/master/utils.py
# 2. https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5


import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pklot_trainer_config as config
from PkLotDataLoader import PklotDataSet, collate_fn

class_id_to_name_dict = {1: "Space-empty", 2: "Space-occupied"}

class PklotTrainer:

    def __init__(self, continue_training=False):
        self.continue_training = continue_training

        # create own Dataset
        train_pklot_dataset = PklotDataSet(
            root_path=config.train_data_dir, annotation_path=config.train_coco, transforms=self.get_transform()
        )

        val_pklot_dataset = PklotDataSet(
            root_path=config.val_data_dir, annotation_path=config.val_coco, transforms=self.get_transform()
        )

         # own DataLoader
        self.train_data_loader = torch.utils.data.DataLoader(
            train_pklot_dataset,
            batch_size=config.train_batch_size,
            shuffle=config.train_shuffle_dl,
            num_workers=config.num_workers_dl,
            collate_fn=collate_fn,
        )

        self.val_data_loader = torch.utils.data.DataLoader(
            val_pklot_dataset,
            batch_size=config.train_batch_size,
            shuffle=False,
            num_workers=config.num_workers_dl,
            collate_fn=collate_fn,
        )

        # select device (whether GPU or CPU)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = self.get_model(config.num_classes)

        # move model to the right device
        self.model.to(self.device)

        # parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
        )

        # If models folder exist, remove it, then create another one
        if self.continue_training == False and os.path.exists(config.models_folder):
            os.system(f"rm -rf {config.models_folder}")
        os.makedirs(config.models_folder)


    # In my case, just added ToTensor
    def get_transform(self):
        custom_transforms = [] # TODO: AB: If you are going to use a trained model, make sure that the normalization is the same as the one used during training
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)


    def get_model(self, num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False) # TODO: AB: Experiment with this. If you want to use a pre-trained model, set it to True
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model
    
    def local_train(self):
        metrics = [] # The metrics for each epoch
        len_dataloader = len(self.train_data_loader)
        # Training
        running_loss = 0.0
        for epoch in range(config.num_epochs):
            print(f"Epoch: {epoch}/{config.num_epochs}")
            self.model.train()
            i = 0
            for imgs, annotations in self.train_data_loader:
                imgs = list(img.to(self.device) for img in imgs)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
                if len(annotations) == 0:
                    continue
                loss_dict = self.model(imgs, annotations)
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
            metric = self.validate(self.val_data_loader)
            metric['epoch'] = epoch
            metric['loss'] = epoch_loss
            metrics.append(metric)
            import pickle
            pickle.dump(metrics, open(config.mAP_metric_file_path, "wb"))
            # Write the metrics as a json file
            import json
            with open(config.mAP_metric_file_path.replace(".p", ".json"), "w") as f:
                json.dump(metrics, f)
            # Save the model
            model_path = os.path.join(config.models_folder, f"model_{epoch}.pth")
            torch.save(self.model.state_dict(), model_path)
    
    def validate(self, val_loader, detection_threshold=0.5):
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
        os.system(f"rm -rf {config.mAP_val_prediction_directory}/*.txt")
        os.system(f"rm -rf {config.mAP_val_gt_directory}/*.txt")
        self.model.eval()  # Set the model to evaluation mode
        device = self.device
        len_val_loader = len(val_loader)
        with torch.no_grad():  # No need to track gradients
            for batch_id, (imgs, annotations) in enumerate(val_loader):
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                predictions = self.model(imgs)  # Get model predictions
                
                for i, prediction in enumerate(predictions):
                    # Post-process the predictions to remove low scoring parts
                    pred_scores = prediction['scores']
                    pred_boxes = prediction['boxes']
                    pred_labels = prediction['labels']

                    # Filter out predictions based on detection threshold
                    keep = pred_scores > detection_threshold
                    pred_boxes = pred_boxes[keep].cpu().numpy().tolist()
                    pred_labels = pred_labels[keep].cpu().numpy().tolist()
                    pred_scores = pred_scores[keep].cpu().numpy().tolist()

                    # Ground truth
                    gt_boxes = list(annotations[i]['boxes'].cpu().numpy())
                    gt_labels = list(annotations[i]['labels'].cpu().numpy())

                    unique_image_id = batch_id * len(imgs) + i
                    file_name = f'{unique_image_id}.txt'
                    prediction_file_path = os.path.join(config.mAP_val_prediction_directory, file_name)
                    with open(prediction_file_path, 'w') as f:
                        for box, label_id, score in zip(pred_boxes, pred_labels, pred_scores):
                            label_name = class_id_to_name_dict[label_id]
                            f.write(f'{label_name} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')

                    gt_file_path = os.path.join(config.mAP_val_gt_directory, file_name)
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
        from mAP import calculate_mAP
        metric = calculate_mAP('input', 'outputs') # TODO: AB: Add the folder names as part of the configuration
        print("Validation complete.")
        return metric

if __name__ == "__main__":
    trainer = PklotTrainer()
    trainer.local_train()
