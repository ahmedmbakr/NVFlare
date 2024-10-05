import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # For bounding box regression
        self.bce_loss = nn.BCELoss()  # For objectness score
        self.ce_loss = nn.CrossEntropyLoss()  # For classification loss
    
    def forward(self, predictions, targets, num_classes=3):
        """
        Arguments:
        predictions: Model output (bounding boxes, objectness, class predictions).
        targets: Ground truth annotations (bounding boxes, class labels).

        Returns:
        total_loss: Combined loss (objectness, classification, and bounding box regression).
        """
        predictions = [torch.clamp(pred, min=0.0, max=1.0) for pred in predictions]

        # 1. Unpack predictions
        pred_boxes = [pred[..., :4] for pred in predictions]
        pred_objectness = [pred[..., 4] for pred in predictions]
        pred_class_probs = [pred[..., 5:] for pred in predictions]


        # 2. Unpack ground truth (targets)
        true_boxes = [ann['boxes'] for ann in targets] if isinstance(targets, list) else targets['boxes']
        
        # To get the total number of boxes for objectness:
        num_boxes_per_image = [box.shape[0] for box in true_boxes]  # Number of boxes per image

        # Create true_objectness as a list of tensors filled with ones, same length as number of boxes per image
        true_objectness = [torch.ones(num_boxes, device=box.device) for num_boxes, box in zip(num_boxes_per_image, true_boxes)]

        true_class = [ann['labels'] for ann in targets] if isinstance(targets, list) else targets['labels']

        # Concatenate the list of tensors along the batch dimension (first dimension)
        pred_boxes = torch.cat([pred.view(pred.size(0), -1, pred.size(-1)) for pred in pred_boxes], dim=1)  # Concatenate along dimension 1
        true_boxes = torch.cat([true.view(true.size(0), -1, true.size(-1)) for true in true_boxes], dim=1)  # Do the same for true_boxes

        # Reshape the predictions (if needed, but they should already be in a flat form)
        pred_objectness = torch.cat([obj.view(obj.size(0), -1) for obj in pred_objectness], dim=1)  # Flatten predictions

        # Reshape the ground truth objectness to match the prediction size
        # Assuming true_objectness is a list of tensors, similar to the predictions
        true_objectness = torch.cat([obj.view(-1) for obj in true_objectness], dim=0)  # Flatten true labels

        # Ensure the true_objectness is expanded to match pred_objectness
        # true_objectness now needs to match the batch size and number of anchors
        if true_objectness.size(0) != pred_objectness.size(1):
            true_objectness = true_objectness.repeat(1, pred_objectness.size(1) // true_objectness.size(0))


        # 3. Compute losses
        print("AB: pred_boxes shape", pred_boxes.shape)
        print("AB: true_boxes shape", true_boxes.shape)
        
        # Bounding box regression loss (using MSE or IoU)
        box_loss = self.mse_loss(pred_boxes, true_boxes)

        print("AB: pred_objectness shape", pred_objectness.shape)
        print("AB: true_objectness shape", true_objectness.shape)
        
        # Objectness loss (Binary Cross Entropy)
        objectness_loss = self.bce_loss(pred_objectness, true_objectness)

        # true_class = [cls.long() for cls in true_class]

        # Remove any values in pred_class_probs with values less than 0 or greater than 1
        pred_class_probs = [torch.clamp(pred, min=0.0, max=1.0) for pred in pred_class_probs]
        # Convert pred_class_probs to numpy array
        # pred_class_probs = [pred.cpu().detach().numpy() for pred in pred_class_probs]
        # print (pred_class_probs)
        # unique_values = [torch.unique(pred) for pred in pred_class_probs]
        # print(unique_values)

        # Concatenate predicted class probabilitie
        # pred_class_probs = torch.cat([pred.view(pred.size(0), -1, pred.size(-1)) for pred in pred_class_probs], dim=1)

        # true_class = torch.cat([cls.view(cls.size(0), -1) for cls in true_class], dim=1) 
        true_class_one_hot = [F.one_hot(cls, num_classes=num_classes) for cls in true_class]

        pred_class_probs = torch.cat([pred.view(pred.size(0), -1, pred.size(-1)) for pred in pred_class_probs], dim=1)  # Concatenate along dimension 1

        true_class_one_hot = torch.cat([true.view(true.size(0), -1, true.size(-1)) for true in true_class_one_hot], dim=1)  # Do the same for true_boxes

        true_class_one_hot = true_class_one_hot.view(1, -1, 3)

        pred_class_probs = pred_class_probs[:, :true_class_one_hot.size(1), :] # Slice to match the true class predictions

        true_class_one_hot = true_class_one_hot.float()

        print("AB: pred_class_probs shape", pred_class_probs.shape)
        print("AB: true_class_one_hot shape", true_class_one_hot.shape)

        # Classification loss (Cross Entropy Loss)
        class_loss = self.ce_loss(pred_class_probs, true_class_one_hot)

        # 4. Total loss
        total_loss = box_loss + objectness_loss + class_loss

        return total_loss

class YOLOv5(nn.Module):
    
    def __init__(self, num_classes=3, pretrained=True):  # Adjust based on the number of your classes
        super(YOLOv5, self).__init__()
        self.num_classes = num_classes
        self.model = self.get_pretrained_model(num_classes, pretrained)

    @staticmethod
    def get_pretrained_model(num_classes, pretrained=True):
        # Load YOLOv5 model using torch hub
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=pretrained)

        # Modify the YOLOv5 detection head (Detect layer)
        # YOLOv5 uses the 'model' attribute for its detection layers, and the last layer is the Detect layer.
        detect_layer = model.model.model.model[-1]  # Get the Detect layer. I know this by printing the model

        detect_layer.nc = num_classes  # Update the number of classes in the model

        num_outputs_per_anchor = (num_classes + 5)  # Number of outputs per anchor (class scores + 5 for objectness and bounding box regression)

        detect_layer.no = num_outputs_per_anchor  # Update the number of outputs per anchor (class predictions + bbox)

        print("AB: My modified Number of outputs per anchor: ", num_outputs_per_anchor)
        print("AB: Model's number of anchors", detect_layer.na)
        print("AB: Model's number of anchors of outputs per anchor", detect_layer.no)
        print("AB: Model's number of classes", detect_layer.nc)
        # print("AB: The height of feature maps:", model.ny)

        # Modify the convolutional layers in the Detect module to match the new number of classes
        for i, conv in enumerate(detect_layer.m):
            in_channels = conv.in_channels
            detect_layer.m[i] = nn.Conv2d(in_channels, detect_layer.anchors[i].shape[0] * num_outputs_per_anchor, kernel_size=1)

        return model

    def forward(self, x):
        # Forward pass through the YOLOv5 model
        return self.model(x)

    # Optionally: Define the transformation pipeline based on the pre-trained weights
    @staticmethod
    def get_transform():
        # Use the default transformations defined for YOLOv5
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((640, 640)),  # Resize to YOLOv5 input size (can be changed)
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

if __name__ == "__main__":
    model = YOLOv5(num_classes=3)
    print(model)
