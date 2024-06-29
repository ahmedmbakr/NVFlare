import os

# Currently this file is used for CNR-EXT dataset # TODO: AB: To be changed when we integrate the CNR dataset as well.
ROOT_DIR = "/home/bakr/NVFlare/examples/hello-world/parking-federated-training"
# path to your own data and coco file
train_data_dir = "/home/bakr/CNR-EXT/train"
train_coco = "/home/bakr/CNR-EXT/train/_annotations.coco.json"
val_data_dir = "/home/bakr/CNR-EXT/valid"
val_coco = "/home/bakr/CNR-EXT/valid/_annotations.coco.json"
test_data_dir = "/home/bakr/CNR-EXT/test"
test_coco = "/home/bakr/CNR-EXT/test/_annotations.coco.json"

# Batch size
train_batch_size = 8
pre_trained_model_allowed = True

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

# Three classes; Space-empty (1), Space-occupied (2)
num_classes = 3 # Number of classes + 1 (background)
num_epochs = 30

lr = 0.001
momentum = 0.9
weight_decay = 0.005

models_folder = os.path.join(ROOT_DIR, "models")

# mAP calculation configurations
mAP_detection_threshold = 0.5
mAP_val_prediction_directory = os.path.join(ROOT_DIR, "mAPInput/detection-results")
mAP_val_gt_directory = os.path.join(ROOT_DIR, "mAPInput/ground-truth")
mAP_metric_file_path = os.path.join(ROOT_DIR, "outputs/overall_trackers.pkl")
