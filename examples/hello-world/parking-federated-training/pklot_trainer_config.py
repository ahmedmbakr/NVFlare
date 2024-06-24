import os

ROOT_DIR = "/home/bakr/NVFlare/examples/hello-world/parking-federated-training"
# path to your own data and coco file
PKLOT_DATA_DIR = "/home/bakr/pklot"
train_data_dir = "/home/bakr/pklot/train"
train_coco = "/home/bakr/pklot/train/_annotations.coco.json"
val_data_dir = "/home/bakr/pklot/valid"
val_coco = "/home/bakr/pklot/valid/_annotations.coco.json"
test_data_dir = "/home/bakr/pklot/test"
test_coco = "/home/bakr/pklot/test/_annotations.coco.json"

# Batch size
train_batch_size = 12

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

# Three classes; Space-empty (1), Space-occupied (2)
num_classes = 3 # Number of classes + 1 (background)
num_epochs = 10

lr = 0.001
momentum = 0.9
weight_decay = 0.005

models_folder = os.path.join(ROOT_DIR, "models")

# mAP calculation configurations
mAP_val_prediction_directory = os.path.join(ROOT_DIR, "input/detection-results")
mAP_val_gt_directory = os.path.join(ROOT_DIR, "input/ground-truth")
mAP_metric_file_path = os.path.join(ROOT_DIR, "outputs/metrics.p")

# Split by parking lot
SPLIT_BY_PARKING_LOT_ROOT_DIR = PKLOT_DATA_DIR
