import os

ROOT_DIR = "/home/bakr/NVFlare/examples/hello-world/parking-federated-training"
# path to your own data and coco file
PKLOT_DATA_DIR = "/home/bakr/pklot/UFPR04"
train_data_dir = "/home/bakr/pklot/UFPR04/train"
train_coco = "/home/bakr/pklot/UFPR04/train/_annotations.coco.json"
val_data_dir = "/home/bakr/pklot/UFPR04/valid"
val_coco = "/home/bakr/pklot/UFPR04/valid/_annotations.coco.json"

test_coco_paths_list = ["/home/bakr/pklot/PUCPR/test/_annotations.coco.json", "/home/bakr/pklot/UFPR04/test/_annotations.coco.json", "/home/bakr/pklot/UFPR05/test/_annotations.coco.json", "/home/bakr/CNR-EXT/test/_annotations.coco.json"]
test_names = ["PUCPR", "UFPR04", "UFPR05", "CNR-EXT"]

# Batch size
train_batch_size = 12
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

# Split by parking lot
SPLIT_BY_PARKING_LOT_ROOT_DIR = PKLOT_DATA_DIR
