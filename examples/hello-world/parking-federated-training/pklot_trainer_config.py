# path to your own data and coco file
train_data_dir = "/home/bakr/pklot/train"
train_coco = "/home/bakr/pklot/train/_annotations.coco.json"

# Batch size
train_batch_size = 12

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

# Three classes; Space-empty (1), Space-occupied (2)
num_classes = 4 # Number of classes + 1 (background)
num_epochs = 10

lr = 0.001
momentum = 0.9
weight_decay = 0.005