import os.path

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torch.utils.data import random_split
import torchvision

data_path="~/data/gtsrb/GTSRB"
users_split = 2
batch_size = 4

# Create Cifar10 dataset for training.
transforms = Compose(
            [
                Resize([112, 112]),
                ToTensor()
            ]
        )
train_data_path = os.path.join(data_path, "Training")
_train_dataset = torchvision.datasets.ImageFolder(root = train_data_path, transform = transforms)

# Calculate the size for each split
total_size = len(_train_dataset)
first_split_size = total_size // users_split
second_split_size = total_size - first_split_size

# Split the dataset
first_split_dataset, second_split_dataset = random_split(_train_dataset, [first_split_size, second_split_size])
is_first_client = "site-1" in os.path.abspath(__file__)
print(f"The initialization is running from this folder: {os.path.abspath(__file__)} and the value of is_first_client is: {is_first_client}")
_train_dataset = first_split_dataset if is_first_client else second_split_dataset

_train_loader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True)
_n_iterations = len(_train_loader)
print(f"Number of iterations: {_n_iterations}")
print(f"Shape of the whole dataset: {_train_dataset.dataset.data.shape}")
print(f"Number of samples from the whole data: {len(_train_dataset.indices)} = Number of iterations ({_n_iterations}) * batch size ({batch_size})")
