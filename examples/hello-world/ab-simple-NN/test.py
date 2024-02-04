import os.path

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import random_split

data_path="~/data"


# Create Cifar10 dataset for training.
transforms = Compose(
    [
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_dataset = CIFAR10(root=data_path, transform=transforms, download=True, train=True)

total_size = len(train_dataset)
first_split_size = total_size // 2
second_split_size = total_size - first_split_size

# Split the dataset
first_split_dataset, second_split_dataset = random_split(train_dataset, [first_split_size, second_split_size])
train_loader = DataLoader(first_split_dataset, batch_size=4, shuffle=True)
print(f"Shape of the whole dataset: {train_dataset.data.shape}")
print(f"Number of samples from the whole data: {len(first_split_dataset.indices)}")
n_iterations = len(train_loader) # Number 
print(f"Number of iterations: {n_iterations}")
print("Finished!")
