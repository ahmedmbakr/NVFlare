# The base of this class was taken from the following source: https://github.com/tomlawrenceuk/GTSRB-Dataloader/blob/master/gtsrb_dataset.py

import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class GTSRB_TestDataLoader(Dataset):

    def __init__(self, dataset_root_dir, transform=None):

        self.test_data_dir = dataset_root_dir
        self.csv_file_name = "GT-final_test.csv"

        csv_file_path = os.path.join(self.test_data_dir, '..', self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        csv_row = self.csv_data.iloc[idx, 0].split(';')
        image_name = csv_row[0]
        classId = torch.tensor(int(csv_row[7]))
        img_path = os.path.join(self.test_data_dir, 'Final_Test/Images', image_name) # AB: I tried this but it did not work.
        # print(f"Path: {img_path}, exist: {os.path.isfile(img_path)}") # AB: I added this line to see if the path is correct
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, classId