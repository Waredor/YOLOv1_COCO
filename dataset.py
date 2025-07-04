import pandas as pd
import math
import torch
from PIL import Image
from torch.utils.data import Dataset
import os

class CUBSATDataset(Dataset):
    def __init__(self, data_csv, img_dir, s=7, b=2, c=3, transform=None):
        '''
        Parameters:
            data_csv (str): csv file path
            img_dir (str): Image directory path
            s (int): Grid cell size (ex. 7x7)
            b (int): Number of bounding boxes per one cell
            c (int): Number of classes
            transform (compose): A list of transforms
        '''

        self.df = pd.read_csv(data_csv)
        self.img_dir = img_dir
        self.s = s
        self.b = b
        self.c = c
        self.transform = transform
        empty_arr = torch.zeros(size=(13,), dtype=torch.float32)
        self.labels_arr = torch.zeros(size=(7, 7, 13), dtype=torch.float32)
        self.labels_arr[:, :] = empty_arr

    def __len__(self):
        '''
        Return the length of the dataset
        '''
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Returns one data item with idx

        Parameters:
            idx (int): Index number of data

        Returns:
            tuple (image, label) where label is in format (S, S, 5 * B + C).
            Note that last 5 elements of label will NOT be used.
        '''

        # Path for image and label
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])

        # Read image
        image_pil = Image.open(img_path)
        image = torch.tensor(image_pil, dtype=torch.float32)

        #csv_file data preprocessing
        x_max_norm = round((self.df.iloc[idx, 4] / 1000) * 448, 0)
        y_max_norm = round((self.df.iloc[idx, 5] / 1000) * 448, 0)
        x_min_norm = round((self.df.iloc[idx, 2] / 1000) * 448, 0)
        y_min_norm = round((self.df.iloc[idx, 3] / 1000) * 448, 0)

        x = (x_max_norm - x_min_norm) // 2
        y = (y_max_norm - y_min_norm) // 2
        i = math.floor(y / 64)
        j = math.floor(x / 64)
        x_cell = i * 64
        y_cell = j * 64
        x = (x - x_cell) / 64
        y = (y - y_cell) / 64
        w = (x_max_norm - x_min_norm) / 448
        h = (y_max_norm - y_min_norm) / 448
        one_hot = pd.get_dummies(self.df.iloc[idx, 2], prefix='class').astype(int)
        concat_df = pd.concat([self.df.iloc[idx], one_hot], axis=1)
        p1 = concat_df.iloc[idx, 6]
        p2 = concat_df.iloc[idx, 7]
        p3 = concat_df.iloc[idx, 8]
        label = self.labels_arr
        label[i, j] = torch.tensor(data=[x, y, w, h, 1, p1, p2, p3], dtype=torch.float32)

        return image, label