import math

import cv2
import numpy as np
import torch
from albumentations import (
    Compose
)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_transforms():
    list_transforms = []

    list_transforms.extend([])

    list_transforms.extend([ToTensorV2(), ])
    list_trfms = Compose(list_transforms)
    return list_trfms


class data_batch(Dataset):

    def __init__(self, img_list):
        self.transforms = get_transforms()
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]

        augmented = self.transforms(image=img)
        img = augmented['image']

        image = torch.cat((img, img), 0)

        return image


class data_batch2(Dataset):

    def __init__(self, raw_img, cut_size, overlap):
        self.transforms = get_transforms()

        shapes = raw_img.shape
        x_nums = math.ceil(shapes[0] / (cut_size - overlap))
        y_nums = math.ceil(shapes[1] / (cut_size - overlap))
        self.x_list = []
        self.y_list = []
        self.img_list = []
        for x_temp in range(x_nums):
            for y_temp in range(y_nums):
                x_begin = max(0, x_temp * (cut_size - overlap))
                y_begin = max(0, y_temp * (cut_size - overlap))
                x_end = min(x_begin + cut_size, shapes[0])
                y_end = min(y_begin + cut_size, shapes[1])
                i = raw_img[x_begin: x_end, y_begin: y_end]
                self.x_list.append(x_begin)
                self.y_list.append(y_begin)
                self.img_list.append(i)

        self.ori_size = []

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.ori_size.append([img.shape[0], img.shape[1]])

        pad_img = np.full((256, 256, 3), 0, dtype='uint8')
        pad_img[:img.shape[0], :img.shape[1], :] = img

        augmented = self.transforms(image=pad_img)
        pad_img = augmented['image']

        image = torch.cat((pad_img, pad_img), 0)

        return image

    def get_list(self):
        return (self.x_list, self.y_list, self.ori_size)
