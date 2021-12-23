import os
from skimage import io
import cv2
import torch
from torch.utils.data import Dataset
from albumentations import (HorizontalFlip, Normalize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2


def get_transforms():
    list_transforms = []

    list_transforms.extend(
        [
            # HorizontalFlip(p=0.5),
            # GaussNoise(p=0.4),
        ])

    # list_transforms.extend([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),])

    list_transforms.extend(
        [
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(),
        ])
    list_trfms = Compose(list_transforms)
    return list_trfms


class data_batch(Dataset):

    def __init__(self, img_list):
        self.transforms = get_transforms()
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        ### data1
        img = self.img_list[idx]
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = transform.resize(img,(256,256))

        augmented = self.transforms(image=img)
        img = augmented['image']

        image = torch.cat((img, img), 0)

        return image