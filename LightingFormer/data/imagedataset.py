import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from basicsr.utils.registry import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class ImageDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # Transforms for low resolution images and high resolution images
        root = opt['dataroot_gt']
        gt_size = self.opt['gt_size']
        self.pre_trans = transforms.Compose([
            # transforms.CenterCrop(self.crop_size),
            transforms.RandomCrop(gt_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
        ])

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((gt_size // 4, gt_size // 4), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = self.pre_trans(img)
        img_lq = self.lr_transform(img)
        img_gt = self.hr_transform(img)

        return {'lq': img_lq, 'gt': img_gt}

    def __len__(self):
        return len(self.files)