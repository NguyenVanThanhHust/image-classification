# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing
import torch

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.RandomResizedCrop(cfg.INPUT.SIZE_TRAIN), # Randomly crop and resize to 224x224
            T.RandomHorizontalFlip(), # Randomly flip horizontally
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly change brightness, contrast, saturation, hue
            T.ToTensor(), # Convert PIL Image or numpy.ndarray to tensor
            normalize_transform # Normalize with ImageNet mean and std
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
