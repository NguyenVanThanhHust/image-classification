# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from os.path import join
from torch.utils import data

from .datasets.mini_imagenet import CustomImageFolder
from .transforms import build_transforms


def build_dataset(cfg, transforms, is_train=True):
    if cfg.INPUT.NAME == "mini_imagenet":
        if is_train:
            data_dir = join(cfg.INPUT.DATA_DIR, "train")
        else:
            data_dir = join(cfg.INPUT.DATA_DIR, "val")
        datasets = CustomImageFolder(root=data_dir, transform=transforms)
    else:
        print(f"Unrecognize dataset name, get {cfg.INPUT.NAME}")
        raise NotImplementedError
    return datasets


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
