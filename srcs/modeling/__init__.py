# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from os.path import join, isfile, isdir
from .resnet34 import resnet34

def build_model(cfg):
    if isfile(cfg.MODEL.PRETRAINED_MODEL_PATH):
        model = resnet34(cfg.MODEL.NUM_CLASSES, cfg.MODEL.PRETRAINED_MODEL_PATH)
    else:
        model = resnet34(cfg.MODEL.NUM_CLASSES)
    return model
