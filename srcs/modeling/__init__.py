# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .resnet34 import resnet34

def build_model(cfg):
    model = resnet34(cfg.MODEL.NUM_CLASSES)
    return model
