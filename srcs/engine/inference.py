# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
"""
@modified: thanh
@contact: nguyenvanthanhhust@gmail.com
"""

import torch
import logging

def inference(
        cfg,
        model,
        val_loader
):
    device = cfg.MODEL.DEVICE
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0
    with torch.no_grad(): # Disable gradient calculation for validation
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            val_total_samples += labels.size(0)
            val_correct_predictions += (predicted == labels).sum().item()

        val_epoch_accuracy = val_correct_predictions / val_total_samples
        print(f"Val Accuracy: {val_epoch_accuracy:.4f}")