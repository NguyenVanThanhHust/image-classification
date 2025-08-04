# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
"""
@modified: thanh
@contact: nguyenvanthanhhust@gmail.com
"""
import os
import logging
import torch

def train_one_epoch():
    return

def evaluate():
    return

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        writer,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    device = cfg.MODEL.DEVICE
    num_epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("main_process.train")
    logger.info("Start training")

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    best_acc = 0
    device = torch.device(device)
    model = model.to(device)
    train_global_step = 1
    val_global_step = 1

    for epoch in range(num_epochs):
        model.train()
        criterion.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            writer.add_scalar('Loss/train_iteration', loss.item(), train_global_step)
            writer.add_scalar('Accuracy/train_iteration', correct_predictions/total_samples, train_global_step)
            train_global_step += 1
            if i % log_period == 0:
                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_accuracy = correct_predictions / total_samples
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss.item():.4f}, Train Accuracy: {epoch_accuracy:.4f}")

        
        if epoch % checkpoint_period == 0:
            # Validation Phase
            model.eval() # Set model to evaluation mode
            val_loss = 0.0
            val_correct_predictions = 0
            val_total_samples = 0

            with torch.no_grad(): # Disable gradient calculation for validation
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total_samples += labels.size(0)
                    val_correct_predictions += (predicted == labels).sum().item()
                    writer.add_scalar('Loss/val_iteration', val_loss, val_global_step)
                    writer.add_scalar('Accuracy/val_iteration', correct_predictions/val_total_samples, val_global_step)
                    val_global_step += 1

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_accuracy = val_correct_predictions / val_total_samples
            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}")
            if val_epoch_accuracy > best_acc:
                best_acc = val_epoch_accuracy
                output_model_path = os.path.join(output_dir, f"{cfg.MODEL.MODEL_NAME}_{epoch}_{best_acc:.4f}.pth")
                torch.save(model.state_dict(), output_model_path)
                output_model_path = os.path.join(output_dir, f"{cfg.MODEL.MODEL_NAME}_best.pth")
                torch.save(model.state_dict(), output_model_path)
                print(f"Saving model at epoch {epoch} at {output_model_path}")

    writer.flush()