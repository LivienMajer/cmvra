
from omnimae import vit_base_mae_finetune_ssv2, vit_base_mae_pretraining, vit_large_mae_finetune_ssv2, vit_base_mae_finetune_in1k
model = vit_base_mae_finetune_in1k(pretrained=True)

import torch
import torch.nn as nn
N, C, T, H, W = 1, 3, 16, 224, 224
video_input = torch.randn(N, C, T, H, W)
patch_size_T, patch_size_H, patch_size_W = 2, 16, 16  # Patch siz
# Calculate number of patches
num_patches_T = T // patch_size_T
num_patches_H = H // patch_size_H
num_patches_W = W // patch_size_W

# Total number of patches
total_patches = num_patches_T * num_patches_H * num_patches_W

# Create a mask tensor where no patches are masked
mask = torch.zeros(N, total_patches, dtype=torch.bool)
model.head = nn.Linear(768, 34, bias = False)
output = model(video_input)
print("Output shape:", output.shape)

import sys
sys.path.append("/home/bas06400/Thesis/VIP/src/")

from zeta.data_loader import load_dataloaders

data_root = "/net/polaris/storage/deeplearning/ntu"
modalities = ["rgb"]
batch_size = 8
num_workers = 10
pin_memory = True
split = "0"
random_sample = False
config = {'dataset':'DAA'}
train_data, val_data, test_data = load_dataloaders(data_root=data_root,
                                                       modalities=modalities,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory,
                                                        split=split,
                                                        random_sample=random_sample,
                                                        config=config)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import gc
from tqdm import tqdm
import os
import json 
import logging
import math



def train_classefier_process(multi_modality_model, train_loader, val_loader, test_loader):
    # Extracting configuration parameters
    num_epochs = 10
    learning_rate = 0.0001
    device = 'cuda:3'
    modalities = ['rgb']
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(multi_modality_model.parameters(), lr=learning_rate)

    # to do implement learning rate sceduler
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize Step LR learning rate scheduler
    step_size = int(math.floor(num_epochs * 0.4))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    best_val_loss = float('inf')

    # Dictionary to hold training stats for each modality
    training_stats = {"epochs": [], "train_loss": {}, "val_loss": {}, "test_loss": {}}
    start_epoch = 0
    

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Train and validate for each epoch
        train_losses, train_accuracies = train_epoch(multi_modality_model, device, train_loader, criterion, optimizer, epoch, num_epochs)
        val_losses, val_accuracies = evaluate_model(multi_modality_model, device, val_loader, criterion, epoch, num_epochs)

        lr_scheduler.step()

        # Update training stats
        training_stats["epochs"].append(epoch + 1)
        training_stats["train_loss"][epoch + 1] = train_losses
        training_stats["val_loss"][epoch + 1] = val_losses

        # Checkpoint logic based on overall validation loss (modify as needed for modality-specific checkpoints)
        overall_val_loss = sum(val_losses.values()) / len(val_losses)  # Average validation loss across modalities
        

        # Evaluate on the test set
        test_losses, test_accuracies = evaluate_model(multi_modality_model, device, test_loader, criterion, epoch, num_epochs)
        training_stats["test_loss"][epoch + 1] = test_losses
        print(f"Test Loss: {test_losses}, Test Accuracy: {test_accuracies}")
    

    print("Training, validation, and testing complete!")



def train_epoch(model, device, train_loader, criterion, optimizer, epoch, num_epochs):
    epoch_losses = {modality: 0.0 for modality in modalities}
    epoch_accuracies = {modality: 0.0 for modality in modalities}

    model.train()
    for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        for modality in batch_data:
            
            data = batch_data['rgb'].cuda(device)
            labels = batch_labels.cuda(device)
            #print(modality)
            optimizer.zero_grad()
            data = data.permute(0,2,1,3,4)
            outputs = model(data)
            #print(f"Outputs {outputs}")
            #print(f"Labels {labels}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = compute_accuracy(outputs, labels)
            #print(f"accuracy {accuracy}")
            epoch_losses[modality] += loss.item()
            epoch_accuracies[modality] += accuracy

    # Calculate average loss and accuracy for each modality
    avg_losses = {modality: epoch_losses[modality] / len(train_loader) for modality in epoch_losses}
    avg_accuracies = {modality: epoch_accuracies[modality] / len(train_loader) for modality in epoch_accuracies}

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    for modality in modalities:
        print(f"Modality: {modality}, Loss: {avg_losses[modality]:.4f}, Accuracy: {avg_accuracies[modality]:.4f}")

    return avg_losses, avg_accuracies

def evaluate_model(model, device, loader, criterion, epoch, num_epochs):
    val_losses = {modality: 0.0 for modality in modalities}
    val_accuracies = {modality: 0.0 for modality in modalities}

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(loader, desc=f"Validation/Test Epoch {epoch+1}/{num_epochs}"):
            for modality in batch_data:
                
                data = batch_data['rgb'].cuda(device)
                labels = batch_labels.cuda(device)
                data = data.permute(0,2,1,3,4)
                outputs = model(data)
                loss = criterion(outputs, labels)

                val_losses[modality] += loss.item()
                val_accuracies[modality] += compute_accuracy(outputs, labels)

            

    # Calculate average loss and accuracy for each modality
    avg_val_losses = {modality: val_losses[modality] / len(loader) for modality in val_losses}
    avg_val_accuracies = {modality: val_accuracies[modality] / len(loader) for modality in val_accuracies}

    print(f"Validation/Test Epoch [{epoch+1}/{num_epochs}]")
    for modality in modalities:
        print(f"Modality: {modality}, Loss: {avg_val_losses[modality]:.4f}, Accuracy: {avg_val_accuracies[modality]:.4f}")

    return avg_val_losses, avg_val_accuracies


def compute_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)
device = 'cuda:3'
model = model.cuda(device)

train_classefier_process(model, train_data, val_data, test_data)