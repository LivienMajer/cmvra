import os
import logging
from datetime import datetime
import logging
import torch.optim as optim
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from glob import glob


import random

from zeta.loss import mse_loss, create_scheduler



def generate_random_mask(batch_size, num_patches, mask_ratio):
    """
    Generate a random boolean mask for MAE training.
    
    Args:
        batch_size (int): Number of samples in the batch.
        num_patches (int): Total number of patches in each sample.
        mask_ratio (float): Ratio of patches to mask.
    
    Returns:
        torch.Tensor: Boolean mask of shape (batch_size, num_patches).
    """
    num_masked = int(mask_ratio * num_patches)
    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool)
    for i in range(batch_size):
        masked_indices = random.sample(range(num_patches), num_masked)
        mask[i, masked_indices] = True
    return mask


def find_latest_checkpoint(checkpoint_dir, modalities):
    """Find the most recent checkpoint file for given modalities."""
    list_of_files = glob(os.path.join(checkpoint_dir, f'checkpoint_{modalities}_*.pth'))
    if list_of_files:
        return max(list_of_files, key=os.path.getctime)
    return None



def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, training_stats, checkpoint_dir, timestamp, config):
    modalities = '_'.join(config['modalities'])
    checkpoint_filename = f"checkpoint_{modalities}_{config['split']}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'training_stats': training_stats
    }, checkpoint_path)
    logging.info(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
    return checkpoint_filename[:-4]



def train(model, train_loader, optimizer, scheduler, val_loader, test_loader, device, config):
    """
    Main training loop for the MAE model. Fixed on 16 input frames with size 224x224
    
    Handles training, validation, and checkpointing for each epoch.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.train()
    num_patches = 1568
    best_val_loss = float('inf')
    training_stats = {"epochs": [], "train_loss": [], "val_loss": []}
    checkpoint_dir = config['cktp_dir']

    for epoch in range(config['epochs']):
        total_loss = 0
        for batch_data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            rgb_input = batch_data[config['modalities'][0]].to(device)
            optimizer.zero_grad()

            batch_size = rgb_input.size(0)
            mask = generate_random_mask(batch_size, num_patches, config['mask_ratio']).to(device)
            reconstruction = model(rgb_input.permute(0,2,1,3,4), mask)
            
            loss = mse_loss(reconstruction.view(-1,3,16,224,224), batch_data[config['modalities'][0]].permute(0,2,1,3,4).to(device))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch: {epoch}, Average Training Loss: {avg_loss}")

        val_loss = validate(model, val_loader, device, config)
        logging.info(f"Epoch: {epoch}, Validation Training Loss: {val_loss}")

        scheduler.step(val_loss)
        training_stats["epochs"].append(epoch)
        training_stats["train_loss"].append(avg_loss)
        training_stats["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stats_filename = save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, training_stats, checkpoint_dir, timestamp, config)



    test_loss = validate(model, test_loader, device, config)
    logging.info(f"Final Test Loss: {test_loss}")
    with open(os.path.join(checkpoint_dir, stats_filename), 'w') as f:
        json.dump(training_stats, f, indent=4)
    logging.info(f"Training statistics saved to {stats_filename}")

def validate(model, val_loader, device, config):
    model.eval()
    num_patches = 1568
    total_val_loss = 0
    with torch.no_grad():
        for batch_data, _ in tqdm(val_loader, desc="Validating", leave=False):
            input = batch_data[config['modalities'][0]].to(device)
            batch_size = input.size(0)
            mask = generate_random_mask(batch_size, num_patches, config['mask_ratio']).to(device)
            reconstruction = model(input.permute(0,2,1,3,4), mask)
            loss = mse_loss(reconstruction.view(-1,3,16,224,224), batch_data[config['modalities'][0]].permute(0,2,1,3,4).to(device))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss



   
def mae_training(model, train_data, val_data, test_data, device, config):

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = create_scheduler(optimizer, config)

    
    # Start training
    train(model, train_data, optimizer, scheduler, val_data, test_data, device, config)

 