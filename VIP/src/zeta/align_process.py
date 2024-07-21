import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import logging
import os
from datetime import datetime
from glob import glob
import json
import gc

from zeta.loss import InfoNCELoss1, NCEContrastiveLoss

def align_modalities_process(multi_modality_model, 
                             train_loader, 
                             val_loader, 
                             num_epochs=10, 
                             learning_rate=0.0001, 
                             temperature=0.1, 
                             resume_from_checkpoint=False, 
                             checkpoint_dir='/home/bas06400/Thesis/VIP/src/align_checkpoints',
                             device=None,
                             config=None):
    """
    Train and validate a multi-modality model for aligning different modalities.

    This function performs the following steps:
    1. Sets up the training environment (optimizer, scheduler, loss function)
    2. Loads a checkpoint if resuming training
    3. Trains the model for the specified number of epochs
    4. Validates the model after each epoch
    5. Saves checkpoints and training statistics

    Args:
        multi_modality_model (nn.Module): The multi-modality model to be trained
        train_loader (DataLoader): DataLoader for the training data
        val_loader (DataLoader): DataLoader for the validation data
        num_epochs (int): Number of epochs for training
        learning_rate (float): Learning rate for the optimizer
        temperature (float): Temperature parameter for InfoNCE loss
        resume_from_checkpoint (bool): Whether to resume training from a checkpoint
        checkpoint_dir (str): Directory to save checkpoints
        device (torch.device): Device to run the training on
        config (dict): Configuration dictionary containing model and training settings

    Returns:
        None
    """
    modalities = '_'.join(config['modalities'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_filename = f"checkpoint_{modalities}_{config['encoder_model']}_{config['dataset']}_{config['split']}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    stats_path = os.path.join(checkpoint_dir, checkpoint_filename[:-4])
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    #overfit_on_one_batch = config.get('overfit_on_one_batch', False)


    # Function to find the latest checkpoint
    def find_latest_checkpoint():
        list_of_files = glob(os.path.join(checkpoint_dir, f'checkpoint_{modalities}_*.pth'))
        if list_of_files:
            return max(list_of_files, key=os.path.getctime)
        return None

    def clear_memory():
        gc.collect()
        torch.cuda.empty_cache()
    # Initialize the optimizer and loss function
    optimizer = optim.Adam(multi_modality_model.parameters(), lr=learning_rate)
    scheduler = create_scheduler(optimizer, config)
    info_nce_loss = InfoNCELoss1(temperature=temperature) 
   
    # Placeholder for best validation loss
    best_val_loss = float('inf')
    start_epoch = 0
    training_stats = {"epochs": [], "train_loss": [], "val_loss": []}

    if resume_from_checkpoint:
        latest_checkpoint_path = find_latest_checkpoint()
        if latest_checkpoint_path:
            logging.info(f"Resuming from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=f"cuda:{device}")
            multi_modality_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                logging.info("Warning: Scheduler state not found in checkpoint! Scheduler will be initialized without state.")

            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            training_stats = checkpoint.get('training_stats', training_stats)
        else:
            logging.info("No checkpoint found, starting training from scratch.")

    logging.info("Starting training loop")

    #if overfit_on_one_batch:
    #    single_batch_data, _ = next(iter(train_loader))
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        individual_losses = {}

        multi_modality_model.train()
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Training")

        for step, (batch_data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            
            
            #if overfit_on_one_batch: # this is inefficent but conveniant
            #    del batch_data
            #    batch_data = single_batch_data
            
            #allowing mixed encoders
            def preprocess_for_clip_vip(data, modality):
                return data  

            def preprocess_for_mae(data, modality):
                return data.permute(0, 2, 1, 3, 4)  
            
            def preprocess_for_maeps(data, modality):
                return data.view(data.size(0), data.size(1), -1) 

            def preprocess_for_omnivore(data, modality):
                if modality =='depth':
                    # Concatenate the first channel with the original tensor along the channel dimension to add a fourth channel
                    return torch.cat((data, data[:, :, 0:1, :, :]), 2).permute(0, 2, 1, 3, 4)
                else:
                    return data.permute(0, 2, 1, 3, 4)

            # Mapping encoders to their preprocessing functions
            preprocessing_map = {
                'CLIP-VIP': preprocess_for_clip_vip,
                'MAE': preprocess_for_mae,
                'OMNIVORE': preprocess_for_omnivore,
                'DINO': preprocess_for_clip_vip, # DINO requires the same structure
                'MAEPS': preprocess_for_maeps
            }
            embeddings = []
            for modality, data in batch_data.items():
                # Determine the encoder to use for this modality
                if config['encoder_model'] == 'MIX':
                    encoder = config['modalities_encoders'].get(modality)
                else:
                    encoder = config['encoder_model']  # Use the unified encoder for all modalities
                
                # Check if the modality is supported and preprocess data accordingly
                if modality in multi_modality_model.module.modalities_encoders:
                    data = data.cuda(device, non_blocking=True)
                    if encoder in preprocessing_map:
                        # Apply preprocessing specific to the selected encoder
                        data = preprocessing_map[encoder](data, modality)
                    
                    # Forward the preprocessed data through the encoder
                    # Assuming forward_encoder can handle different encoder types
                    embeddings.append(multi_modality_model.module.forward_encoder(modality, data))
                else:
                    logging.warn(f"Unsupported modality or encoder: {modality}, Encoder: {encoder}")

            # Calculate the loss across all pairs of modalities
            loss, loss_dict = info_nce_loss(*embeddings)
            loss = loss / gradient_accumulation_steps

            
            #if overfit_on_one_batch:
              #  logging.info(f"loss on overfiting batch {loss.item()}")

            # Accumulate individual losses for logging
            for key, value in loss_dict.items():
                if key not in individual_losses:
                    individual_losses[key] = []
                individual_losses[key].append(value)
            
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()  # Perform an optimization step
                optimizer.zero_grad()  # Reset gradients

                clear_memory()
            epoch_loss += loss.item()
            
            

        # Log the average individual losses
        for key, values in individual_losses.items():
            avg_loss = sum(values) / len(values)
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], {key} Avg Loss: {avg_loss:.4f}")


        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation loop
        multi_modality_model.eval()
        val_loss = 0.0
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Validation")

        with torch.no_grad():
            for batch_data, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                
                #allowing mixed encoders
                def preprocess_for_clip_vip(data, modality):
                    return data  

                def preprocess_for_mae(data, modality):
                    return data.permute(0, 2, 1, 3, 4)  
                
                def preprocess_for_maeps(data, modality):
                    return data.view(data.size(0), data.size(1), -1) 

                def preprocess_for_omnivore(data, modality):
                    if modality =='depth':
                        # Concatenate the first channel with the original tensor along the channel dimension to add a fourth channel
                        return torch.cat((data, data[:, :, 0:1, :, :]), 2).permute(0, 2, 1, 3, 4)
                    else:
                        return data.permute(0, 2, 1, 3, 4)

                # Mapping encoders to their preprocessing functions
                preprocessing_map = {
                    'CLIP-VIP': preprocess_for_clip_vip,
                    'MAE': preprocess_for_mae,
                    'OMNIVORE': preprocess_for_omnivore,
                    'DINO': preprocess_for_clip_vip, # DINO requires the same structure
                    'MAEPS': preprocess_for_maeps
                }
                embeddings = []
                for modality, data in batch_data.items():
                    # Determine the encoder to use for this modality
                    if config['encoder_model'] == 'MIX':
                        encoder = config['modalities_encoders'].get(modality)
                    else:
                        encoder = config['encoder_model']  # Use the unified encoder for all modalities
                    
                    # Check if the modality is supported and preprocess data accordingly
                    if modality in multi_modality_model.module.modalities_encoders:
                        data = data.cuda(device, non_blocking=True)
                        if encoder in preprocessing_map:
                            # Apply preprocessing specific to the selected encoder
                            data = preprocessing_map[encoder](data, modality)
                        
                        # Forward the preprocessed data through the encoder
                        # Assuming forward_encoder can handle different encoder types
                        embeddings.append(multi_modality_model.module.forward_encoder(modality, data))
                    else:
                        logging.warn(f"Unsupported modality or encoder: {modality}, Encoder: {encoder}")
                    # Calculate the loss across all pairs of modalities
                loss, loss_dict = info_nce_loss(*embeddings)
                val_loss += loss.item()

                clear_memory()
            avg_val_loss = val_loss / len(val_loader)
            
            scheduler.step()

            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

            training_stats["epochs"].append(epoch + 1)
            training_stats["train_loss"].append(epoch_loss / len(train_loader))
            training_stats["val_loss"].append(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': multi_modality_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'best_val_loss': best_val_loss,
                    # Add any other things you need to save
                }
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Best val loss {best_val_loss}")
                logging.info(f"New best model saved at epoch {epoch+1}")
        # Save final training statistics
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f)
        logging.info(f"Training statistics saved to {stats_path}")
    logging.info("Training complete!")



def create_scheduler(optimizer, config):
    scheduler_config = config.get('scheduler_config', {})
    scheduler_type = scheduler_config.get('type', 'step')
    scheduler_params = scheduler_config.get('params', {})

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler

@torch.no_grad()
def eval_loss_process(multi_modality_model, 
                             train_loader, 
                             val_loader, 
                             num_epochs=10, 
                             learning_rate=0.0001, 
                             temperature=0.1, 
                             resume_from_checkpoint=False, 
                             checkpoint_dir='/home/bas06400/Thesis/VIP/src/align_checkpoints',
                             device=None,
                             config=None):
    """
    Evaluate loss for unaligend clip vip encoders

    :param multi_modality_model: The multi-modality model to be trained.
    :param train_loader: DataLoader for the training data.
    :param val_loader: DataLoader for the validation data.
    :param modalities_encoders: Dictionary of modality encoders.
    :param num_epochs: Number of epochs for training.
    :param learning_rate: Learning rate for the optimizer.
    :param temperature: Temperature parameter for InfoNCE loss.
    """
    modalities = '_'.join(config['modalities'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_filename = f"checkpoint_{modalities}_{config['split']}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    stats_path = os.path.join(checkpoint_dir, checkpoint_filename[:-4])
    



    # Initialize the optimizer and loss function
    optimizer = optim.Adam(multi_modality_model.parameters(), lr=learning_rate)
    scheduler = create_scheduler(optimizer, config)
    info_nce_loss = InfoNCELoss1(temperature=temperature) #InfoNCE(temperature=temperature, reduction='mean', negative_mode='paired')

    # Placeholder for best validation loss
    best_val_loss = float('inf')
    start_epoch = 0
    training_stats = {"epochs": [], "train_loss": [], "val_loss": []}

    

    logging.info("Starting training loop")
    logging.info("Epoch set to one because it is just eval")
    
    # Training loop
    for epoch in range(0, 1):
        epoch_loss = 0.0
        optimizer.zero_grad()
        individual_losses = {}

        multi_modality_model.train()
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Training")

        for step, (batch_data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            
        

            embeddings = []
            for modality in batch_data.keys():
                if modality in multi_modality_model.module.modalities_encoders:
                    data = batch_data[modality].cuda(device)
                    #if modality != 'rgb':
                        
                     #   data = data.expand(-1, -1, 3, -1, -1)
                       
                    embeddings.append(multi_modality_model.module.forward_encoder(modality, data))

            # Calculate the loss across all pairs of modalities
            loss, loss_dict = info_nce_loss(*embeddings)


            # Accumulate individual losses for logging
            for key, value in loss_dict.items():
                if key not in individual_losses:
                    individual_losses[key] = []
                individual_losses[key].append(value)
            
        

            epoch_loss += loss.item()
            
            

        # Log the average individual losses
        for key, values in individual_losses.items():
            avg_loss = sum(values) / len(values)
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], {key} Avg Loss: {avg_loss:.4f}")


        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation loop
        multi_modality_model.eval()
        val_loss = 0.0
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Validation")

        
        for batch_data, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
        
            embeddings = []
            for modality in batch_data.keys():
                if modality in multi_modality_model.module.modalities_encoders:
                    data = batch_data[modality].cuda(device)
                    #if modality != 'rgb':
                    #    data = data.expand(-1, -1, 3, -1, -1)
                    embeddings.append(multi_modality_model.module.forward_encoder(modality, data))

            # Calculate the loss across all pairs of modalities
            loss, loss_dict = info_nce_loss(*embeddings)
            val_loss += loss.item()

            
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        training_stats["epochs"].append(epoch + 1)
        training_stats["train_loss"].append(epoch_loss / len(train_loader))
        training_stats["val_loss"].append(avg_val_loss)
            
        # Save final training statistics
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f)
        logging.info(f"Training statistics saved to {stats_path}")
    logging.info("Training complete!")
