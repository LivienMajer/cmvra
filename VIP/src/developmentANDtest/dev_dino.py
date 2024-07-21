import logging
from functools import partial
from omnimae import make_conv_or_linear, reshape_and_init_as_mlp
import torch
import torch.nn as nn
from omnivore import omnivore_swinB_imagenet21k
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from omegaconf import OmegaConf
from timm.models.layers import trunc_normal_
from torch.hub import load_state_dict_from_url
import random
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to save the RGB encoder weights
def save_best_model(model, best_val_loss, current_val_loss, epoch, save_dir='model_weights', filename='omnivore.pth'):
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        torch.save(model.state_dict(), save_path)  # Save model state_dict instead of the entire model for best practice
        logger.info(f"Saved new best RGB encoder model at epoch {epoch} to {save_path}")
    return best_val_loss

# Assuming CrossEntropyLoss is suitable for your model's output
def balanced_cross_entropy_loss(output, target, weights=None):
    if weights is not None:
        
        loss = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(output.device))
    else:
        loss = nn.CrossEntropyLoss()
    return loss(output, target)

def train(model, train_loader, optimizer, scheduler, num_epochs, val_loader, device='cuda:2', class_weights=None):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch_data['depth'].to(device), labels.to(device)
            optimizer.zero_grad()
        

            #print(expanded_inputs.shape)
            #print(inputs.expand(-1,-1, 4, -1, -1).shape)
            outputs = model(inputs)
            #print(len(class_weights))
            loss = balanced_cross_entropy_loss(outputs, labels, weights=class_weights)
            
            _, predicted = torch.max(outputs.data, 1)
            total_loss += loss.item()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        logger.info(f"Epoch: {epoch}, Average Training Loss: {avg_loss}, Training Accuracy: {accuracy}")

        if val_loader:
            model.eval()
            val_loss, val_accuracy = validate(model, val_loader, device)
            scheduler.step(val_loss)
            best_val_loss = save_best_model(model, best_val_loss, val_loss, epoch)
            logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

def validate(model, val_loader, device='cuda:2'):
    total_val_loss = 0
    total_correct = 0
    num_batches = 0

    with torch.no_grad():
        for batch_data, labels in val_loader:
            inputs, labels = batch_data['depth'].to(device), labels.to(device)

        
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            num_batches += 1

    avg_val_loss = total_val_loss / num_batches
    avg_accuracy = total_correct / (num_batches * labels.size(0))
    return avg_val_loss, avg_accuracy

def calculate_class_weights(train_loader):
    class_counts = 34 *[0]  # Assuming binary classification
    for _, labels in train_loader:
        # Assuming labels are 0 or 1 for binary classification
        # Update counts based on labels in the batch
        for label in labels:
            class_counts[label] += 1
    
    # Calculate weights inversely proportional to class frequencies
    total_counts = sum(class_counts)
    class_weights = [class_counts[i] / total_counts for i in range(len(class_counts))]
    return class_weights

import torch
import torch.nn as nn
import torch.nn.functional as F



class DINOVforIR(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, freeze_dino=False):
        super(DINOVforIR, self).__init__()
        # Instantiate the DinoVisionTransformer model
        self.vision_transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # A linear layer for dimensionality reduction
        if freeze_dino:
            logger.info('frozen dinov2')
            for param in self.vision_transformer.parameters():
                param.requires_grad = False
                
        self.dim_reduction = nn.Linear(768, embedding_dim)
        # Classifier layer
        self.classifier = nn.Linear(embedding_dim, num_classes)
        # Ensure the vision transformer model is in evaluation mode if not training
        self.vision_transformer.eval()

    def forward(self, video_frames, return_embedding=False):
        """
        video_frames: a tensor of shape (B, T, C, H, W)
        return_embedding: if True, return the clip embedding instead of the classification result
        """
        batch_size, num_frames, C, H, W = video_frames.size()
        # Reshape to process all frames at once
        video_frames = video_frames.view(batch_size * num_frames, C, H, W)
        # Compute frame embeddings
        frame_embeddings = self.vision_transformer(video_frames)
        # Reshape back to (B, T, embedding_dim)
        frame_embeddings = frame_embeddings.view(batch_size, num_frames, -1)
        # Mean pooling across frames
        clip_embedding = torch.mean(frame_embeddings, dim=1)
        # Dimensionality reduction
        clip_embedding = self.dim_reduction(clip_embedding)
        
        # If only the embedding is needed, return it here
        if return_embedding:
            return clip_embedding
        
        # Pass the embedding through the classifier
        class_output = self.classifier(clip_embedding)
        return class_output



if __name__ == "__main__":
    my_model = DINOVforIR(34, embedding_dim=512, freeze_dino=True)
    #print(model)
    
    my_model.to('cuda:2')

    #my_model = nn.DataParallel(model, device_ids= [1, 2])
    import sys
    sys.path.append("/home/bas06400/Thesis/VIP/src/")

    from zeta.data_loader import load_dataloaders

    data_root = "/net/polaris/storage/deeplearning/ntu"
    modalities = ["depth"]
    batch_size = 8
    num_workers = 10
    pin_memory = True
    split = "0"
    random_sample = False
    config = {'dataset':'DAA','encoder_model':'CLIP-VIP'}
    train_data, val_data, test_data = load_dataloaders(data_root=data_root,
                                                        modalities=modalities,
                                                            batch_size=batch_size,
                                                            num_workers=num_workers,
                                                            pin_memory=pin_memory,
                                                            split=split,
                                                            random_sample=random_sample,
                                                            config=config)
    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Number of training epochs
    num_epochs = 100
    # Assuming train_data is your DataLoader for training
    class_weights = calculate_class_weights(train_data)
    #class_weights = 60 * [1/60]
    #print(len(class_weights), class_weights)
    # Start training
    # Start training with class weights
    train(my_model, train_data, optimizer, scheduler, num_epochs, val_data, class_weights=class_weights)




