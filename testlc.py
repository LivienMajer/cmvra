import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from evl.vision_transformer import VisionTransformer2D
from evl.model import EVLTransformer
from collections import OrderedDict
import os
from tqdm import tqdm
import clip
from multimodal_dataset import MultiModalVideoDataset
from torch.utils.data import random_split


def load_model_from_checkpoint(model, checkpoint_path):
    """
    Loads a model's state dictionary from a checkpoint without the "module." prefix.
    
    Parameters:
    - model (torch.nn.Module): The model to which the state dictionary will be loaded.
    - checkpoint_path (str): Path to the checkpoint file.

    Returns:
    - model (torch.nn.Module): The model with its state dictionary loaded from the checkpoint.
    """

    # Load the checkpoint
    ckpt = torch.load(checkpoint_path)

    # Create a new state dictionary without the "module." prefix
    new_state_dict = OrderedDict()
    for key, value in ckpt['model'].items():
        name = key[7:]  # remove the "module." prefix
        new_state_dict[name] = value

    # Load the state dictionary into the model
    model.load_state_dict(new_state_dict)

    return model

class CLIPModelx(nn.Module):
    def __init__(self):
        super(CLIPModelx, self).__init__()
        self.rgb_model = EVLTransformer(backbone_path='/home/bas06400/.cache/clip/ViT-B-16.pt')
        self.ir_model = EVLTransformer(backbone_path='/home/bas06400/.cache/clip/ViT-B-16.pt')
        checkpoint_path = '/home/bas06400/schlaf/k400_vitb16_8f_dec4x768.pth'
        self.rgb_model = load_model_from_checkpoint(self.rgb_model,checkpoint_path)
        load_model_from_checkpoint(self.ir_model,checkpoint_path)

        # Projection layers to transform 400D embeddings to 512D
        self.rgb_projection = nn.Linear(400, 512)
        self.ir_projection = nn.Linear(400, 512)


    def forward(self, image, text):
        
        # Get 400D embeddings from RGB and IR models
        rgb_features_400D = self.rgb_model(image)
        ir_features_400D = self.ir_model(text)
        
        # Project the 400D embeddings to 512D
        rgb_features = self.rgb_projection(rgb_features_400D)
        ir_features = self.ir_projection(ir_features_400D)
        return rgb_features, ir_features

model = CLIPModelx()

data_root = '/home/bas06400/ntu'
data_list = '/home/bas06400/Thesis/rgb_ir_dataset.txt'
data = MultiModalVideoDataset(data_list, data_root, ['rgb','ir'], use_advanced_processing=True)

print(data[0][0]['rgb'].shape, data[0][0]['ir'].shape, data[0][1])

# Calculate lengths of splits
total_len = len(data)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len

# Split the dataset
train_data, val_data, test_data = random_split(data, [train_len, val_len, test_len])

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of data from MultiModalVideoDataset.
    
    Args:
    - batch (list): List of samples fetched from `MultiModalVideoDataset`.
    
    Returns:
    - collated_data (dict): Collated data for each modality.
    - collated_labels (tensor): Collated labels.
    """
    collated_data = {}
    collated_labels = []
    
    # Initialize empty lists for each modality in the first sample
    for modality in batch[0][0].keys():
        collated_data[modality] = []
    
    for data, label in batch:
        collated_labels.append(label-1)
        for modality, frames in data.items():
            collated_data[modality].append(frames)
    
    # Convert lists to tensors for each modality
    for modality, frames_list in collated_data.items():
        collated_data[modality] = torch.stack(frames_list)
    
    collated_labels = torch.tensor(collated_labels)
    
    return collated_data, collated_labels

# Create a DataLoader
batch_size = 64
shuffle = True
num_workers = 10
pin_memory = True

# Create a DataLoader for the training set
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=custom_collate_fn
)

# Create a DataLoader for the validation set
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,  
    shuffle=False,  
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=custom_collate_fn
)

# Create a DataLoader for the test set
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,  
    shuffle=False,  
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=custom_collate_fn
)


ckpt = torch.load('/home/bas06400/Thesis/best_model.pth')
model.load_state_dict(ckpt)

class CLIPModelxClassefier(nn.Module):
    def __init__(self):
        super(CLIPModelxClassefier, self).__init__()
        self.representmodel= CLIPModelx()
        ckpt = torch.load('/home/bas06400/Thesis/best_model.pth')
        self.representmodel.load_state_dict(ckpt)
        for param in self.representmodel.parameters():
            param.requires_grad_(False)
        
        # 

        self.rgb_classefier = nn.Linear(512, 60)
        self.ir_classefier = nn.Linear(512, 60)


    def forward(self, image, text):
        
        rgb_out, ir_out = self.representmodel(image, text)
        
        # Project the 400D embeddings to 512D
        rgb_features = self.rgb_classefier(rgb_out)
        ir_features = self.ir_classefier(ir_out)
        return rgb_features, ir_features
    

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


num_epochs = 10
model = CLIPModelxClassefier()
model = model.to('cuda:3')
# Cross-Entropy Loss function
criterion = torch.nn.CrossEntropyLoss()

# Hyperparameters
learning_rate = 0.001

# Placeholder for best validation loss
best_val_loss = float('inf')

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

# Function to compute accuracy
def compute_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)

# Training loop with modified loss and accuracy calculations
for epoch in range(num_epochs):
    epoch_loss = 0.0
    total_accuracy_rgb = 0.0
    total_accuracy_ir = 0.0
    
    model.train()
    
    for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move data and labels to GPU
        rgb_data = batch_data['rgb'].to('cuda:3')
        ir_data = batch_data['ir'].to('cuda:3')
        labels = batch_labels.to('cuda:3')

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        rgb_logits, ir_logits = model(rgb_data, ir_data)
        
        # Compute the losses
        loss_rgb = criterion(rgb_logits, labels)
        loss_ir = criterion(ir_logits, labels)
        loss = (loss_rgb + loss_ir) / 2  # Average the two losses

        # Compute accuracies
        accuracy_rgb = compute_accuracy(rgb_logits, labels)
        accuracy_ir = compute_accuracy(ir_logits, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()
        total_accuracy_rgb += accuracy_rgb
        total_accuracy_ir += accuracy_ir

    avg_accuracy_rgb = total_accuracy_rgb / len(train_loader)
    avg_accuracy_ir = total_accuracy_ir / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, RGB Accuracy: {avg_accuracy_rgb:.4f}, IR Accuracy: {avg_accuracy_ir:.4f}")
    
    # Validation loop
    model.eval()
    total_val_accuracy_rgb = 0.0
    total_val_accuracy_ir = 0.0
    val_loss = 0.0
    
    for batch_data, batch_labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
        rgb_data = batch_data['rgb'].to('cuda:3')
        ir_data = batch_data['ir'].to('cuda:3')
        labels = batch_labels.to('cuda:3')
        
        rgb_logits, ir_logits = model(rgb_data, ir_data)
        
        # Compute the losses
        loss_rgb = criterion(rgb_logits, labels)
        loss_ir = criterion(ir_logits, labels)
        loss = (loss_rgb + loss_ir) / 2
        
        # Compute accuracies
        accuracy_rgb = compute_accuracy(rgb_logits, labels)
        accuracy_ir = compute_accuracy(ir_logits, labels)

        val_loss += loss.item()
        total_val_accuracy_rgb += accuracy_rgb
        total_val_accuracy_ir += accuracy_ir

    avg_val_accuracy_rgb = total_val_accuracy_rgb / len(val_loader)
    avg_val_accuracy_ir = total_val_accuracy_ir / len(val_loader)
    print(f"Validation Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss / len(val_loader):.4f}, RGB Accuracy: {avg_val_accuracy_rgb:.4f}, IR Accuracy: {avg_val_accuracy_ir:.4f}")
    
    # Save the best model (optional)
    avg_val_loss = val_loss / len(val_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_classefier_model.pth')

print("Training complete!")