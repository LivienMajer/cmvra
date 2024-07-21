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
        assert len(weights) == 34, "Weights should be a list of elements equal to the number of classes, 34 for DAA"
        loss = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(output.device))
    else:
        loss = nn.CrossEntropyLoss()
    return loss(output, target)

def train(model, train_loader, optimizer, scheduler, num_epochs, val_loader, device='cuda', class_weights=None):
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
            #print(inputs.shape)
            # Take the first channel
            first_channel = inputs[:, :, 0, :, :]

            # Expand the first channel to match the shape of inputs for broadcasting
            first_channel_expanded = first_channel.unsqueeze(2)  # Adds a channel dimension back

            # Create a boolean tensor where True indicates element-wise equality
            equality_tensor = inputs == torch.cat((first_channel_expanded, first_channel_expanded, first_channel_expanded), 2)

            # Reduce the boolean tensor across the channel dimension (2) to check if all elements are equal within each channel
            channels_equal = torch.all(equality_tensor, dim=2)

            # Then, check if the result is True across all remaining dimensions to confirm all channels are identical
            are_channels_identical = torch.all(channels_equal)

            #print(are_channels_identical)  # Will print
            replicate_channel = inputs[:, :, 0:1, :, :]  # Take the first channel

            # Concatenate this channel with the original tensor along the channel dimension to add a fourth channel
            expanded_inputs = torch.cat((inputs, replicate_channel), 2)

            #print(expanded_inputs.shape)
            #print(inputs.expand(-1,-1, 4, -1, -1).shape)
            outputs = model(expanded_inputs.permute(0,2,1,3,4))
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

def validate(model, val_loader, device='cuda'):
    total_val_loss = 0
    total_correct = 0
    num_batches = 0

    with torch.no_grad():
        for batch_data, labels in val_loader:
            inputs, labels = batch_data['depth'].to(device), labels.to(device)

            replicate_channel = inputs[:, :, 0:1, :, :]  # Take the first channel

            # Concatenate this channel with the original tensor along the channel dimension to add a fourth channel
            expanded_inputs = torch.cat((inputs, replicate_channel), 2)

            #print(expanded_inputs.shape)
            #print(inputs.expand(-1,-1, 4, -1, -1).shape)
            outputs = model(expanded_inputs.permute(0,2,1,3,4))
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


class DepthNorm(nn.Module):
    """
    Normalize the depth channel: in an RGBD input of shape (4, H, W),
    only the last channel is modified.
    The depth channel is also clamped at 0.0. The Midas depth prediction
    model outputs inverse depth maps - negative values correspond
    to distances far away so can be clamped at 0.0
    """

    def __init__(
        self,
        max_depth: float,
        clamp_max_before_scale: bool = False,
        min_depth: float = 0.01,
    ):
        """
        Args:
            max_depth (float): The max value of depth for the dataset
            clamp_max (bool): Whether to clamp to max_depth or to divide by max_depth
        """
        super().__init__()
        if max_depth < 0.0:
            raise ValueError("max_depth must be > 0; got %.2f" % max_depth)
        self.max_depth = max_depth
        self.clamp_max_before_scale = clamp_max_before_scale
        self.min_depth = min_depth

    def forward(self, image: torch.Tensor):
        C, H, W = image.shape
        if C != 4:
            err_msg = (
                f"This transform is for 4 channel RGBD input only; got {image.shape}"
            )
            raise ValueError(err_msg)
        color_img = image[:3, ...]  # (3, H, W)
        depth_img = image[3:4, ...]  # (1, H, W)

        # Clamp to 0.0 to prevent negative depth values
        depth_img = depth_img.clamp(min=self.min_depth)

        # divide by max_depth
        if self.clamp_max_before_scale:
            depth_img = depth_img.clamp(max=self.max_depth)

        depth_img /= self.max_depth

        img = torch.cat([color_img, depth_img], dim=0)
        return img



if __name__ == "__main__":
    model = omnivore_swinB_imagenet21k()
    print(model)
    model.heads = nn.Linear(1024, 34, bias=False)
    model.to('cuda:1')

    my_model = nn.DataParallel(model, device_ids= [0, 1])
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
    #print(len(class_weights), class_weights)
    # Start training
    # Start training with class weights
    train(my_model, train_data, optimizer, scheduler, num_epochs, test_data, class_weights=class_weights)




