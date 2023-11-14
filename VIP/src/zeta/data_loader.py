import random
import torch
from torch.utils.data import DataLoader, random_split

# Assuming MultiModalVideoDataset is defined elsewhere, import it
from multimodal_dataset import MultiModalVideoDataset

def load_dataloaders(data_list, data_root, modalities=['rgb','ir'], batch_size=16, num_workers=10 ,pin_memory=True):
    # Set the seed for reproducibility
    seed = 42
    random.seed(seed)  # Seed for Python's random module
    torch.manual_seed(seed)  # Seed for PyTorch random number generators

    # Load the dataset
    data = MultiModalVideoDataset(data_list, data_root, modalities , use_advanced_processing=True)

    # Calculate lengths of splits
    total_len = len(data)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    # Split the dataset
    train_data, val_data, test_data = random_split(data, [train_len, val_len, test_len])

    # DataLoader settings
    shuffle = True
    

    # Create the DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader

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
    collated_idx =[]
    
    # Initialize empty lists for each modality in the first sample
    for modality in batch[0][0].keys():
        collated_data[modality] = []
    
    for data, label , idx in batch:
        collated_labels.append(label-1)
        for modality, frames in data.items():
            collated_data[modality].append(frames)
        collated_idx.append(idx)
    # Convert lists to tensors for each modality
    for modality, frames_list in collated_data.items():
        collated_data[modality] = torch.stack(frames_list)
    
    collated_labels = torch.tensor(collated_labels)
    
    return collated_data, collated_labels, collated_idx