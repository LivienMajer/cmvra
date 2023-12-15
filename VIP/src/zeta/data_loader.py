import random
import torch
from torch.utils.data import DataLoader, random_split

# Assuming MultiModalVideoDataset is defined elsewhere, import it
from zeta.multimodal_dataset import MultiModalVideoDataset, MultiModalVideoDataset3



def load_dataloaders(data_root, modalities=['rgb','ir'], batch_size=16, num_workers=10, pin_memory=True, split='CS', random_sample= False):
    # Set the seed for reproducibility
    seed = 42
    random.seed(seed)  # Seed for Python's random module
    torch.manual_seed(seed)  # Seed for PyTorch random number generators

    # Define file paths for datasets based on mode
    if split == 'CS':
        train_data_list = '/home/bas06400/Thesis/CS_training_set.txt'
        test_data_list = '/home/bas06400/Thesis/CS_testing_set.txt'
    elif split == 'CV':
        train_data_list = '/home/bas06400/Thesis/CV_training_set.txt'
        test_data_list = '/home/bas06400/Thesis/CV_testing_set.txt'
    else:
        raise ValueError("Invalid mode. Choose 'CS' for Cross-Subject or 'CV' for Cross-View.")

    # Load the datasets
    #train_data = MultiModalVideoDataset(train_data_list, data_root, modalities, use_advanced_processing=True, random_sample=random_sample)
    #test_data = MultiModalVideoDataset(test_data_list, data_root, modalities, use_advanced_processing=True, random_sample=random_sample)

    train_data = MultiModalVideoDataset3(train_data_list, data_root, modalities, random_sample=random_sample)
    test_data = MultiModalVideoDataset3(test_data_list, data_root, modalities, random_sample=random_sample)
    # Calculate lengths of splits for training and validation
    train_len = int(0.98 * len(train_data))
    val_len = len(train_data) - train_len

    # Split the training dataset into training and validation sets
    train_data, val_data = random_split(train_data, [train_len, val_len])

    

    # Create the DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
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
    
    
    # Initialize empty lists for each modality in the first sample
    for modality in batch[0][0].keys():
        collated_data[modality] = []
    
    for data, label  in batch:
        collated_labels.append(label-1)
        for modality, frames in data.items():
            collated_data[modality].append(frames)
    
    # Convert lists to tensors for each modality
    for modality, frames_list in collated_data.items():
        collated_data[modality] = torch.stack(frames_list)
    
    collated_labels = torch.tensor(collated_labels)
    
    return collated_data, collated_labels