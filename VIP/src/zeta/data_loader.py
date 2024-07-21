import random
import torch
import logging
import os
from torch.utils.data import DataLoader, random_split

from zeta.multimodal_dataset import MultiModalVideoDataset3



def load_dataloaders(data_root, modalities=['rgb','ir'], batch_size=16, num_workers=10, pin_memory=True, split='CS', random_sample= False, config=None):
    """
    Load and prepare data loaders for multimodal video datasets.

    This function sets up train, validation, and test data loaders for various multimodal video datasets,
    including NTU, NTUcropped, NTU120, and DAA. It supports different split modes and encoder models.

    Parameters:
    - data_root (str): Root directory of the dataset.
    - modalities (list): List of modalities to use (e.g., ['rgb', 'ir']).
    - batch_size (int): Number of samples per batch.
    - num_workers (int): Number of subprocesses to use for data loading.
    - pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    - split (str): Split mode to use. Options vary by dataset:
        - For NTU, NTUcropped, NTU120: 'CS' (Cross-Subject) or 'CV' (Cross-View)
        - For DAA: '0', '1', '2' for original splits, or 'zs0' to 'zs9' for zero-shot splits
    - random_sample (bool): If True, randomly sample frames from videos.
    - config (dict): Configuration dictionary containing:
        - 'encoder_model': The encoder model to use (e.g., 'CLIP-VIP', 'MAE', 'MIX', 'OMNIVORE')
        - 'dataset': The dataset to use ('NTU', 'NTUcropped', 'NTU120', or 'DAA')
        - 'modalities_encoders': Dictionary mapping modalities to their respective encoders
        - 'augs': Boolean indicating whether to use data augmentation

    Returns:
    - train_loader (DataLoader): DataLoader for the training set
    - val_loader (DataLoader): DataLoader for the validation set
    - test_loader (DataLoader): DataLoader for the test set

    Note:
    - The function uses a custom collate function to handle the multimodal data.
    - For NTU datasets, labels are shifted by -1 to start from 0.
    - The train/validation split ratio is hardcoded to 0.98/0.02 for NTU since the paper proposes just a train/test split.
    """
    input_frames_for_model = {
        'CLIP-VIP': 12,
        'MAE': 16,
        'MIX': 12,
        'OMNIVORE':12
    }
    if config['encoder_model'] == 'MIX' and any('MAE' == encoder for encoder in config['modalities_encoders'].values()):
        mixed_frames = {}
        for modality, encoder in config['modalities_encoders'].items():
            # Check if this modality is listed in 'modalities' and its encoder is 'MAE'
            if modality in config['modalities']:
                # Fetch the cowrresponding frames
                mixed_frames[modality] = input_frames_for_model[encoder]
    else:
        mixed_frames = False
    frame_count = input_frames_for_model[config['encoder_model']]
    augs = config.get('augs',False)
    seed = 42 # Set the seed for reproducibility
    random.seed(seed)  # Seed for Python's random module
    torch.manual_seed(seed)  # Seed for PyTorch random number generators
    if config['dataset'] == 'NTU':
        # Define file paths for datasets based on mode
        if split == 'CS':
            train_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CS_training_set.txt'
            test_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CS_testing_set.txt'
            print('CS')
        elif split == 'CV':
            train_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CV_training_set.txt'
            test_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CV_testing_set.txt'
        else:
            raise ValueError("Invalid mode. Choose 'CS' for Cross-Subject or 'CV' for Cross-View.")
        
        

        train_data = MultiModalVideoDataset3(train_data_list, data_root, modalities, frame_count=frame_count, random_sample=random_sample, mixed_frames=mixed_frames, mode='train', augs =augs)
        test_data = MultiModalVideoDataset3(test_data_list, data_root, modalities, frame_count=frame_count, random_sample=random_sample, mixed_frames=mixed_frames, mode='test', augs =augs)
        # Calculate lengths of splits for training and validation
        train_len = int(0.98 * len(train_data))
        val_len = len(train_data) - train_len

        # Split the training dataset into training and validation sets
        train_data, val_data = random_split(train_data, [train_len, val_len])

    elif config['dataset'] == 'NTUcropped':
        # Define file paths for datasets based on mode
        if split == 'CS':
            train_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CS_training_set_cropped.txt'
            test_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CS_testing_set_cropped.txt'
        elif split == 'CV':
            train_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CV_training_set_cropped_low_res_cleaned2.txt'
            test_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CV_testing_set_cropped_low_res_cleaned2.txt'
        else:
            raise ValueError("Invalid mode. Choose 'CS' for Cross-Subject or 'CV' for Cross-View.")
        
        

        train_data = MultiModalVideoDataset3(train_data_list, data_root, modalities, frame_count=frame_count, random_sample=random_sample, mixed_frames=mixed_frames, mode='train', augs =augs)
        test_data = MultiModalVideoDataset3(test_data_list, data_root, modalities, frame_count=frame_count, random_sample=random_sample, mixed_frames=mixed_frames, mode='test', augs =augs)
        
        # Calculate lengths of splits for training and validation
        train_len = int(0.98 * len(train_data))
        val_len = len(train_data) - train_len

        # Split the training dataset into training and validation sets
        train_data, val_data = random_split(train_data, [train_len, val_len])

    elif config['dataset'] == 'NTU120':
        # Define file paths for datasets based on mode
        if split == 'CS':
            train_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CS120_training_set.txt'
            test_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CS120_testing_set.txt'
        elif split == 'CV':
            train_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CV120_training_set.txt'
            test_data_list = '/home/bas06400/Thesis/all_dataset_files_copy/NTU_Multimodaldatasets/CV120_testing_set.txt'
        else:
            raise ValueError("Invalid mode. Choose 'CS' for Cross-Subject or 'CV' for Cross-View.")
        

        train_data = MultiModalVideoDataset3(train_data_list, data_root, modalities, frame_count=frame_count, random_sample=random_sample, mixed_frames=mixed_frames, mode='train', augs =augs)
        test_data = MultiModalVideoDataset3(test_data_list, data_root, modalities, frame_count=frame_count, random_sample=random_sample, mixed_frames=mixed_frames, mode='test', augs =augs)
        # Calculate lengths of splits for training and validation
        train_len = int(0.98 * len(train_data))
        val_len = len(train_data) - train_len

        # Split the training dataset into training and validation sets
        train_data, val_data = random_split(train_data, [train_len, val_len])
        
    elif config['dataset'] == 'DAA':
        # Base path for zero-shot splits
        zero_shot_base_path = '/home/bas06400/Thesis/all_dataset_files_copy/DAA_Multimodal_datasets/zero_shot_splits'
        
        if split in ['0', '1', '2']:
            # Original splits
            train_data_list = f'/home/bas06400/Thesis/all_dataset_files_copy/DAA_Multimodal_datasets/daa_split_train{split}_full_balanced.txt'
            val_data_list = f'/home/bas06400/Thesis/all_dataset_files_copy/DAA_Multimodal_datasets/daa_split_val{split}_full.txt'
            test_data_list = f'/home/bas06400/Thesis/all_dataset_files_copy/DAA_Multimodal_datasets/daa_split_test{split}_full.txt'
        elif split.startswith('zs'):
            # Zero-shot splits
            zs_index = split[2:]  # Extract the index from 'zs0', 'zs1', etc.
            train_data_list = os.path.join(zero_shot_base_path, f'data_zero_shot_train_{zs_index}_balanced.txt')
            val_data_list = os.path.join(zero_shot_base_path, f'data_zero_shot_val_{zs_index}.txt')
            test_data_list = os.path.join(zero_shot_base_path, f'data_zero_shot_test_{zs_index}.txt')
        else:
            raise ValueError("Invalid split. Choose '0', '1', '2' for original splits or 'zs0', 'zs1', ..., 'zs9' for zero-shot splits.")
        
        data_root = '/home/bas06400/daa'
        train_data = MultiModalVideoDataset3(train_data_list, data_root, modalities, frame_count=frame_count, random_sample=random_sample, mixed_frames=mixed_frames, mode='train', augs =augs)
        val_data = MultiModalVideoDataset3(val_data_list, data_root, modalities, frame_count=frame_count, random_sample=random_sample, mixed_frames=mixed_frames, mode='val', augs =augs)
        test_data = MultiModalVideoDataset3(test_data_list, data_root, modalities, frame_count=frame_count, random_sample=random_sample, mixed_frames=mixed_frames, mode='test', augs =augs)


    else:
        logging.info('Currently only DAA or NTU are supported dataset options')
    shift_label = False
    if config['dataset'] =='NTU' or config['dataset'] =='NTU120' or config['dataset'] =='NTUcropped':
        shift_label = True
     # Create the DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        collate_fn=lambda batch: custom_collate_fn(batch, shift_label=shift_label)
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        collate_fn=lambda batch: custom_collate_fn(batch, shift_label=shift_label)
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        collate_fn=lambda batch: custom_collate_fn(batch, shift_label=shift_label)
    )
    return train_loader, val_loader, test_loader


def custom_collate_fn1(batch, shift_label=False):
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

def custom_collate_fn(batch, shift_label=False):
    modalities_shapes = {modality: (len(batch),) + frames.shape for modality, frames in batch[0][0].items()}
    collated_data = {modality: torch.empty(shape) for modality, shape in modalities_shapes.items()}
    collated_labels = torch.empty(len(batch), dtype=torch.long)

    for i, (data, label) in enumerate(batch):
        if shift_label:
            collated_labels[i] = label - 1
        else:
            collated_labels[i] = label
        for modality in data:
            collated_data[modality][i] = data[modality]

    return collated_data, collated_labels