import torch
from torchvision.transforms import ToTensor

def calculate_mean_std_per_modality(dataset):
    stats = {}
    for modality in dataset.modalities:
        num_channels = 1 if modality in ['ir', 'depth'] else 3
        stats[modality] = {'sum': torch.zeros(num_channels), 
                           'sum_squared': torch.zeros(num_channels), 
                           'num_elements': 0}

    for idx in range(len(dataset)):
        if idx % 2000 == 0:
            print(idx)
        modality_frames, label = dataset[idx]

        for modality in dataset.modalities:
            frames = modality_frames.get(modality, [])
            num_channels = 1 if modality in ['ir', 'depth'] else 3

            for frame_tensor in frames:
                if num_channels == 1:
                    # If single channel, ensure the tensor is of shape [1, H, W]
                    frame_tensor = frame_tensor.unsqueeze(0) if len(frame_tensor.shape) == 2 else frame_tensor
                stats[modality]['sum'] += torch.sum(frame_tensor, dim=[1, 2])
                stats[modality]['sum_squared'] += torch.sum(frame_tensor ** 2, dim=[1, 2])
                stats[modality]['num_elements'] += frame_tensor.size(1) * frame_tensor.size(2)

    for modality in stats:
        mean = stats[modality]['sum'] / stats[modality]['num_elements']
        std = (stats[modality]['sum_squared'] / stats[modality]['num_elements'] - mean ** 2) ** 0.5
        stats[modality]['mean'] = mean
        stats[modality]['std'] = std

    return stats

from multimodal_dataset import MultiModalVideoDataset
from torch.utils.data import random_split

data_root = '/net/polaris/storage/deeplearning/ntu'
data_list = '/home/bas06400/Thesis/rgb_ir_depth_skeleton_dataset.txt'
data = MultiModalVideoDataset(data_list, data_root, ['rgb','ir','depth'], use_advanced_processing=False)

print(data[0][0]['rgb'].shape, data[0][0]['ir'].shape, data[0][0]['depth'].shape, data[0][1])

# Calculate lengths of splits
total_len = len(data)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len

# Split the dataset
train_data, val_data, test_data = random_split(data, [train_len, val_len, test_len])
# Calculate mean and std for each modality in the dataset
modality_stats = calculate_mean_std_per_modality(data)
for modality in modality_stats:
    print(f"Modality: {modality}, Mean: {modality_stats[modality]['mean']}, Std: {modality_stats[modality]['std']}")