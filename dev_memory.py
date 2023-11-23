from multimodal_dataset import MultiModalVideoDataset
from torch.utils.data import random_split

@profile
def process_dataset():
    data_root = '/net/polaris/storage/deeplearning/ntu'
    data_list = '/home/bas06400/Thesis/rgb_ir_depth_skeleton_dataset.txt'
    data = MultiModalVideoDataset(data_list, data_root, ['rgb', 'ir', 'depth'], use_advanced_processing=True)

    train_len = int(0.8 * len(data))
    val_len = int(0.1 * len(data))
    test_len = len(data) - train_len - val_len
    train_data, val_data, test_data = random_split(data, [train_len, val_len, test_len])

    for idx in range(100):
        x, y, z = train_data[idx][0]
        del x, y, z

# Run this function to start profiling
process_dataset()