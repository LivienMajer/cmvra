import sys
sys.path.append('/home/bas06400/Thesis/VIP/src')
from zeta.multimodal_dataset import MultiModalVideoDataset3
from torch.utils.data import random_split
from line_profiler import LineProfiler



def process_dataset():
    data_root = '/net/polaris/storage/deeplearning/ntu'
    data_list = '/home/bas06400/Thesis/rgb_ir_depth_skeleton_dataset.txt'
    data = MultiModalVideoDataset3(data_list, data_root, ['rgb', 'ir', 'depth'])

    train_len = int(0.8 * len(data))
    val_len = int(0.1 * len(data))
    test_len = len(data) - train_len - val_len
    train_data, val_data, test_data = random_split(data, [train_len, val_len, test_len])

    for idx in range(10):
        x, y, z = train_data[idx][0]
        del x, y, z

# Create a LineProfiler instance
profiler = LineProfiler()
profiler.add_function(process_dataset)
profiler.add_function(MultiModalVideoDataset3.__getitem__)
profiler.add_function(MultiModalVideoDataset3._extract_frames)  # Add the process_dataset function to profiler
profiler.add_function(MultiModalVideoDataset3.load_video) 
# Run the function with profiling
profiler.runcall(process_dataset)

# Print the profiling results
profiler.print_stats()