import torch

def load_and_print_shapes(features_file_path, labels_file_path):
    # Load the features and labels
    features = torch.load(features_file_path)
    labels = torch.load(labels_file_path)

    # Print the shapes
    print(f"Features Shape: {features.shape}")
    print(f"Labels Shape: {labels.shape}")

# Example file paths (replace these with your actual file paths)
features_file_path = '/home/bas06400/Thesis/VIP/src/features/train/train_rgb_features_0.pt'
labels_file_path = '/home/bas06400/Thesis/VIP/src/features/train/train_rgb_labels_0.pt'

# Load and print shapes
load_and_print_shapes(features_file_path, labels_file_path)
