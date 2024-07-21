import os

def read_split_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def create_combined_label_file(splits_dir, output_file):
    all_splits = {}

    for i in range(10):  # Assuming 10 splits (0-9)
        split_file = os.path.join(splits_dir, f'midlevel_unseen_classes_test_{i}.txt')
        if os.path.exists(split_file):
            labels = read_split_file(split_file)
            all_splits[f'zs{i}'] = labels
        else:
            print(f"Warning: Unseen classes test file for split {i} not found.")

    with open(output_file, 'w') as f:
        f.write("# Zero-shot unseen class labels for all splits\n\n")
        for split_name, labels in all_splits.items():
            f.write(f"{split_name} = {labels}\n\n")
    
    print(f"Created combined file: {output_file}")
    print(f"Total splits processed: {len(all_splits)}")

if __name__ == "__main__":
    splits_dir = "/home/bas06400/zs-drive_and_act/splits"
    output_file = "/home/bas06400/zs-drive_and_act/zs_labels.py"
    
    create_combined_label_file(splits_dir, output_file)