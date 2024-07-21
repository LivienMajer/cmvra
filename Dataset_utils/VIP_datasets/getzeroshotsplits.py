import json
import os
from glob import glob

def read_split_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def normalize_class_name(class_name):
    # Add any other special cases here
    if class_name.startswith("looking_or_moving_around"):
        return "looking_or_moving_around"
    return class_name

def process_jsonl_file(file_path, class_names):
    matching_samples = []
    normalized_class_names = [normalize_class_name(cls) for cls in class_names]
    with open(file_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if normalize_class_name(sample['text']) in normalized_class_names:
                matching_samples.append(sample)
    return matching_samples

def get_unique_classes(samples):
    return set(sample['text'] for sample in samples)

def create_zero_shot_dataset(input_file, split_files_dir, output_dir):
    for i, split_file in enumerate(glob(os.path.join(split_files_dir, 'midlevel_seen_classes_*.txt'))):
        class_names = read_split_file(split_file)
        
        output_samples = process_jsonl_file(input_file, class_names)

        output_file = os.path.join(output_dir, f'data_zero_shot_train_{i}.jsonl')
        with open(output_file, 'w') as f:
            for sample in output_samples:
                json.dump(sample, f)
                f.write('\n')

        unique_classes = get_unique_classes(output_samples)
        print(f"Created {output_file} with {len(output_samples)} samples")
        print(f"Unique classes in {output_file}:")
        for cls in sorted(unique_classes):
            print(f"  - {cls}")
        print()  # Add a blank line for readability

        # Print classes from split file that are missing in the output
        missing_classes = set(normalize_class_name(cls) for cls in class_names) - set(normalize_class_name(cls) for cls in unique_classes)
        if missing_classes:
            print(f"Warning: The following classes from the split file are missing in the output:")
            for cls in sorted(missing_classes):
                print(f"  - {cls}")
            print()

        # Sanity check: print unique class names in the output file
        unique_classes = get_unique_classes(output_samples)
        print(f"Created {output_file} with {len(output_samples)} samples")
        print(f"Unique classes in {output_file}:")
        for cls in sorted(unique_classes):
            print(f"  - {cls}")
        print()  # Add a blank line for readability

if __name__ == "__main__":
    input_dir = "/home/bas06400/Thesis/CLIPVIP_Datasets/all_daa_clips_unique_balanced.jsonl"
    split_files_dir = "/home/bas06400/zs-drive_and_act/splits"
    output_dir = "/home/bas06400/Thesis/CLIPVIP_Datasets"

    create_zero_shot_dataset(input_dir, split_files_dir, output_dir)