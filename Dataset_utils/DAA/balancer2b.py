import os
import random
from collections import defaultdict

def balance_dataset(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Group by class
    class_data = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        class_label = parts[-1]  # The class is the last item
        class_data[class_label].append(line)

    # Find the class with the maximum number of samples
    max_samples = max(len(samples) for samples in class_data.values())

    # Balance the dataset
    balanced_data = []
    for class_label, samples in class_data.items():
        while len(samples) < max_samples:
            samples.append(random.choice(samples))
        balanced_data.extend(samples)

    # Shuffle the balanced dataset
    random.shuffle(balanced_data)

    # Write the balanced dataset to the output file
    with open(output_file, 'w') as f:
        f.writelines(balanced_data)

    print(f"Balanced dataset written to {output_file}")
    print("Class distribution:")
    for class_label, samples in class_data.items():
        print(f"Class {class_label}: {len(samples)}")

def balance_all_training_sets(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all training set files
    training_files = [f for f in os.listdir(input_dir) if f.startswith('data_zero_shot_train_') and f.endswith('.txt')]

    for training_file in training_files:
        input_path = os.path.join(input_dir, training_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(training_file)[0]}_balanced.txt")
        
        print(f"\nProcessing {training_file}...")
        balance_dataset(input_path, output_path)

if __name__ == "__main__":
    input_directory = "/home/bas06400/daa/zero_shot_splits"
    output_directory = "/home/bas06400/daa/zero_shot_splits"
    
    balance_all_training_sets(input_directory, output_directory)