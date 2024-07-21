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

# Usage
input_file = '/home/bas06400/daa/daa_split_train2_full.txt'
output_file = '/home/bas06400/daa/daa_split_train2_full_balanced.txt'
balance_dataset(input_file, output_file)