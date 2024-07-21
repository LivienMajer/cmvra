import json
from collections import defaultdict
import random

def balance_dataset(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Parse JSON and group by class
    class_data = defaultdict(list)
    for line in lines:
        entry = json.loads(line)
        class_name = entry['text']
        class_data[class_name].append(entry)

    # Find the class with the maximum number of samples
    max_samples = max(len(samples) for samples in class_data.values())

    # Balance the dataset
    balanced_data = []
    for class_name, samples in class_data.items():
        while len(samples) < max_samples:
            samples.append(random.choice(samples))
        balanced_data.extend(samples)

    # Shuffle the balanced dataset
    random.shuffle(balanced_data)

    # Write the balanced dataset to the output file
    with open(output_file, 'w') as f:
        for entry in balanced_data:
            json.dump(entry, f)
            f.write('\n')

    print(f"Balanced dataset written to {output_file}")
    print("Class distribution:")
    for class_name, samples in class_data.items():
        print(f"{class_name}: {len(samples)}")

# Usage
input_file = '/home/bas06400/Thesis/CLIPVIP_Datasets/all_daa_clips_unique.jsonl'
output_file = '/home/bas06400/Thesis/CLIPVIP_Datasets/all_daa_clips_unique_balanced.jsonl'
balance_dataset(input_file, output_file)