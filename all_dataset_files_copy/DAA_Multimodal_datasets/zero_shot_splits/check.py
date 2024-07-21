import re
from typing import Dict

def extract_class_examples(file_path: str) -> Dict[str, str]:
    class_examples = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            # Extract the class name from the file path
            match = re.search(r'/([^/]+)_\d+_vp\d+', line)
            if match:
                class_name = match.group(1)
                # If we haven't seen this class before, add this line as an example
                if class_name not in class_examples:
                    class_examples[class_name] = line.strip()
    
    return class_examples

def main(input_file: str, output_file: str):
    class_examples = extract_class_examples(input_file)
    
    with open(output_file, 'w') as file:
        for class_name, example in sorted(class_examples.items()):
            file.write(f"{example}\n")
    
    print(f"Extracted examples for {len(class_examples)} classes and saved them to {output_file}")

if __name__ == "__main__":
    # Specify your input and output file paths here
    input_file = "/home/bas06400/daa/zero_shot_splits/data_zero_shot_train_1_balanced.txt"  # Replace with the actual path to your input file
    output_file = "/home/bas06400/daa/zero_shot_splits/unique_lines_output.txt"  # You can change this if you want a different output file name
    
    main(input_file, output_file)