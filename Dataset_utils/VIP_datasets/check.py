import json
from typing import Set

def extract_unique_texts(file_path: str) -> Set[str]:
    unique_texts = set()
    
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    unique_texts.add(data['text'])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line.strip()}")
    
    return unique_texts

def main(input_file: str, output_file: str):
    unique_texts = extract_unique_texts(input_file)
    
    with open(output_file, 'w') as file:
        for text in sorted(unique_texts):
            file.write(f"{text}\n")
    
    print(f"Extracted {len(unique_texts)} unique texts and saved them to {output_file}")

if __name__ == "__main__":
    input_file = "/home/bas06400/Thesis/CLIPVIP_Datasets/data_zero_shot_train_5.jsonl"  
    output_file = "/home/bas06400/Thesis/CLIPVIP_Datasets/unique_texts.txt"  
    main(input_file, output_file)