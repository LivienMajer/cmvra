import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_clip_id(clip_id):
    return clip_id.strip().lower()

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                item['clip_id'] = normalize_clip_id(item['clip_id'])
                data.append(item)
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from line: {line}")
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def update_descriptions(all_descriptions, split_data):
    description_dict = {normalize_clip_id(item['clip_id']): item['text'] for item in all_descriptions}
    
    updated_data = []
    missing_clips = []
    for item in split_data:
        normalized_id = normalize_clip_id(item['clip_id'])
        if normalized_id in description_dict:
            item['text'] = description_dict[normalized_id]
        else:
            # Fallback: use the original text, repeated 9 times
            original_text = item['text']
            if isinstance(original_text, str):
                item['text'] = [original_text] * 9
            elif isinstance(original_text, list):
                # If it's already a list, ensure it has 9 elements
                item['text'] = (original_text * ((9 + len(original_text) - 1) // len(original_text)))[:9]
            missing_clips.append(item['clip_id'])
        updated_data.append(item)
    
    if missing_clips:
        logging.warning(f"Used original text for {len(missing_clips)} clips: {missing_clips[:5]}...")
    
    return updated_data

def main():
    # File paths
    all_descriptions_path = '/home/bas06400/Thesis/CLIPVIP_Datasets/daa_all_generated_descriptions.jsonl'
    split1_path = '/home/bas06400/Thesis/CLIPVIP_Datasets/train_split1_daa.jsonl'
    split2_path = '/home/bas06400/Thesis/CLIPVIP_Datasets/train_split2_daa.jsonl'
    
    # Load data
    all_descriptions = load_jsonl(all_descriptions_path)
    split1_data = load_jsonl(split1_path)
    split2_data = load_jsonl(split2_path)
    
    logging.info(f"Loaded {len(all_descriptions)} descriptions")
    logging.info(f"Loaded {len(split1_data)} items from split 1")
    logging.info(f"Loaded {len(split2_data)} items from split 2")
    
    # Update descriptions
    updated_split1 = update_descriptions(all_descriptions, split1_data)
    updated_split2 = update_descriptions(all_descriptions, split2_data)
    
    logging.info(f"Updated split 1 has {len(updated_split1)} items")
    logging.info(f"Updated split 2 has {len(updated_split2)} items")
    
    # Save updated data
    save_jsonl(updated_split1, '/home/bas06400/Thesis/CLIPVIP_Datasets/llava_train_split1_daa.jsonl')
    save_jsonl(updated_split2, '/home/bas06400/Thesis/CLIPVIP_Datasets/llava_train_split2_daa.jsonl')
    
    logging.info("Updated dataset files have been created.")

if __name__ == "__main__":
    main()