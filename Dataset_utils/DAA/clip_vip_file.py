import os
import re
import json
from activity_mapping import activity_mapping, new_activity_mapping, word_variants




def extract_details(path, is_skeleton=False):
    filename = os.path.basename(path)

    # Remove the file extension
    identifier, _ = os.path.splitext(filename)
    

    # Patterns for activity extraction
    special_case_pattern = r"([^/]+)(?=_\d+_vp\d+)"
    general_pattern = r"([a-zA-Z_]+)_\d+"

    
    
    

    # Attempt to match activity if not a skeleton file
    activity = None
    if not is_skeleton:
        # Attempt to match special case activity
        special_match = re.search(special_case_pattern, filename)
        activity = special_match.group(1) if special_match else None

        # If no special case, attempt general pattern
        if not activity:
            general_match = re.match(general_pattern, filename)
            activity = general_match.group(1) if general_match else None

        # Clean up activity if it's found in the special case
        if activity:
            activity = re.sub(r"\s*\([^)]*\)\s*$", "", activity).strip()


    return identifier, activity

def process_file(file_path, output_file):
    with open(file_path, 'r') as file, open(output_file, 'w') as out:
        for line in file:
            # Use a regular expression to split on space preceded by 'i' or followed by 'n'
            paths = re.split(r'(?<=[iy]) | (?=n)', line.strip())
            if paths:
                color_path = paths[0]
                identifier, activity = extract_details(color_path)
                if identifier and activity:
                    # Map the extracted activity to the descriptive text
                    #descriptive_text = word_variants.get(activity, "Unknown Activity")
                    descriptive_text = activity
                    data = {"clip_id": identifier , "text": descriptive_text}
                    out.write(json.dumps(data) + '\n')  # Write each JSON object as a separate line to the file

# Replace 'your_dataset.txt' with the actual path to your dataset file
dataset_path = '/home/bas06400/daa/daa_split_train2_full.txt'
output_file_path = '/home/bas06400/Thesis/CLIPVIP_Datasets/traindaa_2_words.jsonl'  # The path to the output jsonl file
process_file(dataset_path, output_file_path)
