import json

# Path to the input JSONL file
input_file = '/home/bas06400/Thesis/CLIPVIP_Datasets/CV_testing_set_words.jsonl'
# Path to the output JSONL file
output_file = '/home/bas06400/Thesis/CLIPVIP_Datasets/CV_testing_set_words_reduced.jsonl'

def process_jsonl(input_file, output_file):
    # Dictionary to store unique texts with their first corresponding clip ID
    unique_texts = {}

    # Read the JSONL file and process each line
    with open(input_file, 'r') as infile:
        for line in infile:
            # Convert string to JSON object
            data = json.loads(line)
            text = data['text']#[0]
            clip_id = data['clip_id']

            # If the text is not already in the dictionary, add it with its clip ID
            if text not in unique_texts:
                unique_texts[text] = clip_id

    # Write the unique entries back to a new JSONL file
    with open(output_file, 'w') as outfile:
        for text, clip_id in unique_texts.items():
            # Create a new JSON object with the unique text and its first corresponding clip ID
            data = {'clip_id': clip_id, 'text': text}
            # Convert JSON object to string and write to file
            json_line = json.dumps(data)
            outfile.write(json_line + '\n')

# Call the function with your file paths
process_jsonl(input_file, output_file)

# Call the function with your file paths
process_jsonl(input_file, output_file)