import json

# Path to the input and output files
input_path = '/home/bas06400/Thesis/CLIPVIP_Datasets/CV_training_set_words.jsonl'
output_path = '/home/bas06400/Thesis/CLIPVIP_Datasets/CV_training_set_wordss.jsonl'

# Read the file line by line and filter out unwanted entries
cleaned_data = []
with open(input_path, 'r') as file:
    for line in file:
        try:
            # Try to parse each line as JSON
            entry = json.loads(line)
            # Add to cleaned data if it doesn't have "Unknown action"
            if entry.get('text') != "Unknown action":
                cleaned_data.append(entry)
        except json.JSONDecodeError as e:
            # Handle possible json decoding errors
            print(f"Error decoding JSON from line: {line}, error: {e}")

# Save the cleaned data back to a new JSON file
with open(output_path, 'w') as file:
    for entry in cleaned_data:
        json.dump(entry, file)
        file.write('\n')  # Write each JSON object on a new line

print(f"File has been cleaned and saved as '{output_path}'.")