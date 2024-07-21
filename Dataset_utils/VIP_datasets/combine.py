import json

def combine_jsonl_files_no_duplicates(file1, file2, file3, output_file):
    combined_lines = set()

    # Function to process each file
    def process_file(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                # Convert to tuple for hashability
                line_tuple = tuple(sorted(json.loads(line).items()))
                combined_lines.add(line_tuple)

    # Process all three files
    process_file(file1)
    process_file(file2)
    process_file(file3)

    # Write combined unique lines to the output file
    with open(output_file, 'w') as out_file:
        for line_tuple in combined_lines:
            # Convert tuple back to dictionary
            line_dict = dict(line_tuple)
            out_file.write(json.dumps(line_dict) + '\n')

    print(f"Total unique entries: {len(combined_lines)}")

# Example usage
file1 = '/home/bas06400/Thesis/CLIPVIP_Datasets/traindaa_0_words.jsonl'
file2 = '/home/bas06400/Thesis/CLIPVIP_Datasets/traindaa_1_words.jsonl'
file3 = '/home/bas06400/Thesis/CLIPVIP_Datasets/traindaa_2_words.jsonl'
output_file = '/home/bas06400/Thesis/CLIPVIP_Datasets/all_daa_clips_unique.jsonl'

combine_jsonl_files_no_duplicates(file1, file2, file3, output_file)