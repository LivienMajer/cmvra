def process_line(line):
    columns = line.split()  # Split the line into columns based on whitespace.
    
    # Perform the required replacements and deletions.
    if len(columns) > 1:  # Ensure there is a column 2.
        columns[1] = columns[1].replace("_rgb_", "_ir_")
    if len(columns) > 2:  # Ensure there is a column 3.
        columns[2] = columns[2].replace("_rgb_low", "")
    if len(columns) > 3:  # Ensure there is a column 4.
        columns[3] = columns[3].replace("_rgb_low", "")
    
    return ' '.join(columns)  # Rejoin the modified columns into a single line.

# Replace 'your_file.txt' with the path to your actual text file.
with open('/home/bas06400/Thesis/CV_training_set_low_res.txt', 'r') as file:
    lines = file.readlines()

# Process each line.
processed_lines = [process_line(line) for line in lines]

# Write the processed lines back to a new file or overwrite the old one.
# Replace 'processed_file.txt' with the desired output file name.
with open('/home/bas06400/Thesis/CV_training_set_low_res_cleaned.txt', 'w') as file:
    for line in processed_lines:
        file.write(line + '\n')
