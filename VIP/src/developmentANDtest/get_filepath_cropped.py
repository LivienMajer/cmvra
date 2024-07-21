# Define the path for the input and output files
input_file_path = '/home/bas06400/Thesis/CS_training_set_low_res.txt'
output_file_path = 'CS_training_set_cropped.txt'

# Open the input file in read mode and the output file in write mode
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Iterate over each line in the input file
    for line in input_file:
        # Replace "low_res/" with "cropped_320_320/" and "_low_res.avi" with ""
        modified_line = line.replace("low_res/", "cropped_320x320/").replace("_low_res.avi", "")
        # Write the modified line to the output file
        output_file.write(modified_line)

# Print a confirmation message
print(f"Modified lines have been written to '{output_file_path}'.")