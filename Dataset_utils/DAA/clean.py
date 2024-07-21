def replace_ir_with_depth_robust(line):
    # Split the line into columns
    columns = line.split()

    # Identify the IR and Depth columns
    ir_column = next((col for col in columns if 'kinect_ir/' in col), None)
    depth_column = next((col for col in columns if 'kinect_depth_mp4/' in col), None)

    # Check if both columns are found and have the expected pattern
    if ir_column and '.kinect_ir_' in ir_column and depth_column and '.kinect_depth_' in depth_column:
        # Extract the file names without extensions
        ir_prefix, ir_suffix = ir_column.split('.kinect_ir_')
        depth_suffix = depth_column.split('.kinect_depth_')[1]

        # Replace the IR suffix with the Depth suffix
        new_ir_file = ir_prefix + '.kinect_ir_' + depth_suffix

        # Replace the old IR file name with the new one in the columns
        columns[columns.index(ir_column)] = new_ir_file

    # Return the modified line
    return ' '.join(columns)

def main(input_file_path, output_file_path):
    # Read the lines from the input file
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Apply the replacement to each line and gather the results
    robust_modified_lines = [replace_ir_with_depth_robust(line) for line in lines]

    # Write the modified lines to the output file, adding a newline after each line
    with open(output_file_path, 'w') as file:
        for line in robust_modified_lines:
            file.write(f"{line}\n")


# Replace the paths below with your actual file paths
input_file_path = "/home/bas06400/daa/daa_split0train_false.txt"
output_file_path = "/home/bas06400/daa/daa_split0train.txt.txt"

# Run the script
main(input_file_path, output_file_path)
