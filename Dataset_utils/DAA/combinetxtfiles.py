import os

def combine_text_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for input_file in input_files:
            if os.path.exists(input_file):
                with open(input_file, 'r') as infile:
                    outfile.write(infile.read())
                    # Add a newline between files if it's not already there
                    if not infile.read().endswith('\n'):
                        outfile.write('\n')
            else:
                print(f"Warning: Input file '{input_file}' not found.")

    print(f"Combined file created: {output_file}")

    # Print the contents of the combined file
    print("\nContents of the combined file:")
    with open(output_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    # Specify the paths to your input files
    input_file1 = "/home/bas06400/daa/daa_split_test_full.txt"
    input_file2 = "/home/bas06400/daa/daa_split_val_full.txt"
    input_file3 = "/home/bas06400/daa/daa_split_train_full.txt"

    # Specify the path for the output file
    output_file = "/home/bas06400/daa/all_clips.txt"

    # List of input files
    input_files = [input_file1, input_file2, input_file3]

    # Combine the files
    combine_text_files(input_files, output_file)