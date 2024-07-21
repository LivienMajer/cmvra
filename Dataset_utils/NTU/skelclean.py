def modify_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    modified_lines = [line.replace('nturgb+d_skeleton/', 'nturgb+d_skeletons_npy/') for line in lines]

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)

# Example usage
input_file = '/home/bas06400/Thesis/CV_training_set_low_res_cleaned.txt'  # Replace with your input file name
output_file = '/home/bas06400/Thesis/CV_training_set_low_res_cleaned2.txt'  # The modified file will be saved with this name

modify_file(input_file, output_file)
