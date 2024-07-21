def modify_columns(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []

    for line in lines:
        columns = line.strip().split(' ')

        # Modify the first two columns for directory and filename
        for i in range(2):
            dir_part, file_part = columns[i].rsplit('/', 1)
            file_parts = file_part.split('.')
            file_parts[-2] += '_low_res'  # Append '_low_res' to filename
            columns[i] = dir_part + '_low_res/' + '.'.join(file_parts)


        modified_lines.append(' '.join(columns))

    with open(output_file_path, 'w') as file:
        for line in modified_lines:
            file.write(line + '\n')

# Usage
input_file = '/home/bas06400/Thesis/CS_testing_set.txt'
output_file = 'CS_testing_set_low_res.txt'
modify_columns(input_file, output_file)
