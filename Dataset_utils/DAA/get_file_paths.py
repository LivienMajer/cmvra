import os

def list_files(directory, output_file):
    try:
        # Open the output file in write mode
        with open(output_file, 'w') as file:
            # Walk through the directory
            for root, dirs, files in os.walk(directory):
                for name in files:
                    # Write each filename to the output file
                    file.write(os.path.join(root, name) + '\n')
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
    except OSError as e:
        print(f"An OS error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    for dir in ['kinect_color', 'kinect_ir', 'kinect_depth_mp4', 'openpose_3d','steering_wheel', 'a_column_driver', 'a_column_co_driver', 'ceiling', 'inner_mirror']:
        for set in ['train', 'val', 'test', 'train1', 'val1', 'test1', 'train2', 'val2', 'test2']:
            directory = f'/home/bas06400/daa/{dir}/clips/{set}'  # Replace with the path to the directory you want to scan
            output_file = f'{dir}_{set}.txt'  # The name of the output file

            list_files(directory, output_file)
            print(f"File names have been written to {output_file} (if the directory exists)")

if __name__ == "__main__":
    main()


