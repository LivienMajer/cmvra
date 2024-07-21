import os
import cv2
# Root directory where the video files are stored
root_dir = '/net/polaris/storage/deeplearning/ntu'

# Path to your text file containing the dataset file paths
dataset_file_path = '/home/bas06400/Thesis/CV120_testing_set.txt'

# Function to check if the file exists and is not empty
def file_exists_and_not_empty(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

# Function to check if a file is a valid video
def is_valid_video(file_path):
    if not file_exists_and_not_empty(file_path):
        return False  # File doesn't exist or is empty, so not a valid video
    cap = cv2.VideoCapture(file_path)
    is_valid = cap.isOpened()
    cap.release()
    return is_valid

# Open and read the dataset file
with open(dataset_file_path, 'r') as file:
    for line in file:
        # Split the line into columns
        columns = line.strip().split(' ')
        
        # Check the first two columns (video files) for validity, with root directory prepended
        for col in columns[:2]:  # Adjust index as needed
            full_path = os.path.join(root_dir, col)  # Construct full path
            if not is_valid_video(full_path):
                print(f"File is not a valid video or does not exist: {full_path}")