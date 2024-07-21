import cv2
import os
"""
data_root = "/net/polaris/storage/deeplearning/ntu"  # Original data root
output_dir = "/home/bas06400/lowresvidstest"  # Directory to save processed videos

def create_output_directory(filepath, file_type):
   
    subdir_structure = os.path.dirname(filepath)
    output_subdir = os.path.join(output_dir, file_type, subdir_structure)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

def process_video_file(filepath, output_res=(320,240)):
    full_path = os.path.join(data_root, filepath)

    # Determine if the file is RGB or IR and set the subfolder name accordingly
    if '_rgb' in filepath:
        file_type = 'rgb'
    elif '_ir' in filepath:
        file_type = 'ir'
    else:
        # Return the original filepath for non-RGB and non-IR files
        return filepath

    # Create the output directory for this file
    create_output_directory(filepath, file_type)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_filename = os.path.splitext(filepath)[0] + "_low_res.avi"
    out_full_path = os.path.join(output_dir, file_type, out_filename)
    out = cv2.VideoWriter(out_full_path, fourcc, 30.0, output_res)

    cap = cv2.VideoCapture(full_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame_resized = cv2.resize(frame, output_res, interpolation=cv2.INTER_AREA)

        # Write the resized frame
        out.write(frame_resized)

    cap.release()
    out.release()

    return os.path.join(file_type, out_filename)

def process_videos(file_list, output_res=(320, 240), output_file='rgb_ir_depth_skeleton_dataset_low_res.txt'):
    processed_lines = []
    progress_interval = 100  # Print a message after every 100 files processed
    last_processed_line = 0

    with open(file_list, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            # Skip already processed lines
            if i < last_processed_line:
                continue

            filepaths = line.strip().split()
            processed_filepaths = []

            for filepath in filepaths:
                # Generate the expected output file path
                if '_rgb' in filepath:
                    file_type = 'rgb'
                elif '_ir' in filepath:
                    file_type = 'ir'
                else:
                    processed_filepaths.append(filepath)
                    continue

                out_filename = os.path.splitext(filepath)[0] + "_low_res.avi"
                out_full_path = os.path.join(output_dir, file_type, out_filename)

                # Check if the output file already exists
                if not os.path.exists(out_full_path):
                    processed_path = process_video_file(filepath, output_res)
                    processed_filepaths.append(processed_path)
                else:
                    # If file exists, append the existing path
                    processed_filepaths.append(os.path.join(file_type, out_filename))

            processed_lines.append(" ".join(processed_filepaths))

            # Print progress
            if (i + 1) % progress_interval == 0:
                print(f"Processed {i + 1} lines...")

            # Save progress intermittently
            if (i + 1) % (progress_interval * 10) == 0:
                save_processed_lines(processed_lines, output_file)
                processed_lines = []

    return processed_lines


def save_processed_lines(processed_lines, output_file):
    with open(output_file, 'w') as file:
        for line in processed_lines:
            file.write(line + '\n')



# Usage

cv2.setNumThreads(10)
dataset_file = '/home/bas06400/Thesis/rgb_ir_depth_skeleton_dataset.txt'  # Replace with your dataset file path
processed_file_paths = process_videos(dataset_file)
save_processed_lines(processed_file_paths, 'rgb_ir_depth_skeleton_dataset_low_res.txt')
"""



data_root = "/net/polaris/storage/deeplearning/ntu"  # Original data root
output_dir = "/net/polaris/storage/deeplearning/ntu/lowres2"  # Directory to save processed videos

def create_output_directory(filepath, file_type):
    subdir_structure = os.path.dirname(filepath)
    output_subdir = os.path.join(output_dir, file_type, subdir_structure)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0  # Default frame rate if unable to open
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def process_video_file(filepath, output_res=(320,240)):
    full_path = os.path.join(data_root, filepath)
    if '_rgb' in filepath:
        file_type = 'rgb'
    elif '_ir' in filepath:
        file_type = 'ir'
    else:
        return filepath

    create_output_directory(filepath, file_type)

    # Capture original frame rate
    original_fps = get_frame_rate(full_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_filename = os.path.splitext(filepath)[0] + "_low_res.avi"
    out_full_path = os.path.join(output_dir, file_type, out_filename)
    out = cv2.VideoWriter(out_full_path, fourcc, original_fps, output_res)

    cap = cv2.VideoCapture(full_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, output_res, interpolation=cv2.INTER_AREA)
        out.write(frame_resized)

    cap.release()
    out.release()

    return os.path.join(file_type, out_filename)

def process_videos(file_list, output_res=(320, 240), output_file='rgb_ir_depth_skeleton_dataset_low_res.txt'):
    processed_lines = []
    progress_interval = 100
    last_processed_line = 0
    existing_files_count = 0
    new_files_count = 0

    with open(file_list, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i < last_processed_line:
                continue

            filepaths = line.strip().split()
            processed_filepaths = []

            for filepath in filepaths:
                if '_rgb' in filepath or '_ir' in filepath:
                    file_type = 'rgb' if '_rgb' in filepath else 'ir'
                    out_filename = os.path.splitext(filepath)[0] + "_low_res.avi"
                    out_full_path = os.path.join(output_dir, file_type, out_filename)

                    if not os.path.exists(out_full_path):
                        processed_path = process_video_file(filepath, output_res)
                        processed_filepaths.append(processed_path)
                        new_files_count += 1
                    else:
                        processed_filepaths.append(os.path.join(file_type, out_filename))
                        existing_files_count += 1
                else:
                    processed_filepaths.append(filepath)

            processed_lines.append(" ".join(processed_filepaths))
            if (i + 1) % progress_interval == 0:
                print(f"Processed {i + 1} lines...")
            if (i + 1) % (progress_interval * 10) == 0:
                save_processed_lines(processed_lines, output_file)
                processed_lines = []

    # Print the statistics
    print(f"Total existing files: {existing_files_count}")
    print(f"Total new files created: {new_files_count}")

    return processed_lines

def save_processed_lines(processed_lines, output_file):
    with open(output_file, 'w') as file:
        for line in processed_lines:
            file.write(line + '\n')

# Usage
cv2.setNumThreads(10)
dataset_file = '/home/bas06400/Thesis/rgb_ir_depth_skeleton_120dataset.txt'  # Replace with your dataset file path
processed_file_paths = process_videos(dataset_file)
save_processed_lines(processed_file_paths, 'rgb_ir_depth_skeleton_120dataset_low_res.txt')


#broken_file_path = "nturgb+d_rgb/S023C002P067R001A120_rgb.avi"  # Example file path
#processed_path = process_video_file(broken_file_path)
#print(f"Processed file saved at: {processed_path}")