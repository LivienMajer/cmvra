import cv2
import os
import numpy as np
import glob
from PIL import Image

def process_depth_images_per_video(video_folder, output_root):
    # Iterate over each video directory
    for dir_name in os.listdir(video_folder):
        video_dir_path = os.path.join(video_folder, dir_name)

        # Check if it's a directory
        if os.path.isdir(video_dir_path):
            output_dir_path = os.path.join(output_root, dir_name)
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)

            # Process each image in the video directory
            image_files = glob.glob(os.path.join(video_dir_path, '*.png'))
            for img_file in image_files:
                process_single_depth_image(img_file, output_dir_path)

def process_single_depth_image(img_file, output_dir_path):
    try:
        # Read the depth image
        depth_image = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        # Ensure the image is in 8-bit format
        if depth_image.dtype != np.uint8:
            # Normalize and convert to 8-bit format
            depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # Create a binary mask for contour detection
        _, binary_mask = cv2.threshold(depth_image, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour and extract ROI from the original depth image
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            square_size = max(w, h)
            square_x = max(x + w // 2 - square_size // 2, 0)
            square_y = max(y + h // 2 - square_size // 2, 0)
            square_x = min(square_x, depth_image.shape[1] - square_size)
            square_y = min(square_y, depth_image.shape[0] - square_size)
            roi = depth_image[square_y:square_y + square_size, square_x:square_x + square_size]
            roi = cv2.resize(roi, (320, 320))
            roi_save_path = os.path.join(output_dir_path, f'{os.path.basename(img_file)[:-4]}.png')
            cv2.imwrite(roi_save_path, roi)

    except Exception as e:
        print(f"Error processing {img_file}: {e}")

cv2.setNumThreads(10)


# Paths for depth videos
depth_video_root = "/net/polaris/storage/deeplearning/ntu/nturgb+d_depth_masked"  # Update this to the path where depth videos' directories are stored
depth_output_root = "/net/polaris/storage/deeplearning/ntu/nturgb+d_depth_masked_cropped_320_320"  # Update this to the path where you want to save the ROIs

directory_count = 0
if os.path.exists(depth_video_root):
    for entry in os.listdir(depth_video_root):
        if os.path.isdir(os.path.join(depth_video_root, entry)):
            directory_count += 1

print(directory_count)

process_depth_images_per_video(depth_video_root, depth_output_root)
