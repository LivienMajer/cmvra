import cv2
import os
import numpy as np
import glob
from PIL import Image


def video2image(v_p, output_path, mask_root, video_type='rgb'):
    # Determine the crop size based on the video type
    if video_type == 'rgb':
        crop_size = 640  # Crop size for RGB
        resize_to = (320, 320)  # Resize resolution for RGB
        start = 8
    elif video_type == 'ir':
        crop_size = 251  # Crop size for IR calculated previously
        resize_to = (320, 320)  # Resize resolution for IR
        start = 7
    else:
        raise ValueError("Invalid video type: choose 'rgb' or 'ir'")
    img_path = os.path.join(output_path, v_p[:-4].split('/')[-1])
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cap = cv2.VideoCapture(v_p)
    suc, frame = cap.read()
    frame_count = 1
    while suc:
        
        mask_path = os.path.join(mask_root, v_p[:-start].split('/')[-1], 'MDepth-%08d.png' % frame_count)
        mask = cv2.imread(mask_path)
        if mask is not None:
            mask = mask * 255
            w, h, c = mask.shape
            h2, w2, _ = frame.shape
            ori = frame
            frame = cv2.resize(frame, (h, w))
            h1, w1, _ = frame.shape

            #print("Original frame size:", h2, w2)
            #print("Resized frame size:", h1, w1)
            #print("Mask size:", w, h, c)


            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
            mask = cv2.dilate(mask, np.ones((10, 10), np.uint8))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # Save the processed mask for inspection
            #mask_save_path = os.path.join(img_path, 'mask_{:0>6d}.png'.format(frame_count))
            #success = cv2.imwrite(mask_save_path, mask)
            
            
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            Idx = []
            for i in range(len(contours)):
                Area = cv2.contourArea(contours[i])
                if Area > 500:
                    Idx.append(i)

            centers = []
            for i in Idx:
                rect = cv2.minAreaRect(contours[i])
                center, (h, w), degree = rect
                centers.append(center)

            if centers:
                final_center = np.int0(np.array(centers))
                c_x = min(final_center[:, 0])
                c_y = min(final_center[:, 1])

                center = (c_x, c_y)
                center_new = resize_pos(center, (h1, w1), (h2, w2))

                #print("Center before resize:", center)
                #print("Center after resize:", center_new)
                left = max(center_new[0] - crop_size // 2, 0)
                top = max(center_new[1] - crop_size // 2, 0)
                right = min(left + crop_size, w2)
                bottom = min(top + crop_size, h2)
                rect = (left, top, right, bottom)
                image = Image.fromarray(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB))
                #print("Cropping rectangle:", rect)

                image = image.crop(rect)
                image = image.resize(resize_to, Image.ANTIALIAS)
                image.save('{}/{:0>6d}.jpg'.format(img_path, frame_count))
                
            frame_count += 1
            suc, frame = cap.read()
    cap.release()

def resize_pos(center, original_size, new_size):
    # Scaling the position of the center based on the size change
    oy, ox = original_size
    ny, nx = new_size
    scale_x = nx / ox
    scale_y = ny / oy
    return int(center[0] * scale_x), int(center[1] * scale_y)


def process_all_videos(video_root, output_root, mask_root, viedeo_type):
    video_files = glob.glob(os.path.join(video_root, '*.avi'))  # Assuming .avi format, change if needed
    print(len(video_files))
    for v_file in video_files:
        print(f"Processing {v_file}")
        try:
            output_path = output_root
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            video2image(v_file, output_path, mask_root, viedeo_type)
        except Exception as e:
            print(f"Error processing {v_file}: {e}")

# Paths
data_root = "/net/polaris/storage/deeplearning/ntu"
video_root = os.path.join(data_root, "nturgb+d_ir")
output_root = "/net/polaris/storage/deeplearning/ntu/nturgb+d_ir_cropped_320_320"
mask_root = os.path.join(data_root, "nturgb+d_depth_masked")

process_all_videos(video_root, output_root, mask_root, 'ir')