import os
import cv2
import csv

class FrameExtractor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir

    def extract_frames(self, row, max_frames_per_chunk):
        participant_id, file_id, annotation_id, frame_start, frame_end, activity, chunk_id = row
        #participant_id,file_id,annotation_id,frame_start,frame_end,activity,object,location,chunk_id = row
        video_filepath = os.path.join(self.data_dir, file_id + '.mp4')
        new_file_id = file_id.replace("/", "_")

        # Create a unique identifier combining all available information
        unique_identifier = f"{participant_id}_{new_file_id}_{frame_start}_{frame_end}_{annotation_id}_{chunk_id}"
        
        # Update the output filename to include activity name and the unique identifier
        output_filename = f'{activity}_{unique_identifier}.avi'
        output_path = os.path.join(self.output_dir, output_filename)
        
        os.makedirs(self.output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_filepath)
        frame_count = 0
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # Capture first frame to get frame dimensions
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the first frame.")
            cap.release()
            return
        
        # Print frame dimensions
        height, width, channels = frame.shape
        print(f"Frame dimensions: Width={width}, Height={height}, Channels={channels}")

        # Use the dimensions from the frame to setup VideoWriter
        out = cv2.VideoWriter(output_path, fourcc, 15.0, (width, height))
        if not out.isOpened():
            print(f"Failed to create video writer for the output file {output_path}")
            cap.release()
            return
         
        try:
            # Set the starting frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))
            
            while frame_count < max_frames_per_chunk:
                ret, frame = cap.read()
                if not ret:
                    print(f"Frame number {frame_count + int(frame_start)} is missing.")
                    break

                # Resize the frame
                resized_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA) #optinal improves loading speed during training
                
                # Write the frame into the file 'output.avi'
                out.write(frame) #resized_frame
                
                frame_count += 1
                
                # Break the loop if we've reached the end frame
                if frame_count + int(frame_start) > int(frame_end):
                    break
                
        finally:
            # Release everything if job is finished
            cap.release()
            out.release()

def process_annotations(annotation_file, data_dir, root_dataset_dir, dataset_sub_dir, max_frames_per_chunk=48):
    with open(annotation_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        output_dir = os.path.join(root_dataset_dir, dataset_sub_dir)
        frame_extractor = FrameExtractor(data_dir, output_dir)
        
        for row in reader:
            frame_extractor.extract_frames(row, max_frames_per_chunk)

def main():
    data_dir = "/home/bas06400/daa/kinect_color"
    root_dataset_dir = "/home/bas06400/daa/kinect_color/clips"
    dataset_sub_dirs = ['train','val','test','train1','val1','test1','train2','val2','test2'] # Add 'train' and 'test' as needed ,'test1','train1','val1','test2','train2','val2'
    annotation_files = [
        '/home/bas06400/daa/activities_3s/kinect_depth/midlevel.chunks_90.split_0.train.csv',
        '/home/bas06400/daa/activities_3s/kinect_depth/midlevel.chunks_90.split_0.val.csv',
        '/home/bas06400/daa/activities_3s/kinect_depth/midlevel.chunks_90.split_0.test.csv',
        '/home/bas06400/daa/activities_3s/kinect_depth/midlevel.chunks_90.split_1.train.csv',
        '/home/bas06400/daa/activities_3s/kinect_depth/midlevel.chunks_90.split_1.val.csv',
        '/home/bas06400/daa/activities_3s/kinect_depth/midlevel.chunks_90.split_1.test.csv',
        '/home/bas06400/daa/activities_3s/kinect_depth/midlevel.chunks_90.split_2.train.csv',
        '/home/bas06400/daa/activities_3s/kinect_depth/midlevel.chunks_90.split_2.val.csv',
        '/home/bas06400/daa/activities_3s/kinect_depth/midlevel.chunks_90.split_2.test.csv'
        
        # Add other annotation files for 'train' and 'test' here
    ]
    
    for annotation_file, dataset_sub_dir in zip(annotation_files, dataset_sub_dirs):
        process_annotations(annotation_file, data_dir, root_dataset_dir, dataset_sub_dir)

if __name__ == "__main__":
    main()