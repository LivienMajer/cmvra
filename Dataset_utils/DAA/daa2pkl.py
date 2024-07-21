import csv
import numpy as np
import os
import pandas as pd

CSV_JOINTS = [
    'nose', 'lElbow', 'lWrist', 'rHeel', 'rHip', 'rSmallToe', 'neck', 'lSmallToe', 
    'rWrist', 'rAnkle', 'lHip', 'lHeel', 'lKnee', 'lEye', 'midHip', 'background', 
    'lEar', 'rElbow', 'rShoulder', 'rKnee', 'lShoulder', 'lBigToe', 'rEye', 'rEar', 
    'rBigToe', 'lAnkle'
]
"""
def read_openpose_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            joints = np.array([float(val) for i, val in enumerate(row[2:]) if (i+1) % 4 != 0]).astype(float).reshape(26, 3)
            data.append(joints)
    return np.array(data)
"""
def read_openpose_data(filepath, min_visible_joints=9):
    """
    Read OpenPose data from a file, using data from the previous frame when a frame does not meet 
    the minimum joint visibility requirement.

    :param filepath: Path to the CSV file containing OpenPose data.
    :param min_visible_joints: Minimum number of joints that must be visible for a frame to be included.
    :return: Array of joint data, with insufficiently visible frames replaced by the last valid frame's data.
    """
    data = []
    last_valid_joints = None  # Keep track of the last valid pose
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        next(reader)  # Skip header
        for row in reader:
            joints = np.array([float(val) for i, val in enumerate(row[2:]) if (i+1) % 4 != 0])
            
            if joints.size == 78:  # Ensure the row contains complete joint data
                joints = joints.reshape(26, 3)
                visible_joints = np.count_nonzero(joints, axis=0)
                #print(f'visible_joints:{np.mean(visible_joints)}')
                # Check if the current frame has a sufficient number of visible joints
                if int(np.mean(visible_joints)) >= min_visible_joints:
                    last_valid_joints = joints
                    data.append(joints)
                elif last_valid_joints is not None:
                    # If the current frame doesn't meet the visibility requirement,
                    # use the last valid pose data instead.
                    data.append(last_valid_joints)
                else:
                    # If no valid pose has been encountered yet, we may choose to append
                    # a frame of zeros or handle this case differently.
                    # Here, we append a frame filled with zeros as a placeholder.
                    data.append(np.zeros((26, 3)))
            else:
                # Handle incomplete joint data if necessary (e.g., append zeros or last valid data)
                pass
    
    return np.array(data)


def calculate_midpoint(joint1, joint2):
    return (joint1 + joint2) / 2

def get_joint(pose, joint_name):
    return pose[[i for i, name in enumerate(CSV_JOINTS) if name.startswith(joint_name)]][0]

def find_openpose_file(video_filename, openpose_dir):
    # Normalize the video filename to extract the base part
    base_name = video_filename.split('.')[0]
    
    return os.path.join(openpose_dir, base_name+ ".ids_1.openpose.3d.csv" )
    

def reorder_joints(poses):
    ntu_order = [
        'midHip', 'neck', 'neck', 'nose', 'lShoulder', 'lElbow', 'lWrist',
        'lWrist', 'rShoulder', 'rElbow', 'rWrist', 'rWrist', 'lHip', 'lKnee',
        'lAnkle', 'lBigToe', 'rHip', 'rKnee', 'rAnkle', 'rBigToe', 'neck',
        'lWrist', None, 'rWrist', None
    ]

    reordered = []
    for pose in poses:
        new_pose = []
        for joint in ntu_order:
            if joint:
                if joint == 'neck':  # Calculate the middle of the spine
                    mid_spine = calculate_midpoint(get_joint(pose, 'midHip'), get_joint(pose, 'neck'))
                    new_pose.append(mid_spine)
                elif joint == 'nose':  # Calculate a point for the head
                    head = get_joint(pose, 'nose')  
                    new_pose.append(head)
                elif joint == 'lBigToe':  # Average of left foot joints
                    lfoot = calculate_midpoint(get_joint(pose, 'lBigToe'), get_joint(pose, 'lHeel'))
                    new_pose.append(lfoot)
                elif joint == 'rBigToe':  # Average of right foot joints
                    rfoot = calculate_midpoint(get_joint(pose, 'rBigToe'), get_joint(pose, 'rHeel'))
                    new_pose.append(rfoot)
                elif joint == 'lWrist':  # Placeholder for the left hand
                    lhand = get_joint(pose, 'lWrist')  
                    new_pose.append(lhand)
                elif joint == 'rWrist':  # Placeholder for the right hand
                    rhand = get_joint(pose, 'rWrist')  
                    new_pose.append(rhand)
                else:
                    joint_data = get_joint(pose, joint)
                    new_pose.append(joint_data)
            else:
                new_pose.append(np.array([0.0, 0.0, 0.0]))
            #print(new_pose[-1])  # Placeholder for unavailable joint
        reordered.append(np.array(new_pose))
    return np.array(reordered)

def process_video_clips(clip_file, second_clip_file, openpose_dir, output_dir):
    # Load the second CSV file to get frame_start and frame_end from there
    df_second = pd.read_csv(second_clip_file)

    with open(clip_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for idx, row in enumerate(reader):
            # Debug: Print the current row to verify its contents
            #print("Current CSV row:", row)
            # Access the corresponding row from the second CSV
            try:
                # Attempt to get frame_start and frame_end from the second CSV
                frame_start_second, frame_end_second = df_second.loc[idx, ['frame_start', 'frame_end']]
            except KeyError:
                # If a KeyError occurs, fall back to using frame_start and frame_end from the first CSV
                print(f"KeyError encountered at row {idx}. Using frame_start and frame_end from the first CSV.")
                frame_start_second, frame_end_second = row[3], row[4]  # Adjust the indices if necessary
            #print("Current CSV row:", frame_start_second, frame_end_second)
            participant_id, file_id, annotation_id, start1, end1, activity, chunk_id = row
            # Use frame_start and frame_end from the second CSV
            #frame_start, frame_end = frame_start_second, frame_end_second
            new_file_id = file_id.replace("/", "_")

            # Debug: Print each variable to verify its contents
            #print(f"Participant ID: {participant_id}, File ID: {file_id}, Annotation ID: {annotation_id}, Chunk ID: {chunk_id}")
            # Construct the OpenPose file path
            # Find the corresponding OpenPose file
            openpose_file = find_openpose_file(file_id, openpose_dir)
            if not openpose_file:
                print(f"No OpenPose file found for {file_id}")
                continue 
            #print(f'Openpose_file:{openpose_file}')
            # Read the pose data
            poses = read_openpose_data(openpose_file)
            #print(f'Poses has the following shape:{poses.shape}')
            # Extract the frames for the current clip
            start, end = int(frame_start_second), int(frame_end_second) # skeleton data was captured using ir center mirror which has double the framerate of RGB
            
            clip_poses = poses[start:end]
            #print(clip_poses.shape)
            # Reorder the joints to match NTU format
            #print(f'Some Poses :{clip_poses[2]}')
            clip_poses = reorder_joints(clip_poses)
            #print(f'Some Poses after reorder :{clip_poses[2]}')
            # Check if the extracted poses match the expected dimensions
            #clip_poses = clip_poses[:,:25,:]
            #print(clip_poses.shape)
            if clip_poses.shape[1:] != (25, 3):
                print(clip_poses.shape)
                #raise ValueError("Unexpected shape of pose data.")
            
            # Save the data
            output_filename = f"{participant_id}_{new_file_id}_{start1}_{end1}_{annotation_id}_{chunk_id}"
            output_filepath = os.path.join(output_dir, output_filename)
            #print("Saving to:", output_filepath)  # Debug: Print the full file path before saving
            np.save(output_filepath, clip_poses)
            #break

def main():
    clip_file = "/home/bas06400/daa/activities_3s/kinect_color/midlevel.chunks_90.split_1.test.csv"
    clip_file_center_mirror = "/home/bas06400/daa/activities_3s/inner_mirror/midlevel.chunks_90.split_1.test.csv"
    openpose_dir = "/home/bas06400/daa/openpose_3d"
    output_dir = "/home/bas06400/daa/openpose_3d/clips"

    # Create a unique subdirectory for this execution
    output_dir = os.path.join(output_dir, 'test1')
    os.makedirs(output_dir, exist_ok=True)  
    process_video_clips(clip_file, clip_file_center_mirror,openpose_dir, output_dir)


if __name__ == "__main__":
    main()