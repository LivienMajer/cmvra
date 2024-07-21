import os
import av
import torch
import numpy as np
from typing import Optional
from scipy.interpolate import interp1d
from PIL import Image
from torchvision import transforms
import cv2
import decord
from decord import VideoReader
from decord import cpu, gpu
from torchvision.io import read_video
from zeta.skeleton_transforms import RandomGaussianNoise, RandomRot, RandomScale, PreNormalize3D, Normalize3D
import logging
import re



class MultiModalVideoDataset3(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for handling multi-modal video data, including RGB, IR, depth, and skeleton modalities.

    This dataset is designed to work with different video datasets, particularly NTU and DAA, 
    and supports various data augmentation techniques. It also supports different views from DAA.

    Attributes:
        data_list (list): List of data samples.
        data_root (str): Root directory of the dataset.
        modalities (list): List of modalities to use.
        frame_count (int): Number of frames to sample from each video.
        random_sample (bool): Whether to use random sampling for frames.
        mixed_frames (dict): Dictionary specifying different frame counts for each modality.
        transform (callable): Transformation to apply to the video frames.
        random_rot (RandomRot): Random rotation augmentation for skeleton data.
        random_scale (RandomScale): Random scaling augmentation for skeleton data.
        random_noise (RandomGaussianNoise): Random noise augmentation for skeleton data.
        pre_normalize (PreNormalize3D): Normalization for skeleton data.
    """

    def __init__(self, list_path: str, data_root: str, modalities: list, frame_count=12, 
                random_sample=False, mode='train', mixed_frames=None, augs=False):
        """
        Initialize the MultiModalVideoDataset3.

        Args:
            list_path (str): Path to the file containing the list of data samples.
            data_root (str): Root directory of the dataset.
            modalities (list): List of modalities to use.
            frame_count (int, optional): Number of frames to sample from each video. Defaults to 12.
            random_sample (bool, optional): Whether to use random sampling for frames. Defaults to False.
            mode (str, optional): Dataset mode ('train', 'val', or 'test'). Defaults to 'train'.
            mixed_frames (dict, optional): Dictionary specifying different frame counts for each modality. Defaults to None.
            augs (bool, optional): Whether to apply augmentations. Defaults to False.
        """
        with open(list_path) as f:
            self.data_list = f.read().splitlines()

        self.data_root = data_root
        self.modalities = modalities
        self.frame_count = frame_count
        self.random_sample = random_sample
        self.mixed_frames = mixed_frames
        self.mode = mode
        
        
        if augs:
            logging.info('Applying Augmentations')
            if 'daa' in data_root:
                video_res=[540, 960]
            if 'ntu' in data_root:
                video_res=[1080, 1920]
            self.transform = init_transform_dict(video_res=video_res,
                                             input_res=[224, 224])[mode]
        else:
            self.transform = init_transform_dict_simple(video_res=[540, 960],
                                             input_res=[224, 224])[mode]
         
        # Initialize skeleton augmentations
        self.random_rot = RandomRot(theta=0.3)
        self.random_scale = RandomScale(scale=0.2)
        self.random_noise = RandomGaussianNoise(sigma=0.01)
        self.pre_normalize = PreNormalize3D() 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        This method loads and processes the data for all specified modalities for a given sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - dict: A dictionary with modalities as keys and processed data as values.
                - int: The label of the sample.
        """
        line = self.data_list[idx]
        # split on space preceded by 'i' since some activities from daa contain spaces
        paths = re.split(r'(?<=[iy]) | (?=n)', line)
        label = int(paths[-1])

        modality_indices = {"rgb": 0, "ir": 1, "depth": 2, "skeleton": 3 ,"ceiling": 4, "inner_mirror": 5, "a_column_co_driver": 6, "a_column_driver": 7, "steering_wheel": 8,"rgb2":0, "rgb3":0}
       

        modality_frames = {}
        full_path = os.path.join(self.data_root, paths[0])
        if self.mixed_frames:
            sample_indices_dict = {modality: self._determine_sample_indices(full_path, self.mixed_frames.get(modality, self.frame_count)) for modality in self.modalities}
        else:
            sample_indices = self._determine_sample_indices(full_path, self.frame_count)
            #sample_indices_dict = {
            #modality: [index * 2 for index in sample_indices] if modality != "rgb" and "daa" in full_path else sample_indices
            #for modality in self.modalities}
            
            sample_indices_dict = {modality: sample_indices for modality in self.modalities}


        for modality in self.modalities:
            index = modality_indices[modality]
            path = paths[index]
            sample_indices = sample_indices_dict[modality]  # Use the computed indices
            if path != 'None':  # Checking if the modality is available
                if modality == 'skeleton':
                    full_path = os.path.join(self.data_root, path)
                    
                    try:
                        # Attempt to load and sample the skeleton data
                        skeleton_data = self._load_skeleton_data(full_path)#[:, sample_indices, :, :]
                    except IndexError:
                        # Handle cases where sample_indices are out of bounds
                        # Creating a default array with shape (1, 12, 25, 3)
                        logging.info("There is a broken skeleton)")
                        skeleton_data = np.zeros((1, 12, 25, 3))
                    #if skeleton_data.ndim < 4:
                        # Add a new axis to ensure skeleton_data has 4 dimensions
                        #skeleton_data = skeleton_data[np.newaxis, :, :, :]
                    #skeleton_data = self.interpolate_clip(skeleton_data)

                    

                    # skeleton_data = Normalize3D()({'keypoint': skeleton_data})['keypoint']
                    if 'ntu' in full_path:
                        skeleton_data = self.pre_normalize({'keypoint': skeleton_data})['keypoint']
                        
                    # Check if skeleton_data already has 4 dimensions
                    if skeleton_data.ndim < 4:
                        # Add a new axis to ensure skeleton_data has 4 dimensions
                        skeleton_data = skeleton_data[np.newaxis, :, :, :]
                    #print(skeleton_data.shape)
                    # Apply augmentations only during training
                    if self.mode == 'train':    
                        if 'ntu' in full_path:
                            skeleton_data = self.random_rot({'keypoint': skeleton_data})['keypoint']
                            skeleton_data = self.random_scale({'keypoint': skeleton_data})['keypoint']
                            skeleton_data = self.random_noise({'keypoint': skeleton_data})['keypoint']
                    

                    # Convert to tensor and integrate
                    skeleton_tensor = torch.tensor(skeleton_data, dtype=torch.float32).permute(1, 0, 2, 3)
                    required_frame_count = 90
                    
                    current_frame_count = skeleton_tensor.shape[0]

                    if current_frame_count < required_frame_count:
                        # Calculate the number of frames to repeat
                        repeat_count = required_frame_count - current_frame_count
                        # Repeat the last frame
                        last_frame = skeleton_tensor[-1, :, :, :].unsqueeze(0)
                        repeated_frames = last_frame.repeat(repeat_count, 1, 1, 1)
                        # Concatenate the repeated frames to the skeleton data
                        skeleton_tensor = torch.cat((skeleton_tensor, repeated_frames), dim=0)
                    modality_frames['skeleton'] = skeleton_tensor
                else:
                    full_path = os.path.join(self.data_root, path)
                    modality_frames[modality] = self._extract_frames(full_path, sample_indices)


        return modality_frames, label
    
    def interpolate_clip(self, clip, num_frames=12):
        # Number of frames, vertices, and coordinates in the original clip
        #print(clip.shape)
        _, num_original_frames, num_vertices, num_coordinates = clip.shape

        # Initialize an array to hold the interpolated clip
        interpolated_clip = np.zeros((1 ,num_frames, num_vertices, num_coordinates))

        # Interpolate each coordinate of each vertex
        for vertex in range(num_vertices):
            for coordinate in range(num_coordinates):
                # Extract the series for the current vertex and coordinate across all frames
                y_series = clip[0,:, vertex, coordinate]
                #print(num_original_frames, y_series.shape)
                # Create an interpolator for this series
                interpolator = interp1d(np.arange(num_original_frames), y_series, axis=0, kind='linear')

                # Use the interpolator to fill in the interpolated values for this vertex and coordinate
                interpolated_clip[:,:, vertex, coordinate] = interpolator(
                    np.linspace(0, num_original_frames - 1, num_frames))

        return interpolated_clip


    def _determine_sample_indices(self, sample_path, num_frames):
        cap = cv2.VideoCapture(sample_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.random_sample:
            return np.random.choice(total_frames, num_frames, replace=True).tolist()
        else:
            return np.linspace(0, total_frames-2, num_frames).astype(int).tolist()
    
    def _extract_frames(self, path, sample_indices):
        if 'depth' in path and 'ntu' in path:
            # Handling depth data
            depth_images = sorted(os.listdir(path))
            # Ensure that the sample_indices are within the range of available images
            sample_indices = [i for i in sample_indices if i < len(depth_images)]
            
            extracted_frames = [self._load_depth_image(os.path.join(path, depth_images[idx])) for idx in sample_indices]
            del sample_indices, depth_images
            frames_tensor = torch.stack(extracted_frames)
            del extracted_frames
            frames_tensor = self.transform(frames_tensor.expand(-1, 3, -1, -1)) # change to three channel depth
        elif 'ir' or ('depth' and 'daa') in path:
            frames_tensor = self.load_video(path, sample_indices) # From NHWC to NCHW format
        else:
            frames_tensor = self.load_video(path, sample_indices)
            

        return frames_tensor
    
    def _load_skeleton_data(self, skeleton_path):
        if 'ntu' in skeleton_path:
            skeleton_data = np.load(skeleton_path, allow_pickle=True).item()
            # Assume using RGB skeleton data from the first body
            skel_body0 = skeleton_data.get('skel_body0', np.zeros((1000, 25, 3)))
            return skel_body0[np.newaxis, ...]
        elif 'daa' in skeleton_path:
            skeleton_data = np.load(skeleton_path, allow_pickle=True)
            #logging.info(f'skel frames {skeleton_data.shape}')
            if len(skeleton_data.shape) != 3:
                skeleton_data = np.zeros((300,25,3))
            return skeleton_data[np.newaxis, ...]
        else:
            raise ValueError("Unsupported skeleton data format or path incorrect.")

    def _load_depth_image(self, img_path):
        # Load a single depth image as a grayscale tensor
        image = Image.open(img_path).convert('L')  # Convert to grayscale ('L' mode)
        tensor_image = transforms.ToTensor()(image)
        return tensor_image
    
    def load_video(self, vis_path, sample_idx):
        

        video_tensor, _, _ = read_video(vis_path, start_pts=0, end_pts=None, pts_unit='sec')
        # video_tensor shape: (T, H, W, C)
        if max(sample_idx) >= video_tensor.size(0):
        # Select frames based on sample_idx
            img_array = torch.zeros(12, 224, 224, 3)
            print('empty file')
        else:
            img_array = video_tensor[sample_idx]
        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        return img_array



class MultiModalVideoDataset(torch.utils.data.Dataset):
    def __init__(self, list_path: str, data_root: str, modalities: list, active_modalities: Optional[list] = None, mean=None, std=None, spatial_size=224, use_advanced_processing=False, random_sample=False):
        with open(list_path) as f:
            self.data_list = f.read().splitlines()

        self.data_root = data_root
        self.modalities = modalities
        self.active_modalities = active_modalities if active_modalities else modalities
        self.random_sample = random_sample
        self.mean = mean if mean else torch.tensor([0.5, 0.5, 0.5])
        self.std = std if std else torch.tensor([0.5, 0.5, 0.5])
        self.spatial_size = spatial_size
        self.use_advanced_processing = use_advanced_processing

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line = self.data_list[idx]
        paths = line.split(' ')
        label = int(paths[-1])

        modality_indices = {"rgb": 0, "ir": 1, "depth": 2, "skeleton": 3}
        

        modality_frames = {}
        # Determine frame indices to sample
        sample_indices = self._determine_sample_indices(os.path.join(self.data_root, paths[0]))

        for modality in self.modalities:
            index = modality_indices[modality]
            path = paths[index]
            if path != 'None':  # Checking if the modality is available
                full_path = os.path.join(self.data_root, path)
                modality_frames[modality] = self._extract_frames(full_path, sample_indices)

                if self.use_advanced_processing:
                    frames_tensor = self._advanced_processing(modality_frames[modality])
                    modality_frames[modality] = frames_tensor

        return modality_frames, label
    

    def _determine_sample_indices(self, sample_path, use_random_sampling=True):
        # Open the video file with OpenCV
        cap = cv2.VideoCapture(sample_path)

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Release the video capture object
        cap.release()

        # Choose the sampling method based on the flag
        if use_random_sampling:
            return self._random_sample_frame_idx(total_frames)
        else:
            return self._deterministic_sample_frame_idx(total_frames)
    
    def _random_sample_frame_idx(self, length):
        # Ensure the random selection does not exceed the number of frames
        num_samples = min(12, length)

        # Randomly select unique frame indices
        frame_indices = np.random.choice(length, num_samples, replace=False)

        # Sort the indices to maintain correct sequence
        frame_indices.sort()

        return frame_indices.tolist()

    def _deterministic_sample_frame_idx(self, length):
        # Evenly sample 12 frames throughout the video
        return np.linspace(0, length-1, 12).astype(int).tolist()
    
    def _extract_frames(self, path, sample_indices):
        if 'depth' in path:
            # Handling depth data
            #print(path)
            depth_images = sorted(os.listdir(path))
            # Ensure that the sample_indices are within the range of available images
            sample_indices = [i for i in sample_indices if i < len(depth_images)]
            
            extracted_frames = [self._load_depth_image(os.path.join(path, depth_images[idx])) for idx in sample_indices]
            #print(len(extracted_frames))
            del sample_indices, depth_images
            frames_tensor = torch.stack(extracted_frames)
            del extracted_frames
            #print('depth',frames_tensor.shape)
        else:
            # Handling RGB and IR videos using OpenCV
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Adjust the sample_indices if needed
            sample_indices = [i for i in sample_indices if i < total_frames]

            extracted_frames = []
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    if 'ir' in path:
                        # Convert frame to grayscale for IR videos
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = np.expand_dims(frame, axis=-1)  # Add channel dimension
                    frame_tensor = transforms.ToTensor()(frame)
                    extracted_frames.append(frame_tensor)

            cap.release()

            # Stack the frames along the first dimension and permute dimensions to match PyTorch format
            frames_tensor = torch.stack(extracted_frames) # From NHWC to NCHW format

        return frames_tensor

    def _load_depth_image(self, img_path):
        # Load a single depth image as a grayscale tensor
        image = Image.open(img_path).convert('L')  # Convert to grayscale ('L' mode)
        tensor_image = transforms.ToTensor()(image)
        return tensor_image
    
    def _advanced_processing(self, frames_tensor):
        # Normalization
        if frames_tensor.size(1) == 1:  # Single-channel (IR or Depth)
            mean = torch.tensor([0.5])
            std = torch.tensor([0.5])
        else:  # Multi-channel (RGB)
            mean = self.mean
            std = self.std

        mean = mean.view(-1, 1, 1)  # Reshape for broadcasting
        std = std.view(-1, 1, 1)
        frames_tensor = (frames_tensor - mean) / std


        # Resizing and Cropping with modality-specific interpolation
        new_height, new_width = self.spatial_size, self.spatial_size
        if frames_tensor.size(1) == 3:
            interpolation_mode = 'bilinear'
        else:  # For 'ir' and 'depth'
            interpolation_mode = 'nearest'

        frames_tensor = torch.nn.functional.interpolate(
            frames_tensor, size=(new_height, new_width),
            mode=interpolation_mode, align_corners=False if interpolation_mode == 'bilinear' else None
        )

        return frames_tensor


class MultiModalVideoDataset2(torch.utils.data.Dataset):
    """
    A custom dataset class for handling multi-modal video data.

    This class is designed to work with datasets where each sample may include
    different modalities (e.g., RGB, depth, infrared) of video data. It supports
    loading and processing of these modalities based on the active modalities specified.

    Args:
        list_path (str): The path to the file containing the list of samples.
        data_root (str): The root directory of the dataset.
        modalities (list): A list of strings representing the modalities in the dataset.
        active_modalities (list, optional): A list of strings representing the modalities to be used.
        mean (torch.Tensor, optional): The mean for normalization.
        std (torch.Tensor, optional): The standard deviation for normalization.
        spatial_size (int): The size to which the frames should be resized.
        use_advanced_processing (bool): Whether to apply advanced processing to the frames.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the data corresponding to the index `idx`.
    """

    def __init__(self, list_path: str, data_root: str, modalities: list, active_modalities: Optional[list] = None, mean=None, std=None, spatial_size=224, use_advanced_processing=False):
        with open(list_path) as f:
            self.data_list = f.read().splitlines()

        self.data_root = data_root
        self.modalities = modalities
        self.active_modalities = active_modalities if active_modalities else modalities
        self.random_sample = False
        self.mean = mean if mean else torch.tensor([0.5, 0.5, 0.5])
        self.std = std if std else torch.tensor([0.5, 0.5, 0.5])
        self.spatial_size = spatial_size
        self.use_advanced_processing = use_advanced_processing

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line = self.data_list[idx]
        paths = line.split(' ')
        label = int(paths[-1])
        modality_frames = {}
        
        # Determine frame indices to sample
        sample_indices = self._determine_sample_indices(os.path.join(self.data_root, paths[0]))

        for modality, path in zip(self.modalities, paths[:-1]):
            if modality in self.active_modalities:
                full_path = os.path.join(self.data_root, path)
                modality_frames[modality] = self._extract_frames(full_path, sample_indices)
                
                if self.use_advanced_processing:
                    frames_tensor = self._advanced_processing(modality_frames[modality])
                    modality_frames[modality] = frames_tensor.permute(1,0,2,3)

        return modality_frames, label, idx
    

    def _determine_sample_indices(self, sample_path):
        """
        Determines the indices of frames to sample from a video.

        This method opens a video file, counts the total number of frames, and
        decides the frame indices to sample based on the total frame count. It
        currently uses random sampling strategy.

        Args:
            sample_path (str): The path to the video file.

        Returns:
            list: A list of frame indices to be sampled.
        """
        container = av.open(sample_path)
        total_frames = sum(1 for _ in container.decode(video=0))
        container.close()
        # determine sampling procedure
        if True:
            return self._random_sample_frame_idx(total_frames)
        else:
            # Implement a method to determine fixed frame indices if desired
            pass
    
    def _random_sample_frame_idx(self, length):
        """
        Randomly samples frame indices from a video.

        The method calculates frame indices to be sampled from a video of a given
        length. The strategy involves evenly spacing out the indices or listing all
        frames for short videos.

        Args:
            length (int): The total number of frames in the video.

        Returns:
            list: A list of randomly sampled frame indices.
        """
        frame_indices = np.linspace(0, length-1, 12).astype(int).tolist()
        return frame_indices
    """
    def _random_sample_frame_idx(self, length):
        frame_indices = []
        # Implement your random sampling logic here
        # For example:
        if length >= 7:
            frame_indices = list(range(0, length, length//7))
        else:
            frame_indices = list(range(length))
        return frame_indices
    """
    def _extract_frames(self, path, sample_indices):
        """
        Randomly samples frame indices from a video.

        The method calculates frame indices to be sampled from a video of a given
        length. The strategy involves evenly spacing out the indices or listing all
        frames for short videos.

        Args:
            length (int): The total number of frames in the video.

        Returns:
            list: A list of randomly sampled frame indices.
        """
        container = av.open(path)
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        extracted_frames = [frames[k] for k in sorted(frames.keys()) if k in sample_indices]

        # Check if the video is an IR video
        if 'ir' in path:
            tensor_frames = [torch.tensor(frame.to_ndarray()) for frame in extracted_frames]
            # Stack the tensors together
            frames_tensor = torch.stack(tensor_frames).unsqueeze(3)
        else:
            tensor_frames = [torch.tensor(frame.to_rgb().to_ndarray()) for frame in extracted_frames]

            # Stack the tensors together
            frames_tensor = torch.stack(tensor_frames)

        return frames_tensor

    def _advanced_processing(self, frames_tensor):
        """
        Applies advanced processing to a tensor of video frames.

        This method performs normalization, resizing, and cropping on the input
        frames tensor. It adjusts the normalization based on the number of channels
        and resizes the frames to a specified spatial size.

        Args:
            frames_tensor (torch.Tensor): A tensor of video frames.

        Returns:
            torch.Tensor: The processed tensor of video frames.
        """
        frames_tensor = frames_tensor.float() / 255.0
        #print(frames_tensor.shape)
        # Adjust normalization based on number of channels
        num_channels = frames_tensor.size(-1)
        if num_channels == 1:
            mean = torch.tensor([0.5])
            std = torch.tensor([0.5])
        else:
            mean = self.mean
            std = self.std

        frames_tensor = (frames_tensor - mean) / std

        

        # Resizing and Cropping
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # C, T, H, W
        if frames_tensor.size(-2) < frames_tensor.size(-1):
            new_width = self.spatial_size #frames_tensor.size(-1) * self.spatial_size // frames_tensor.size(-2)
            new_height = self.spatial_size
        else:
            new_height = self.spatial_size #frames_tensor.size(-2) * self.spatial_size // frames_tensor.size(-1)
            new_width = self.spatial_size
        frames_tensor = torch.nn.functional.interpolate(
            frames_tensor, size=(new_height, new_width),
            mode='bilinear', align_corners=False,
        )

        return frames_tensor  
    
class SingleFrameVideoDataset(MultiModalVideoDataset):
    def __init__(self, list_path: str, data_root: str, modalities: list, active_modalities: Optional[list] = None, mean=None, std=None, spatial_size=224, use_advanced_processing=False):
        # Call the constructor of the parent class
        super().__init__(list_path, data_root, modalities, active_modalities, mean, std, spatial_size, use_advanced_processing)

    def _determine_sample_indices(self, sample_path):
        container = av.open(sample_path)
        total_frames = sum(1 for _ in container.decode(video=0))
        container.close()
        # Just get a single frame index for sampling, you can choose how to pick this frame
        return [np.random.randint(0, total_frames)]  # Randomly select one index

    def _extract_frames(self, path, sample_indices):
        container = av.open(path)
        frame_index = sample_indices[0]  # Use the single frame index
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        # Extract the single frame using the index
        extracted_frame = frames.get(frame_index, None)

        if extracted_frame is None:
            raise ValueError("Frame index out of bounds")

        # Process the extracted frame
        if 'ir' in path:
            frame_array = extracted_frame.to_ndarray()
            # Add a channel dimension if it's missing (IR frames might be single-channel)
            if frame_array.ndim == 2:
                frame_array = np.expand_dims(frame_array, axis=-1)
            tensor_frame = torch.tensor(frame_array, dtype=torch.float32)
        else:
            tensor_frame = torch.tensor(extracted_frame.to_rgb().to_ndarray(), dtype=torch.float32)

        # The shape should be [channels, height, width], no batch dimension added here
        tensor_frame = tensor_frame.permute(2, 0, 1)  # Change from HWC to CHW

        return tensor_frame
    
    def _advanced_processing(self, frame_tensor):
        # Normalize the frame tensor
        frame_tensor = frame_tensor.float() / 255.0

        # Adjust normalization based on the number of channels
        num_channels = frame_tensor.shape[0]  # Changed from -1 to 0 assuming [C, H, W] format
        if num_channels == 1:
            mean = torch.tensor([0.5])
            std = torch.tensor([0.5])
        else:
            mean = torch.tensor(self.mean).view(-1, 1, 1)  # Adjust shape for broadcasting
            std = torch.tensor(self.std).view(-1, 1, 1)

        # Normalize the tensor
        frame_tensor = (frame_tensor - mean) / std

        # Resizing
        # No need for permutation as it should already be [C, H, W]
        if frame_tensor.size(1) < frame_tensor.size(2):
            new_height = self.spatial_size
            new_width = self.spatial_size
        else:
            new_height = self.spatial_size
            new_width = self.spatial_size

        # Resize the frame to the desired spatial size
        frame_tensor = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0),  # Add batch dimension for interpolation
            size=(new_height, new_width),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # Remove batch dimension after interpolation

        return frame_tensor

    def __getitem__(self, idx):
        line = self.data_list[idx]
        paths = line.split(' ')
        label = int(paths[-1])
        modality_frame = {}
        
        # Use the single frame index
        sample_index = self._determine_sample_indices(os.path.join(self.data_root, paths[0]))

        for modality, path in zip(self.modalities, paths[:-1]):
            if modality in self.active_modalities:
                full_path = os.path.join(self.data_root, path)
                frame = self._extract_frames(full_path, sample_index)
                
                if self.use_advanced_processing:
                    # Apply advanced processing that is expected to return the tensor in [C, H, W] format
                    frame = self._advanced_processing(frame)
                
                modality_frame[modality] = frame

        return modality_frame, label
    

def init_transform_dict(video_res=(240, 320),
                        input_res=(224, 224),
                        randcrop_scale=(0.4, 0.8),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    transform_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize([video_res[0], video_res[1]], antialias=True),
            transforms.CenterCrop([int(video_res[0]*0.6), int(video_res[1]*0.6)]),
            transforms.Resize(input_res, antialias=True),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize([video_res[0], video_res[1]], antialias=True),
            transforms.CenterCrop([int(video_res[0]*0.6), int(video_res[1]*0.6)]),
            transforms.Resize(input_res, antialias=True),
            normalize,
        ])
    }
    return transform_dict


def init_transform_dict_simple(video_res=(240, 320),
                        input_res=(224, 224),
                        randcrop_scale=(0.8, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711),
                        grey=False):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    if grey:
        norm_mean_gray = [0.5]
        norm_std_gray = [0.5]

        normalize = transforms.Normalize(mean=norm_mean_gray, std=norm_std_gray)
    transform_dict = {
        'train': transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_res),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_res),
            normalize,
        ])
    }
    return transform_dict