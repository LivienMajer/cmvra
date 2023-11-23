import os
import av
import torch
import numpy as np
from typing import Optional
from PIL import Image
from torchvision import transforms
import cv2

class MultiModalVideoDataset(torch.utils.data.Dataset):
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
                    modality_frames[modality] = frames_tensor

        return modality_frames, label
    

    def _determine_sample_indices(self, sample_path):
        # Open the video file with OpenCV
        cap = cv2.VideoCapture(sample_path)

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Release the video capture object
        cap.release()
        
        # determine sampling procedure
        if True:
            return self._random_sample_frame_idx(total_frames)
        else:
            # Implement a method to determine fixed frame indices if desired
            pass
    
    def _random_sample_frame_idx(self, length):
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
        if 'depth' in path:
            # Handling depth data
            depth_images = sorted(os.listdir(path))
            # Ensure that the sample_indices are within the range of available images
            sample_indices = [i for i in sample_indices if i < len(depth_images)]
            
            extracted_frames = [self._load_depth_image(os.path.join(path, depth_images[idx])) for idx in sample_indices]
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