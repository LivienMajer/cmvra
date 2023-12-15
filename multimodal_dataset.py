import os
import av
import torch
import numpy as np
from typing import Optional
from torchvision import transforms
from PIL import Image
import cv2
import torchvision

from evl.video_dataset.transform import create_random_augment, random_resized_crop

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
        self.original_video_size = (1920, 1080)

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
            full_path = os.path.join(self.data_root, path)
            if modality in self.active_modalities:
                if modality == 'skeleton':
                    skeleton_data = self.load_skeleton_data(full_path)
                    sampled_skeleton_data = skeleton_data[sample_indices, :, :]
                    scaled_skeleton_data = self.scale_skeleton_data(sampled_skeleton_data)
                    normalized_skeleton_data = self.normalize_skeleton_data(scaled_skeleton_data)
                    modality_frames[modality] = normalized_skeleton_data.unsqueeze(1)
                else:
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
        frame_indices = np.linspace(0, length-1, 8).astype(int).tolist()
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
            # Handling RGB and IR videos using torchvision
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
    
    def load_skeleton_data(self, skeleton_path):
        skeleton_data = np.load(skeleton_path, allow_pickle=True).item()
        # Assume using RGB skeleton data from the first body
        skel_body0 = skeleton_data.get('rgb_body0', np.zeros((1, 25, 3)))
        return torch.tensor(skel_body0, dtype=torch.float32)

    def scale_skeleton_data(self, skeleton_data):
        original_width, original_height = self.original_video_size
        target_width, target_height = self.spatial_size, self.spatial_size

        # Calculate separate scaling factors for width and height
        scale_factor_x = target_width / original_width
        scale_factor_y = target_height / original_height

        # Apply scaling factors to x and y coordinates directly
        # Assuming skeleton_data shape is [nframes, njoints, 2] with last dimension being (x, y)
        skeleton_data[:, :, 0] *= scale_factor_x  # Scale x coordinates
        skeleton_data[:, :, 1] *= scale_factor_y  # Scale y coordinates

        return skeleton_data

    
    def _load_depth_image(self, img_path):
        # Load a single depth image as a grayscale tensor
        image = Image.open(img_path).convert('L')  # Convert to grayscale ('L' mode)
        tensor_image = transforms.ToTensor()(image)
        return tensor_image
    
    def normalize_skeleton_data(self, skeleton_data):
        # Assuming skeleton_data is scaled to have coordinates in the range [0, 224]
        # Normalize to the range [0, 1]
        normalized_data = ((skeleton_data / 224.0) - 0.5) * 2
        return normalized_data
    
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

        # Augmentation
        if self.random_sample and self.auto_augment is not None:
            aug_transform = create_random_augment(
                input_size=(frames_tensor.size(2), frames_tensor.size(3)),  # Height and Width
                auto_augment=self.auto_augment,
                interpolation=self.interpolation,
            )
            # Apply augmentation to each frame
            frames_list = [transforms.ToPILImage()(frame) for frame in frames_tensor]
            augmented_frames = [aug_transform(frame) for frame in frames_list]
            frames_tensor = torch.stack([transforms.ToTensor()(frame) for frame in augmented_frames])

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
                modality_frame[modality] = self._extract_frames(full_path, sample_index)
                
                if self.use_advanced_processing:
                    frame_tensor = self._advanced_processing(modality_frame[modality])
                    modality_frame[modality] = frame_tensor.permute(1,0,2,3)

        return modality_frame, label