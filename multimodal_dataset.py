import os
import av
import torch
import numpy as np
from typing import Optional
from torchvision import transforms

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

        return modality_frames, label
    

    def _determine_sample_indices(self, sample_path):
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

        # Augmentation
        if self.random_sample and self.auto_augment is not None:
            aug_transform = create_random_augment(
                input_size=(frames_tensor.size(1), frames_tensor.size(2)),
                auto_augment=self.auto_augment,
                interpolation=self.interpolation,
            )
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # T, C, H, W
            frames_tensor = [transforms.ToPILImage()(frames_tensor[i]) for i in range(frames_tensor.size(0))]
            frames_tensor = aug_transform(frames_tensor)
            frames_tensor = torch.stack([transforms.ToTensor()(img) for img in frames_tensor])
            frames_tensor = frames_tensor.permute(0, 2, 3, 1)

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