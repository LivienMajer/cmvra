import torch
import torch.nn as nn
import torch.nn.functional as F

class FinalProjection(nn.Module):
    def __init__(self, input_dim, T, H, W, output_channels):
        super(FinalProjection, self).__init__()
        # input_dim: The number of features in each patch from the decoder output.
        # T: The temporal dimension (number of time steps) to reshape the patches into.
        # H, W: The spatial dimensions (height and width) to reshape the patches into.
        # output_channels: The number of channels in the final output (e.g., 3 for RGB images).

        self.input_dim = input_dim
        self.T = T
        self.H = H
        self.W = W
        self.output_channels = output_channels

        # Define the upscaling layers
        self.upscale_layers = nn.Sequential(
            # First, adjust the number of features from input_dim to 64.
            # This is a 3D convolution layer with a kernel size of 1, meaning it only changes the feature/channel dimension.
            nn.Conv3d(input_dim, 64, kernel_size=1),

            # Upsample the spatial dimensions (height and width).
            # ConvTranspose3d is a transposed convolution (or deconvolution) layer used for upsampling.
            # The stride and padding are configured to double the spatial dimensions.
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # Activation function for non-linearity

            # Another upsampling layer to further increase spatial dimensions.
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # Activation function

            # Finally, adjust the number of output channels to match the target (e.g., 3 for RGB images).
            # This is a 3D convolution with a kernel size of 1, altering the feature dimension to the desired number of output channels.
            nn.Conv3d(32, output_channels, kernel_size=1)
        )

    def forward(self, x):
        # The input x is expected to have a shape of [B, 1568, 384] from the decoder.

        # Reshape x to a 5D tensor [B, T, H, W, C].
        # This matches the desired temporal and spatial dimensions, and keeps the feature/channel dimension at the end.
        x = x.view(-1, self.T, self.H, self.W, self.input_dim)

        # Permute the tensor to bring the channel dimension to the second position.
        # The shape becomes [B, C, T, H, W], which is the standard format for 3D convolution operations.
        x = x.permute(0, 4, 1, 2, 3)

        # Apply the defined upscaling layers to the reshaped tensor.
        # The output will have the spatial dimensions upscaled and the channel dimension adjusted to output_channels.
        x = self.upscale_layers(x)

        return x