import torch
import torch.nn as nn

# Define your input tensor with shape (batch_size, in_channels, height, width)
input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with 3 input channels

# Define the number of input and output channels
in_channels = 3
out_channels = 3  # Same as the number of input channels for channel-wise convolution

# Define the kernel size and padding
kernel_size = 3
padding = 1

# Perform channel-wise convolution
conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=in_channels)

# Apply convolution to the input tensor
output_tensor = conv(input_tensor)

# Print the shapes of input and output tensors
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
