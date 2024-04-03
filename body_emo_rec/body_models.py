import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # Use inplace=True to save memory
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelWiseConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ChannelWiseConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        #self.conv.bias = nn.Parameter(self.conv.bias.double())
        
    def forward(self, x):
        
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights and bias
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights with a uniform distribution U[-a, a]
        a = (6 / (self.fc1.in_features + self.fc1.out_features)) ** 0.5
        nn.init.uniform_(self.fc1.weight, -a, a)
        nn.init.uniform_(self.fc2.weight, -a, a)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # Squeeze operation
        squeeze = self.avg_pool(x)
        squeeze = squeeze.view(squeeze.size(0), -1)  # Flatten

        # Excitation operation
        excitation = F.relu(self.fc1(squeeze))
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)

        # Scale the original features
        output = x * excitation.unsqueeze(2).unsqueeze(3)

        return output
