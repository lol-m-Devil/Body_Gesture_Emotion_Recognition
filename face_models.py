import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import Any, Callable, List, Optional, Type, Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)  # Convert feature maps to 1 channel
        self.softmax = nn.Softmax(dim=1)  # Softmax along the channel dimension

    def forward(self, x1, x2):
        # Compute attention weights
        att_weights = self.compute_attention(x1, x2)

        # Weighted fusion
        fused_representation = att_weights[:, 0].unsqueeze(1) * x1 + att_weights[:, 1].unsqueeze(1) * x2

        return fused_representation

    def compute_attention(self, x1, x2):
        # Convert feature maps to 1 channel
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)

        # Concatenate and apply softmax to get attention weights
        concat_features = torch.cat((x1, x2), dim=1)
        att_weights = self.softmax(concat_features)

        return att_weights

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class Conv1x3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=True):
        super(Conv1x3x3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                              stride=(1, stride, stride), padding=(0, padding, padding), bias=bias)

    def forward(self, x):
        return self.conv(x)
    
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SISTCM(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = Conv1x3x3(planes, planes)
        self.stride = stride
        self.fusion = FeatureFusion(in_channels=planes) #Not Confirmed what is input channel
        self.downsample = downsample
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x  # T x C x H x W 
        
        T, C, H, W = x.size()
        
        superimage_wt = x.permute(1, 2, 0, 3).contiguous().view(C, H, T*W)   # channels x H x (T*W)
        superimage_ht = x.permute(1, 0, 2, 3).contiguous().view(C, T*H, W)  # channels x (T*H) x W
        
        out_wt = self.conv1(superimage_wt)  # channels x H x (T*W)
        out_ht = self.conv1(superimage_ht)  # channels x (T*H) x W
        
        C, temp, W = out_ht.size()
        H = W
        T = temp // H 
        
        out_wt = out_wt.view(C, H, T, W).permute(2, 0, 1, 3)  # T x C x H x W
        out_ht = out_ht.view(C, T, H, W).permute(1, 0, 2, 3)  # T x C x H x W
        
        out = self.fusion(out_wt, out_ht) # T x C x H x W        
        out = out.permute(1,0,2,3)
        out = self.conv2(out)
        out = out.permute(1,0,2,3)
        
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        # print(out.shape)
        return out

class SpatialAveragePooling(nn.Module):
    def __init__(self):
        super(SpatialAveragePooling, self).__init__()

    def forward(self, x):
        #dimension of x --> N x m x s
        pooled = torch.mean(x, dim=(2,3))
        return pooled
    
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[SISTCM],
        layers: List[int],
        num_classes: int = 8,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #Replace Block by SISTCM Block 
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        #Make it 3d layer actually
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SISTCM) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[SISTCM]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)  -> 1 x 64 x 56 x 56     

        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)

        #ADD 3D ADAPTIVE POOLING
        x = x.permute(1, 0, 2, 3)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        LST_features = x
        LST_features = LST_features.squeeze()
        #ADD FULLY CONNECTED LAYER
        x = x.permute(1, 0)
        x = self.fc(x)
        clip_emotion_level = x.permute(1, 0)
        clip_emotion_level = clip_emotion_level.squeeze()
                
        return LST_features, clip_emotion_level

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[SISTCM],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    
    model = ResNet(block, layers, **kwargs)

    return model


class BiLSTM(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size1, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2 * 2, output_size)  # Multiply by 2 due to bidirectional LSTM for both vectors

    def forward(self, x1, x2):
        h0 = torch.zeros(self.num_layers * 2, x1.size(0), self.hidden_size).to(device)  # Initial hidden state
        c0 = torch.zeros(self.num_layers * 2, x1.size(0), self.hidden_size).to(device)  # Initial cell state
        out1, _ = self.lstm1(x1, (h0, c0))  # Forward pass through the first LSTM
        out2, _ = self.lstm2(x2, (h0, c0))  # Forward pass through the second LSTM
        out = torch.cat((out1[:, -1, :], out2[:, -1, :]), dim=1)  # Concatenate the outputs of both LSTMs
        out = self.fc(out)  # Fully connected layer
        out = F.softmax(out, dim=1)  # Softmax activation function
        return out