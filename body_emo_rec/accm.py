import torch
import torch.nn as nn
import body_models
import torch.nn.functional as F

class Architecture(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block1 = body_models.ConvolutionBlock(in_channels=config["Joints"], out_channels=config["block1_filters"], kernel_size=(3, 3), stride=1, padding=1)
        self.block2 = body_models.ConvolutionBlock(in_channels=config["block1_filters"], out_channels=config["block2_filters"], kernel_size=(3, 3), stride=1, padding=1)
        self.cwcl = body_models.ChannelWiseConvolution(in_channels=config["Joints"], out_channels=config["Joints"], kernel_size=(3, 3), stride=1, padding=1)
        self.attention = body_models.SEBlock(in_channels=config["block1_filters"], reduction_ratio=config["reduction_ratio"])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(config["block2_filters"], config["out_classes"])

    def forward(self, x):
        o1 = self.cwcl(x)
        o1 = self.block1(o1)
        o1 = self.attention(o1)
        o1 = self.block2(o1)
        
        o2 = self.block1(x)
        o2 = self.block2(o2)
        
        output = o1 + o2
        output = self.pool(output)
        output = output.squeeze(-1).squeeze(-1)
        output = self.fc(output)
        output = F.softmax(output, dim=-1)
        return output
    
# x = torch.randn(1, 33, 100, 2)
# print(x)    
# accm = Architecture(config = {"Joints": 33, "block1_filters": 128, "block2_filters": 256, "reduction_ratio": 16, "out_classes": 8})
# y = accm(x)
# print(type(y))
# print(y.shape)
# print(y)