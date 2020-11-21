import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.util.unet import UNet
from modules.util.unet3d import UNet3d


class UNetWrapper(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.input = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)

        self.init_weights()

    def forward(self, inputs):
        x = self.input(inputs)
        x = self.unet(x)
        return torch.sigmoid(x)

    def init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.ConvTranspose2d}:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

class UNet3dWrapper(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.input = nn.BatchNorm3d(kwargs['in_channels'])
        self.unet3d = UNet3d(**kwargs)

        self.init_weights()

    def forward(self, inputs):
        x = self.input(inputs)
        x = self.unet3d(x)
        return torch.sigmoid(x)

    def init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv3d, nn.ConvTranspose3d}:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)



