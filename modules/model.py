import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.unet import UNet


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