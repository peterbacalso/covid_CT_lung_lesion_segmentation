import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.util.covid_seg_net import CovidSegNet 

class CovidSegNetWrapper(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.input = nn.BatchNorm3d(kwargs['in_channels'])
        self.segnet = CovidSegNet(**kwargs)

        self.init_weights()

    def forward(self, inputs):
        x = self.input(inputs)
        return self.segnet(x)

    def init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d, nn.ConvTranspose3d}:
                if type(m) in {nn.Linear}:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                else:
                    nn.init.kaiming_normal_(m.weight.data, a=0, 
                                            mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)



