import torch
from torch import nn
import torch.nn.functional as F


class DUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv', pad_type='zero'):
