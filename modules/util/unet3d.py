import torch
from torch import nn
import torch.nn.functional as F


class UNet3d(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv', pad_type='zero'):
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        assert pad_type in ('zero', 'replicate')
        self.padding = padding
        self.pad_type = pad_type
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm, pad_type))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm, pad_type))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv3d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool3d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, pad_type):
        super(UNetConvBlock, self).__init__()
        block = []

        if padding:
            if pad_type == 'replicate':
                block.append(nn.ReplicationPad3d(int(padding)))
            else:
                block.append(nn.ConstantPad3d(int(padding),0.))

        block.append(nn.Conv3d(in_size, out_size, kernel_size=3))
        block.append(nn.ReLU())

        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        if padding:
            if pad_type == 'replicate':
                block.append(nn.ReplicationPad3d(int(padding)))
            else:
                block.append(nn.ConstantPad3d(int(padding),0.))

        block.append(nn.Conv3d(out_size, out_size, kernel_size=3))
        block.append(nn.ReLU())

        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, pad_type):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv3d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, pad_type)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width, layer_depth = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        diff_z = (layer_depth - target_size[2]) // 2
        return layer[:, :, 
                     diff_y:(diff_y + target_size[0]), 
                     diff_x:(diff_x + target_size[1]),
                     diff_z:(diff_z + target_size[2])]

    '''
    def pad_match(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0])
        diff_x = (layer_width - target_size[1])
        left = right = top = bot = 0
        if diff_y % 2 == 0: # even:
            top = bot = diff_y // 2
        else:
            top = diff_y // 2
            bot = top + 1
        if diff_x % 2 == 0: # even:
            left = right = diff_x // 2
        else:
            left = diff_x // 2
            right = left + 1
        return left, right, top, bot
    '''

    def forward(self, x, bridge):
        up = self.up(x)
        #left, right, top, bot = self.pad_match(bridge, up.shape[2:])
        #up = F.pad(up, (left,right,top,bot), mode="replicate")
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out



