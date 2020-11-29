import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovidSegNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, wf=6, padding=True):
        super().__init__()

        self.encoder = nn.ModuleList()

        self.encoder.append(ResBlock(in_channels, 2**wf, padding))

        prev_channels = 2**wf
        for i in range(1,3):
            self.encoder.apppend(
                nn.Sequential([
                    FVBlock(prev_channels, 2**(wf+i), padding),
                    ResBlock(2**(wf+i), 2**(wf+i), padding)
                ])
            )
            prev_channels = 2**(wf+i)

        self.encoder.apppend(
            nn.Sequential([
                FVBlock(prev_channels, 2**(wf+3), padding),
                PASPPBlock(2**(wf+3), padding)
            ])
        )
        prev_channels = 2**(wf+3)

        self.decoder = nn.ModuleList()
        for i in reversed(range(3)):
            self.decoder.append(UpBlock(prev_channels, 2**(wf+i), padding))
            prev_channels = 2**(wf+i)

        self.last = ConvBlock(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for i, encode in enumerate(self.encoder):
            x = encode(x)
            if i != len(self.encoder)-1:
                skip_connections.append(x)
                x = F.avg_pool3d(x, 2)

        for i, decode in enumerate(self.decoder):
            x = decode(x, skip_connections[-i-1])

        x = self.last(x)
        return F.softmax(x, dim=1)


class ConvBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size, 
                 padding=False, dilation=1):
        self.block = nn.Sequential([
            nn.Conv3d(in_size, out_size, kernel_size=kernel_size, 
                      padding=int(padding), dilation=dilation),
            nn.BatchNorm3d(out_size)
        ])

    def forward(self, input):
        x = self.block(input)
        return F.relu(x)
        
class ResBlock(nn.Module):

    def __init__(self, in_size, out_size, padding):
        super().__init__()

        block = []

        block.append(ConvBlock(in_size, out_size, kernel_size=3, 
                               padding=int(padding)))
        block.append(nn.Conv3d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.BatchNorm3d(out_size))

        self.block = nn.Sequential(block)

        self.skip_layers = nn.ModuleList()
        if in_size > out_size:
            self.skip_layers.append(nn.Conv3d(in_size, out_size, kernel_size=1))
            self.skip_layers.append(nn.BatchNorm3d(out_size))

    def forward(self, input):
        x = self.block(input)
        skip_x = input
        for layer in self.skip_layers:
            skip_x = layer(skip_x)
        return F.relu(x + skip_x)


class FVBlock(nn.Module):
    '''
    Feature Variation Block
    Lesions and chest boundaries are convoluted so feature variation block
    tries to tackle this problem by implicitly enhancing contrast and 
    adjusting intensity at the feature level automatically
    '''

    def __init__(self, in_size, out_size, padding):
        super().__init__()
        self.conv1 = ConvBlock(in_size, out_size, kernel_size=1)
        self.con_branch = ContrastEnhancementBranch(out_size)
        self.pos_branch = PositionSensitiveBranch(out_size)
        self.conv3 = ConvBlock(out_size*3, out_size, kernel_size=3,
                               padding=int(padding))

    def forward(self, input):
        fv1 = self.conv1(input)
        fc = self.con_branch(fv1)
        fp = self.pos_branch(fv1)
        f_conca = torch.cat([fv1, fc, fp], 1)
        fv3 = self.conv3(f_conca)
        return input + fv3


class ContrastEnhancementBranch(nn.Module):

    def __init__(self, in_size):
        super().__init__()

        self.in_size = in_size
        out_size = in_size // 2
        self.lin1 = nn.Linear(in_size, out_size)
        self.lin2 = nn.Linear(out_size, 1)

    def forward(self, fv1):
        gap = F.avg_pool3d(fv1, kernel_size=fv1.shape[-3:]).squeeze()
        x = self.lin1(gap)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        fg = x.repeat(self.in_size) # expansion

        return fg * fv1

class PositionSensitiveBranch(nn.Module):

    def __init__(self, in_size, padding):
        super().__init__()
        self.conv3_1 = nn.Conv3d(in_size, in_size, kernel_size=3,
                                 padding=int(padding))
        self.conv3_2 = nn.Conv3d(in_size, in_size, kernel_size=3,
                                 padding=int(padding))
        
    def forward(self, fv1):
        x = self.conv3_1(fv1)
        x = F.relu(x)
        x = self.conv3_2(x)
        attention_map = torch.sigmoid(x)
        return attention_map * fv1


class PASPPBlock(nn.Module):
    '''
    Progressive Atrous Spatial Pyramid Pooling
    '''
    def __init__(self, in_size, padding):
        super().__init__()
        out_size = in_size // 4

        self.conv1_1 = ConvBlock(in_size, out_size, kernel_size=1)
        self.conv1_2 = ConvBlock(in_size, out_size, kernel_size=1)
        self.conv1_3 = ConvBlock(in_size, out_size, kernel_size=1)
        self.conv1_4 = ConvBlock(in_size, out_size, kernel_size=1)

        self.atrous_1 = ConvBlock(out_size, out_size, kernel_size=3,
                                 padding=int(padding), dilation=1)
        self.atrous_2 = ConvBlock(out_size, out_size, kernel_size=3,
                                 padding=int(padding), dilation=2)
        self.atrous_3 = ConvBlock(out_size, out_size, kernel_size=3,
                                 padding=int(padding), dilation=4)
        self.atrous_4 = ConvBlock(out_size, out_size, kernel_size=3,
                                 padding=int(padding), dilation=8)

        out_size_prime = in_size // 2

        self.conv1_5 = ConvBlock(out_size_prime, out_size_prime, kernel_size=1)
        self.conv1_6 = ConvBlock(out_size_prime, out_size_prime, kernel_size=1)

        self.conv1_7 = ConvBlock(in_size, in_size, kernel_size=1)

    def forward(self, input):
        fp1 = self.conv1_1(input)
        fp2 = self.conv1_2(input)
        fp3 = self.conv1_3(input)
        fp4 = self.conv1_4(input)

        fd1 = self.atrous_1(fp1)
        fd2 = self.atrous_2(fp2)
        fd3 = self.atrous_3(fp3)
        fd4 = self.atrous_4(fp4)

        fd1_prime = fd1 + fp1 + fp2
        fd2_prime = fd2 + fp1 + fp2
        fd3_prime = fd3 + fp3 + fp4
        fd4_prime = fd4 + fp3 + fp4

        fd1_dbl_prm = self.conv1_5(torch.cat([fd1_prime, fd2_prime], 1))
        fd2_dbl_prm = self.conv1_6(torch.cat([fd3_prime, fd4_prime], 1))

        fp_out = self.conv1_7(torch.cat([fd1_dbl_prm, fd2_dbl_prm], 1))

        return fp_out



class UpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_size, out_size, 
                                         kernel_size=2, stride=2)
        self.res = ResBlock(in_size, out_size, padding)

    def forward(self, input, skip_con):
        up = self.upconv(input)
        out = torch.cat([up, skip_con], 1)
        return self.res(out)



