import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovidSegNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, depth=4, wf=6, padding=True):
        super().__init__()

        self.encoder = nn.ModuleList()
        for i in range(depth):
            blocks = []
            channel_size = 2**(wf+i)
            if i > 0:
                blocks.append(FVBlock(channel_size, padding))
            if i < depth - 1:
                blocks.append(ResBlock(in_channels if i == 0 else channel_size, 
                                       channel_size, padding))
            else:
                blocks.append(PASPPBlock(channel_size, padding))
            
            self.encoder.append(nn.Sequential(*blocks))
            if i < depth - 1:
                # downsampling layer 
                self.encoder.append(ConvBlock(channel_size, 2**(wf+i+1), 
                                              kernel_size=1, stride=2))                

        self.decoder = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.decoder.append(UpBlock(channel_size, 2**(wf+i), padding))
            channel_size = 2**(wf+i)

        self.last = ConvBlock(channel_size, n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for i, encode in enumerate(self.encoder):
            x = encode(x)
            if i % 2 == 0 and i != len(self.encoder)-1:
                skip_connections.append(x)

        for i, decode in enumerate(self.decoder):
            x = decode(x, skip_connections[-i-1])

        x = self.last(x)
        return F.softmax(x, dim=1)


class ConvBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size, 
                 stride=1, padding=False, dilation=1):
        super().__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size=kernel_size, 
                              stride=stride, padding=int(padding), dilation=dilation)
        self.bn = nn.BatchNorm3d(out_size)

    def forward(self, input):
        x = self.bn(self.conv(input))
        return F.relu(x)
        
            
class ResBlock(nn.Module):

    def __init__(self, in_size, out_size, padding):
        super().__init__()

        self.res_block = nn.Sequential(
            ConvBlock(in_size, out_size, kernel_size=3, padding=int(padding)),
            nn.Conv3d(out_size, out_size, kernel_size=3, padding=int(padding)),
            nn.BatchNorm3d(out_size)
        )
        self.skip_layers = nn.ModuleList()
        if in_size != out_size:
            self.skip_layers.append(nn.Conv3d(in_size, out_size, kernel_size=1))
            self.skip_layers.append(nn.BatchNorm3d(out_size))

    def forward(self, input):
        x = self.res_block(input)
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

    def __init__(self, in_size, padding):
        super().__init__()
        self.conv1 = ConvBlock(in_size, in_size, kernel_size=1)
        self.con_branch = ContrastEnhancementBranch(in_size)
        self.pos_branch = PositionSensitiveBranch(in_size, padding=padding)
        self.conv3 = ConvBlock(in_size*3, in_size, kernel_size=3,
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
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        fg = x.repeat(1, self.in_size)[:,:,None,None,None] # expansion
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
                                  padding=2, dilation=2)
        self.atrous_3 = ConvBlock(out_size, out_size, kernel_size=3, 
                                  padding=4, dilation=4)
        self.atrous_4 = ConvBlock(out_size, out_size, kernel_size=3, 
                                  padding=8, dilation=8)

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



