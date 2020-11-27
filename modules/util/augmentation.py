import math
import random
import torch
from torch import nn
import torch.nn.functional as F

class SegmentationAugmentation(nn.Module):

    def __init__(self, flip=None, offset=None, scale=None, 
                 rotate=None, noise=None):
        super().__init__()

        self.flip, self.offset, self.scale = flip, offset, scale
        self.rotate, self.noise = rotate, noise

    def forward(self, input_g, label_g):
        input_g = input_g.transpose(-3,-1)
        label_g = label_g.transpose(-3,-1)
        transform_t = self.build_3d_transform()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:,:3], 
                                 input_g.size(), align_corners=False)
        augmented_input_g = F.grid_sample(input_g, affine_t, 
                                          padding_mode='border', 
                                          align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32), 
                                          affine_t, 
                                          padding_mode='border', 
                                          align_corners=False)
        if self.noise and torch.rand(1) < .15:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise
            augmented_input_g += noise_t
        augmented_input_g = augmented_input_g.transpose(-3,-1)
        augmented_label_g = augmented_label_g.transpose(-3,-1)

        return augmented_input_g, augmented_label_g > .5

    def build_3d_transform(self):

        transform_t = torch.eye(4)

        for i in range(3):
            if self.flip:
                if random.random() > .5:
                    transform_t[i,i] *= -1

            if self.offset:
                random_f = random.random() * 2 - 1 # range -1 to 1
                transform_t[i, 3] = self.offset * random_f

            if self.scale:
                random_f = random.random() * 2 - 1 # range -1 to 1
                transform_t[i,i] *= 1. + self.scale * random_f

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])

            transform_t @= rotation_t

        return transform_t


        

