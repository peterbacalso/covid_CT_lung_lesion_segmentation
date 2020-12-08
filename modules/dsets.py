import os
import math
import glob
import torch
import random
import feather
import functools
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F

from pathlib import Path
from scipy import ndimage
from dotenv import load_dotenv
from diskcache import FanoutCache
from torch.utils.data import Dataset
from skimage.measure import regionprops

# local imports
from modules.util.logconf import logging
from modules.util.util import window_image 
from modules.util.util import find_borders
from modules.util.gaussian_smoothing import GaussianSmoothing

# Load environment variables to get local datasets path
load_dotenv()
data_dir = os.environ.get('datasets_path')
local_dataset_path = Path(f'{data_dir}/COVID-19-20_v2')

raw_cache = FanoutCache('cache/raw', shards=64, 
                        timeout=1, size_limit=3e11)

# set logging level
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Ct:

    def __init__(self, uid, dataset_path=None):
        self.uid = uid

        if dataset_path is None:
            ct_paths = sorted(glob.glob(f'{str(local_dataset_path)}/*/*-0{uid}_*.nii.gz'))
        else:
            ct_paths = sorted(glob.glob(f'{str(dataset_path)}/*-0{uid}_*.nii.gz'))
        assert len(ct_paths) > 0, repr(f'No CT found for given uid {uid}')
        assert 'ct' in ct_paths[0]

        ct = nib.load(ct_paths[0])
        self.ct_t = torch.from_numpy(ct.get_fdata().T).float()
        self.affine = ct.affine
        self.spacing_t = torch.tensor(ct.header.get_zooms()[::-1])

        if len(ct_paths) > 1:
            assert len(ct_paths) <= 2, repr([uid, ct_paths])
            assert 'seg' in ct_paths[1]

            seg = nib.load(ct_paths[1])
            self.mask_t = torch.from_numpy(seg.get_fdata().T).float()
        else:
            self.mask_t = torch.zeros_like(self.ct_t)

    '''
    def get_spaced_ct(self, width_irc, mode='bilinear', from_spacing=None, 
                      size_like=None, to_spacing=None):
        #ct and mask must be NxCxDxHxW, from_spacing and to_spacing must be DxHxW
        ct_t = torch.from_numpy(self.ct_a)[None,None].float()
        mask = None if self.mask is None else torch.from_numpy(self.mask)[None][None].float()

        if from_spacing is None:
            mm_dims = [int(math.ceil(ct_t.shape[-3:][i]*self.spacing[i])) for i in range(3)]
        else:
            mm_dims = [int(math.ceil(ct_t.shape[-3:][i]*from_spacing[i])) for i in range(3)]

        if size_like is None:
            assert to_spacing is not None
            new_size = [int(math.ceil(mm_dims[i]/to_spacing[i])) for i in range(3)]
        else:
            new_size = size_like.size()[-3:]

        d = torch.linspace(-1, 1, new_size[0])
        h = torch.linspace(-1, 1, new_size[1])
        w = torch.linspace(-1, 1, new_size[2])
        meshz, meshy, meshx = torch.meshgrid((d, h, w))
        grid = torch.stack((meshx, meshy, meshz), 3)
        grid = grid.unsqueeze(0) # add batch dim

        spaced_ct = F.grid_sample(ct_t, grid, mode=mode, 
                                  padding_mode='reflection', align_corners=False)
        # height and width must not be less than 192
        padded_ct = self.pad_to(spaced_ct.squeeze(0), width_irc[-2:])
        ct_t = padded_ct.squeeze()

        mask_t = None 
        if mask is not None:
            spaced_mask = F.grid_sample(mask, grid, mode=mode, 
                                        padding_mode='reflection', 
                                        align_corners=False)
            padded_mask = self.pad_to(spaced_mask.squeeze(0), width_irc[-2:])
            mask_t = padded_mask.squeeze()
        

        return ct_t, mask_t

    def pad_to(self, img, min_size, mode='reflect'):
        h_dim, w_dim = img.size()[-2:]
        h_pad = max(0,(min_size[0]-h_dim)//2)
        w_pad = max(0,(min_size[1]-w_dim)//2)
        
        h_pad_1 = h_pad_2 = h_pad
        w_pad_1 = w_pad_2 = w_pad
        if img.size()[-2:][0] + h_pad*2 < min_size[0]:
            h_pad_1 = h_pad + 1
            h_pad_2 = h_pad
        if img.size()[-2:][1] + w_pad*2 < min_size[1]:
            w_pad_1 = w_pad + 1
            w_pad_2 = w_pad
        padding = (w_pad_1,w_pad_2,h_pad_1,h_pad_2)

        return F.pad(img, padding, mode=mode)
    '''

@functools.lru_cache(1, typed=True)
def get_ct(uid, dataset_path=None):
    return Ct(uid, dataset_path)

'''
@raw_cache.memoize(typed=True)
def get_spaced_ct(uid, width_irc=[7,192,192], to_spacing=[5, 1.25, 1.25]):
    ct = get_ct(uid)
    ct_t, mask_t = ct.get_spaced_ct(width_irc=width_irc,
                                                to_spacing=to_spacing)
    return ct_t, mask_t 

@raw_cache.memoize(typed=True)
def get_ct_index_info(uid, width_irc=[7,192,192], to_spacing=[5,1.25,1.25]):
    ct_t, mask_t = get_spaced_ct(uid, width_irc, to_spacing)
    positive_indices = mask_t.numpy().sum(axis=(-2,-1)).nonzero()[0].tolist()
    return int(ct_t.shape[0]), positive_indices
'''

@raw_cache.memoize(typed=True)
def get_cropped_ct(uid, width_irc):
    ct = get_ct(uid)
    ct_t = ct.ct_t
    mask_t = ct.mask_t
    spacing_t = ct.spacing_t

    gaussian_blur = GaussianSmoothing(channels=ct_t.shape[0], 
                                      kernel_size=3, sigma=3)
    x = ct_t[None]
    x_pad = F.pad(x, (1,1,1,1), mode='reflect')
    x_blur = gaussian_blur(x_pad)
    x_blur.clamp(-1000,1000)
    x_blur /= 2000
    x_blur += .5
    x_mask = (x_blur > .2).float()
    x_mask = torch.tensor(ndimage.binary_erosion(
        x_mask.numpy(), structure=np.ones((1,5,5,5)))).float()
    thresh_1 = int((x_mask.size()[-2] * x_mask.size()[-1]) * .1)
    thresh_2 = int((x_mask.size()[-3] * x_mask.size()[-1]) * .1)
    thresh_3 = int((x_mask.size()[-3] * x_mask.size()[-2]) * .1)
    lo_i, hi_i = find_borders(torch.squeeze(x_mask.sum(axis=(-2,-1))), thresh=thresh_1)
    lo_r, hi_r = find_borders(torch.squeeze(x_mask.sum(axis=(-3,-1))), thresh=thresh_2)
    lo_c, hi_c = find_borders(torch.squeeze(x_mask.sum(axis=(-3,-2))), thresh=thresh_3)

    if hi_i - lo_i < width_irc[0]:
        lo_i, hi_i = width_correction(hi_i, lo_i, width_irc[0], ct_t.shape[0]-1)
    if hi_r - lo_r < width_irc[1]:
        lo_r, hi_r = width_correction(hi_r, lo_r, width_irc[1], ct_t.shape[1]-1)
    if hi_c - lo_c < width_irc[2]:
        lo_c, hi_c = width_correction(hi_c, lo_c, width_irc[2], ct_t.shape[2]-1)

    ct_clip = ct_t[lo_i:hi_i,lo_r:hi_r,lo_c:hi_c]
    mask_clip = mask_t[lo_i:hi_i,lo_r:hi_r,lo_c:hi_c]

    return ct_clip, mask_clip, spacing_t

def width_correction(hi, lo, width, max_index):
    pad_size = width - (hi - lo)
    head_pad = pad_size // 2
    tail_pad = pad_size // 2 if pad_size % 2 == 0 else pad_size // 2 + 1
    if lo - tail_pad < 0:
        head_pad += abs(lo-tail_pad)
    if hi + head_pad > max_index:
        tail_pad += ((hi + head_pad) - max_index)
    lo = max(0, lo-tail_pad)
    hi = min(max_index, hi+head_pad)
    return lo, hi


def get_ct_augmented(augmentation_dict, uid, width_irc, use_cache=True):
    if use_cache:
        ct_t, mask_t, spacing_t = get_cropped_ct(uid, width_irc)
    else:
        ct = get_ct(uid)
        ct_t = ct.ct_t
        mask_t = ct.mask_t
        spacing_t = ct.spacing_t
    '''
    if use_cache:
        #ct_t, mask_t = get_spaced_ct(uid)
    else:
        ct = get_ct(uid)
        ct_t, mask_t = ct.get_spaced_ct(to_spacing=[5,1.25,1.25])
    '''

    ct_t, mask_t, spacing_t = ct_t[None,None], mask_t[None,None], spacing_t[None]

    transform_t = torch.eye(4)
    
    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > .5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset = augmentation_dict['offset']
            random_f = random.random() * 2 - 1 # range -1 to 1
            transform_t[i, 3] = offset * random_f

        if 'scale' in augmentation_dict:
            scale = augmentation_dict['scale']
            random_f = random.random() * 2 - 1 # range -1 to 1
            transform_t[i,i] *= 1. + scale * random_f

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0 ,0, 1]
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(
        transform_t[:3][None,:].to(torch.float32),
        ct_t.size(),
        align_corners=False)

    augmented_ct = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode='border',
        align_corners=False
    ).to('cpu')

    augmented_mask = F.grid_sample(
        mask_t.to(torch.float32),
        affine_t,
        padding_mode='border',
        align_corners=False
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_ct)
        noise_t *= augmentation_dict['noise'] 
        augmented_ct += noise_t

    augmented_ct = augmented_ct.squeeze()
    augmented_mask = augmented_mask.squeeze() > .5

    return augmented_ct, augmented_mask, spacing_t

def group_lesions(ct_t, mask_t, num_erosions=0):
    ct_a = ct_t.numpy()
    mask_a = mask_t.numpy()
    # get rid of tiny lesions by using morphology erosion
    if num_erosions > 0:
        clean_mask = ndimage.binary_erosion(mask_a, iterations=num_erosions)
    else:
        clean_mask = mask_a
    # group blobs that are together
    lesion_labels, lesion_count = ndimage.label(clean_mask) 

    center_irc_list = ndimage.center_of_mass(
        ct_a.clip(-1000,1000) + 1001, # function needs +'ve input
        labels = lesion_labels,
        index=np.arange(1,lesion_count+1))

    lesions = []

    # center_of_mass will produce floating point values so we round
    for center_irc in center_irc_list:
        idx = max(1, int(round(center_irc[0])))
        idx = min(mask_t.shape[-3], idx)
        row = max(1, int(round(center_irc[1])))
        row = min(mask_t.shape[-2], row)
        col = max(1, int(round(center_irc[2])))
        col = min(mask_t.shape[-1], col)
        lesions.append((idx, row, col))

    random.shuffle(lesions)
    return lesions[:3]

def get_random_center(mask_t, label_value):
    if label_value not in mask_t:
        label_value = 1 - label_value
    while True:
        i = np.random.choice(mask_t.shape[-3]-1,1)[0] + 1
        r = np.random.choice(mask_t.shape[-2]-1,1)[0] + 1
        c = np.random.choice(mask_t.shape[-1]-1,1)[0] + 1
        if mask_t[i][r][c] == label_value:
            return (i,r,c)

def get_chunk(ct_t, mask_t, center_irc, width_irc):

    slice_list = []
    for axis, center_val in enumerate(center_irc):
        start_idx = int(round(center_val - width_irc[axis]/2))
        end_idx = int(round(start_idx + width_irc[axis]))

        assert width_irc[axis] <= ct_t.shape[axis], repr(f'width at axis {axis} is larger than the ct')
        assert center_val > 0 and center_val < ct_t.shape[axis], \
            repr([center_irc, axis, ct_t.shape, width_irc])

        # shift if coord is at the border of the axis
        if start_idx < 0:
            start_idx = 0
            end_idx = width_irc[axis]

        if end_idx > ct_t.shape[axis]:
            end_idx = ct_t.shape[axis]
            start_idx = ct_t.shape[axis] - width_irc[axis]

        slice_list.append(slice(start_idx, end_idx))

    ct_chunk = ct_t[tuple(slice_list)]
    mask_chunk = mask_t[tuple(slice_list)]

    return ct_chunk, mask_chunk

@functools.lru_cache(1)
def get_meta_dict():
    df_meta = pd.read_feather('metadata/df_meta.fth')
    df_meta_trn = df_meta[df_meta.is_valid==False]
    meta_dict = {}

    for _, row in df_meta_trn.iterrows():
        meta_dict.setdefault(row.uid, []).append(row)

    return meta_dict 

def collate_fn(batch):
    imgs,targets,spacings = zip(*batch)
    return torch.cat(imgs),torch.cat(targets),torch.cat(spacings)

class Covid2dSegmentationDataset(Dataset):

    def __init__(self, uid=None, is_valid=None, splitter=None, 
                 window=None, steps_per_epoch=None):

        if uid:
            self.uid_list = [uid]
        else:
            self.uid_list = sorted(get_meta_dict().keys())

        if is_valid:
            assert splitter
            _, self.uid_list = splitter(self.uid_list)
        elif splitter:
            self.uid_list, _ = splitter(self.uid_list)

        self.window = window
        self.steps_per_epoch = len(self.uid_list) \
            if steps_per_epoch is None else steps_per_epoch

        log.info(f"{type(self).__name__}: " \
                 + "{} mode, ".format({None:'general',True:'validation',False:'training'}[is_valid]) \
                 + f"{len(self.uid_list)} uid's, "
                 + f"{self.steps_per_epoch} steps_per_epoch")

    def __len__(self):
        return self.steps_per_epoch

    def shuffle(self):
        random.shuffle(self.uid_list)

    def __getitem__(self, idx):
        uid = self.uid_list[idx % len(self.uid_list)]
        #ct_t, mask_t = get_spaced_ct(uid, self.width_irc)
        ct = get_ct(uid)
        ct_t = window_image(ct.ct_t, self.window)
        return ct_t[None], ct.mask_t[None], ct.spacing_t[None]

class TrainingCovid2dSegmentationDataset(Covid2dSegmentationDataset):

    def __init__(self, width_irc=(7,192,192), augmentation_dict={}, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.width_irc = width_irc 
        self.augmentation_dict = augmentation_dict
        log.info(f"{type(self).__name__}: {self.width_irc} width_irc")

    def __len__(self):
        return self.steps_per_epoch

    def shuffle(self):
        random.shuffle(self.uid_list)

    def __getitem__(self, idx):
        uid = self.uid_list[idx % len(self.uid_list)]
        return self.getitem_cropbox(uid)

    def getitem_cropbox(self, uid):
        aug_ct_t, aug_mask_t, spacing_t = get_ct_augmented(
            self.augmentation_dict, uid, self.width_irc, use_cache=True)

        num_erosions = np.random.choice(2,1)[0]

        center_irc_list = group_lesions(aug_ct_t, aug_mask_t, num_erosions)
        #center_irc_list = []
        while len(center_irc_list) < 3:
            center_irc = get_random_center(aug_mask_t, 1)
            center_irc_list.append(center_irc)

        ct_chunks = []
        mask_chunks = []
        spacings = []
        for center_irc in center_irc_list:
            ct_chunk, mask_chunk = get_chunk(aug_ct_t, 
                                             aug_mask_t, 
                                             center_irc,
                                             self.width_irc)
            ct_chunk = window_image(ct_chunk, self.window)
            ct_chunks.append(ct_chunk[None])
            mask_chunks.append(mask_chunk[None])
            spacings.append(spacing_t[None])
        
        return torch.stack(ct_chunks), torch.stack(mask_chunks), torch.stack(spacings)

class TrainingV2Covid2dSegmentationDataset(Covid2dSegmentationDataset):

    def __init__(self, width_irc=(7,192,192), augmentation_dict={}, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.width_irc = width_irc 
        self.augmentation_dict = augmentation_dict
        log.info(f"{type(self).__name__}: {self.width_irc} width_irc")

    def __len__(self):
        return self.steps_per_epoch

    def shuffle(self):
        random.shuffle(self.uid_list)

    def __getitem__(self, idx):
        uid = self.uid_list[idx % len(self.uid_list)]
        return self.getitem_cropbox(uid)

    def getitem_cropbox(self, uid):
        aug_ct_t, aug_mask_t, spacing_t = get_ct_augmented(
            self.augmentation_dict, uid, self.width_irc, use_cache=True)

        center_irc_list = []
        while len(center_irc_list) < 3:
            label_value = np.random.choice(2,1).item()
            center_irc = get_random_center(aug_mask_t, label_value)
            center_irc_list.append(center_irc)

        ct_chunks = []
        mask_chunks = []
        spacings = []
        for center_irc in center_irc_list:
            ct_chunk, mask_chunk = get_chunk(aug_ct_t, 
                                             aug_mask_t, 
                                             center_irc,
                                             self.width_irc)
            ct_chunk = window_image(ct_chunk, self.window)
            ct_chunks.append(ct_chunk[None])
            mask_chunks.append(mask_chunk[None])
            spacings.append(spacing_t[None])
        
        return torch.stack(ct_chunks), torch.stack(mask_chunks), torch.stack(spacings)

class PrepcacheCovidDataset(Dataset):
    def __init__(self, width_irc, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.width_irc = width_irc
        df_meta = pd.read_feather('metadata/df_meta.fth')
        self.df_meta = df_meta[df_meta.is_valid==False].sort_values(by='uid')

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, idx):

        uid = self.df_meta.uid.iloc[idx]
        get_ct(uid)
        get_cropped_ct(uid, self.width_irc)
        '''
        get_spaced_ct(uid, width_irc=self.width_irc, 
                      to_spacing=[5, 1.25, 1.25])
        get_ct_index_info(uid, width_irc=self.width_irc, 
                          to_spacing=[5,1.25,1.25])
        '''

        return 0, 1 
