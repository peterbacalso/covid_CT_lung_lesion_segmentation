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
from pathlib import Path
from scipy import ndimage
from dotenv import load_dotenv
from diskcache import FanoutCache
from torch.utils.data import Dataset
from skimage.measure import regionprops

# local imports
from modules.util.util import window_image 
from modules.util.logconf import logging

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
        self.ct_a = ct.get_fdata().T
        self.affine = ct.affine

        self.mask = None
        self.coords = None
        self.positive_indices = None
        if len(ct_paths) > 1:
            assert len(ct_paths) <= 2, repr([uid, ct_paths])
            assert 'seg' in ct_paths[1]

            seg = nib.load(ct_paths[1])
            self.mask = seg.get_fdata().T
            coords_dict = get_coords_dict()
            if uid in coords_dict:
                self.coords = coords_dict[uid]
            self.positive_indices = self.mask.sum(axis=(1,2))\
                                    .nonzero()[0].tolist()


    def group_lesions(self, output_df=True, num_erosions=0):
        # get rid of tiny lesions by using morphology erosion
        if num_erosions > 0:
            clean_mask = ndimage.binary_erosion(self.mask, iterations=num_erosions)
        else:
            clean_mask = self.mask
        #clean_mask = self.mask
        # group blobs that are together
        lesion_labels, lesion_count = ndimage.label(clean_mask) 
        properties = regionprops(lesion_labels)

        center_irc_list = ndimage.center_of_mass(
            self.ct_a.clip(-1000,1000) + 1001, # function needs +'ve input
            labels = lesion_labels,
            index=np.arange(1,lesion_count+1))
        lesion_cols = ['uid', 'coordI', 'coordR', 'coordC', 
                       'min_index', 'max_index', 'min_row', 'max_row', 
                       'min_column', 'max_column', 
                       'index_width', 'row_width', 'column_width',
                       'largest_side_px']
        lesions = []

        assert len(properties) == len(center_irc_list), repr([len(properties), len(center_irc_list)])
        zip_iter = zip(center_irc_list, properties)
        # center_of_mass will produce floating point values so we round
        for i, (center_irc, p) in enumerate(zip_iter):
            min_i, min_r, min_c, max_i, max_r, max_c = p.bbox
            largest_side = max(max_i-min_i, max_r-min_r, max_c-min_c)
            lesions.append([self.uid, 
                            int(round(center_irc[0])), 
                            int(round(center_irc[1])), 
                            int(round(center_irc[2])),
                            min_i, max_i, min_r,
                            max_r, min_c, max_c,
                            max_i-min_i, max_r-min_r, max_c-min_c,
                            largest_side])

        return lesions if not output_df \
            else pd.DataFrame(lesions, columns=lesion_cols)

    def get_raw_coord(self, center_irc, width_irc):
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_idx = int(round(center_val - width_irc[axis]/2))
            end_idx = int(round(start_idx + width_irc[axis]))

            assert width_irc[axis] <= self.ct_a.shape[axis], repr(f'width at axis {axis} is larger than the ct')
            assert center_val > 0 and center_val < self.ct_a.shape[axis], \
                repr([self.uid, center_irc, axis, width_irc])

            # shift if coord is at the border of the axis
            if start_idx < 0:
                start_idx = 0
                end_idx = width_irc[axis]

            if end_idx > self.ct_a.shape[axis]:
                end_idx = self.ct_a.shape[axis]
                start_idx = self.ct_a.shape[axis] - width_irc[axis]

            slice_list.append(slice(start_idx, end_idx))

        ct_chunk = self.ct_a[tuple(slice_list)]
        pos_chunk = self.mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1)
def get_coords_dict():
    coords = pd.read_feather('metadata/df_rand_coords.fth')
    coords_dict = {}

    for _, coord in coords.iterrows():
        coords_dict.setdefault(coord.uid, []).append(coord)

    return coords_dict 

@functools.lru_cache(1, typed=True)
def get_ct(uid, dataset_path=None):
    return Ct(uid, dataset_path)

@raw_cache.memoize(typed=True)
def get_raw_coord(uid, center_irc, width_irc): 
    ct = get_ct(uid)
    ct_chunk, pos_chunk, center_irc = ct.get_raw_coord(center_irc, width_irc)
    return ct_chunk, pos_chunk, center_irc

@raw_cache.memoize(typed=True)
def get_ct_index_info(uid):
    ct = get_ct(uid)
    return int(ct.ct_a.shape[0]), ct.positive_indices

class Covid2dSegmentationDataset(Dataset):

    def __init__(self, uid=None, is_valid=None, splitter=None, shuffle=True,
                 width_irc=(15,192,192), is_full_ct=False, window=None):

        if uid:
            self.uid_list = [uid]
        else:
            self.uid_list = sorted(get_coords_dict().keys())

        if shuffle:
            random.shuffle(self.uid_list)

        if is_valid:
            assert splitter
            _, self.uid_list = splitter(self.uid_list)
        elif splitter:
            self.uid_list, _ = splitter(self.uid_list)

        self.window = window

        self.index_slices = []
        for uid in self.uid_list:
            index_count, positive_indices = get_ct_index_info(uid)

            if is_full_ct: # get entire ct
                self.index_slices += [(uid, slice_idx) 
                                      for slice_idx in range(index_count)]
            else: # get only slices with a lesion present
                self.index_slices += [(uid, slice_idx) 
                                      for slice_idx in positive_indices]

        self.width_irc = width_irc # only be using 66% of this so make the width_irc 1.5 times larger than intended
        self.context_slice_count = self.width_irc[0] // 2

        uid_set = set(self.uid_list)
        self.coords = pd.read_feather('metadata/df_rand_coords.fth')
        self.coords.sort_values(by='uid',inplace=True)
        self.coords = self.coords[self.coords.uid.isin(uid_set)]

        log.info(f"{type(self).__name__}: " \
                 + "{} mode, ".format({None:'general',True:'validation',False:'training'}[is_valid]) \
                 + f"{len(self.uid_list)} uid's, " \
                 + f"{len(self.index_slices)} index slices, " \
                 + f"{len(self.coords)} total coords")

    def __len__(self):
        return len(self.index_slices)

    def shuffle(self):
        random.shuffle(self.index_slices)

    def __getitem__(self, idx):

        uid, slice_idx = self.index_slices[idx % len(self.index_slices)]
        return self.getitem_fullslice(uid, slice_idx)

    def getitem_fullslice(self, uid, slice_idx):
        ct = get_ct(uid)
        ct_slice = torch.zeros((self.context_slice_count*2+1, 512, 512))

        start_idx = slice_idx - self.context_slice_count
        end_idx = slice_idx + self.context_slice_count + 1
        for i, context_idx in enumerate(range(start_idx, end_idx)):
            context_idx = max(context_idx,0)
            context_idx = min(context_idx, ct.ct_a.shape[0]-1)
            ct_slice[i] = torch.from_numpy(ct.ct_a[context_idx].astype(np.float32))
        ct_slice = window_image(ct_slice, self.window)

        if ct.mask is not None:
            mask_slice = torch.from_numpy(ct.mask[slice_idx]).unsqueeze(0)
        else:
            mask_slice = torch.zeros_like(ct_slice).unsqueeze(0)

        return ct_slice, mask_slice, uid, slice_idx

class TrainingCovid2dSegmentationDataset(Covid2dSegmentationDataset):

    def __init__(self, is_valid=False, steps_per_epoch=10000, *args, **kwargs):
        super().__init__(is_valid=is_valid, *args, **kwargs)

        self.steps_per_epoch = steps_per_epoch
        log.info(f"{type(self).__name__}: {self.width_irc} width_irc, "
                 + f" {self.steps_per_epoch} steps_per_epoch")

    def __len__(self):
        return self.steps_per_epoch

    def shuffle(self):
        self.coords = self.coords.sample(frac=1)

    def __getitem__(self, idx):
        coord = self.coords.iloc[idx % len(self.coords)]
        return self.getitem_cropslice(coord)

    def getitem_cropslice(self, coord):
        ct_chunk, mask_chunk, center_irc = get_raw_coord(
            coord.uid,
            tuple(coord[['coordI', 'coordR', 'coordC']]),
            self.width_irc)

        mask = mask_chunk[self.context_slice_count:self.context_slice_count+1]

        '''
        max_row_offset = int(math.ceil(self.width_irc[1]*.33))
        max_col_offset = int(math.ceil(self.width_irc[2]*.33))
        row_offset = random.randrange(0, max_row_offset)
        col_offset = random.randrange(0, max_col_offset)
        row_width = self.width_irc[1] - max_row_offset
        col_width = self.width_irc[2] - max_row_offset
        '''

        #hu_chunk = torch.from_numpy(hu_chunk[:, row_offset:row_offset+row_width, 
        #        col_offset:col_offset+col_width]).to(torch.float32)
        ct_chunk = torch.from_numpy(ct_chunk).to(torch.float32)
        ct_chunk = window_image(ct_chunk, self.window)

        #mask = torch.from_numpy(mask[:, row_offset:row_offset+row_width, 
        #        col_offset:col_offset+col_width]).to(torch.float32)
        mask = torch.from_numpy(mask).to(torch.float32)

        return ct_chunk, mask, coord.uid, coord.coordI


class PrepcacheCovidDataset(Dataset):
    def __init__(self, width_irc=(15,192,192), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.width_irc = width_irc
        self.coords = pd.read_feather('metadata/df_rand_coords.fth')

        self.seen_set = set()
        self.coords.sort_values(by='uid', inplace=True)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        coord = self.coords.iloc[idx]
        get_raw_coord(coord.uid, 
                      tuple(coord[['coordI', 'coordR', 'coordC']]),
                      self.width_irc)

        uid = coord.uid
        if uid not in self.seen_set:
            self.seen_set.add(uid)
            get_ct_index_info(uid)

        return 0, 1 
