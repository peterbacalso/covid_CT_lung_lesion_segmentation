import os
import glob
import feather
import functools
import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from scipy import ndimage
from diskcache import FanoutCache

# Load environment variables to get local datasets path
from dotenv import load_dotenv
load_dotenv()
data_dir = os.environ.get('datasets_path')
dataset_path = Path(f'{data_dir}/COVID-19-20_v2')

raw_cache = FanoutCache('data/cache/raw', shards=64, 
                        timeout=1, size_limit=3e11)

class Ct:

    def __init__(self, uid):
        self.uid = uid

        ct_paths = sorted(glob.glob(f'{str(dataset_path)}/*/*-0{uid}_*.nii.gz'))
        assert len(ct_paths) > 0, repr(f'No CT found for given uid {uid}')
        assert 'ct' in ct_paths[0]

        ct = nib.load(ct_paths[0])
        self.hu = ct.get_fdata().T
        self.affine = ct.affine

        self.mask = None
        self.lesions = None
        if len(ct_paths) > 1:
            assert len(ct_paths) <= 2, repr([uid, ct_paths])
            assert 'seg' in ct_paths[1]

            seg = nib.load(ct_paths[1])
            self.mask = seg.get_fdata().T
            lesions_dict = get_lesions_dict()
            if uid in lesions_dict:
                self.lesions = lesions_dict[uid]


    def group_lesions(self, output_df=True):
        # get rid of tiny lesions by using morphology erosion
        clean_mask = ndimage.binary_erosion(self.mask)
        # group blobs that are together
        lesion_label_a, lesion_count = ndimage.label(clean_mask) 
        center_irc_list = ndimage.center_of_mass(
            self.hu.clip(-1000,1000) + 1001, # function needs +'ve input
            labels = lesion_label_a,
            index=np.arange(1,lesion_count+1))
        lesion_cols = ['uid', 'coordI', 'coordR', 'coordC']
        lesions = []
        # center_of_mass will produce floating point values so we round
        for i, center_irc in enumerate(center_irc_list):
            lesions.append([self.uid, 
                            int(round(center_irc[0])), 
                            int(round(center_irc[1])), 
                            int(round(center_irc[2]))])

        return lesions if not output_df \
            else pd.DataFrame(lesions, columns=lesion_cols)

    def get_raw_lesion(self, center_irc, width_irc):
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_idx = int(round(center_val - width_irc[axis]/2))
            end_idx = int(round(start_idx + width_irc[axis]))

            assert width_irc[axis] <= self.hu.shape[axis], repr(f'width at axis {axis} is larger than the ct')
            assert center_val > 0 and center_val < self.hu.shape[axis], \
                repr([self.uid, center_irc, axis, width_irc])

            # shift if lesion is at the border of the axis
            if start_idx < 0:
                start_idx = 0
                end_idx = width_irc[axis]

            if end_idx > self.hu.shape[axis]:
                end_idx = self.hu.shape[axis]
                start_idx = self.hu.shape[axis] - width_irc[axis]

            slice_list.append(slice(start_idx, end_idx))

        print(slice_list)

        hu_chunk = self.hu[tuple(slice_list)]
        pos_chunk = self.mask[tuple(slice_list)]

        return hu_chunk, pos_chunk, center_irc


@functools.lru_cache(1)
def get_lesions_dict():
    df_lesions = pd.read_feather('../df_lesion_coords.fth')
    lesions_dict = {}

    for _, lesion in df_lesions.iterrows():
        lesions_dict.setdefault(lesion.uid, []).append(lesion)

    return lesions_dict 

@functools.lru_cache(1, typed=True)
def get_ct(uid):
    return Ct(uid)

@raw_cache.memoize(typed=True)
def get_raw_lesion(uid, center_irc, width_irc): 
    ct = get_ct(uid)
    hu_chunk, pos_chunk, center_irc = ct.get_raw_lesion(center_irc, width_irc)
    '''
    if window is None:
        hu_chunk.clip(-1000,1000,hu_chunk)
    else:
        hu_min = window.level - window.width//2
        hu_max = window.center + window.width//2
        hu_chunk.clip(hu_min,hu_max,hu_chunk)
    '''
    return hu_chunk, pos_chunk, center_irc





