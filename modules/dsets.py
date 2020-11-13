import os
import glob
import feather
import functools
import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from scipy import ndimage

# Load environment variables to get local datasets path
from dotenv import load_dotenv
load_dotenv()
data_dir = os.environ.get('datasets_path')
dataset_path = Path(f'{data_dir}/COVID-19-20_v2')

class Ct:

    def __init__(self, uid):
        self.uid = uid

        ct_paths = sorted(glob.glob(f'{str(dataset_path)}/*/*{uid}*.nii.gz'))
        assert len(ct_paths) > 0, repr(f'No CT found for given uid {uid}')
        assert 'ct' in ct_paths[0]

        ct = nib.load(ct_paths[0])
        self.hu = ct.get_fdata()
        self.affine = ct.affine

        self.mask = None
        if len(ct_paths) == 2:
            assert 'seg' in ct_paths[1]

            seg = nib.load(ct_paths[1])
            self.mask = seg.get_fdata()
        self.lesions = get_lesions_dict()[uid]


    def group_lesions(self, output_df=True):
        # group blobs using morphology erosion
        lesion_label_a, lesion_count = ndimage.label(self.mask) 
        center_irc_list = ndimage.center_of_mass(
            self.hu.clip(-1000,1000) + 1001, # function needs +'ve input
            labels = lesion_label_a,
            index=np.arange(1,lesion_count+1))
        lesion_cols = ['uid', 'coordI', 'coordR', 'coordC']
        lesions = []
        # center_of_mass will produce floating point values so we round
        for i, center_irc in enumerate(center_irc_list):
            lesions.append([self.uid, 
                               int(center_irc[0]), 
                               int(center_irc[1]), 
                               int(center_irc[2])])

        return lesions if not output_df \
            else pd.DataFrame(lesions, columns=lesion_cols)


@functools.lru_cache(1)
def get_lesions_dict():
    df_lesions = pd.read_feather('../df_lesion_coords.fth')
    lesions_dict = {}

    for _, lesion in df_lesions.iterrows():
        lesions_dict.setdefault(lesion.uid, []).append(lesion)

    return lesions_dict 




