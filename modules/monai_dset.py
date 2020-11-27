import os
import glob
import monai
import logging
import numpy as np

from pathlib import Path
from dotenv import load_dotenv

from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    CastToTyped,
    LoadNiftid,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

load_dotenv()
data_dir = os.environ.get('datasets_path')
local_dataset_path = Path(f'{data_dir}/COVID-19-20_v2')

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def get_xforms(mode="trn", keys=("image", "label"), 
               spacing=(1.25,1.25,5.0), width_cri=None):

    xforms = [
        LoadNiftid(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="RAS"),
        Spacingd(keys, pixdim=spacing, mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "trn":
        assert width_cri is not None
        xforms.extend([
            SpatialPadd(keys, spatial_size=(width_cri[0], width_cri[1], -1), mode="reflect"),
            RandAffined(
                        keys,
                        prob=0.15,
                        rotate_range=(-0.05, 0.05),
                        scale_range=(-0.1, 0.1),
                        mode=("bilinear", "nearest"),
                        as_tensor_output=False,
                    ),
            RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=width_cri, num_samples=2),
            #RandGaussianNoised(keys[0], prob=0.15, std=0.01),
            #RandFlipd(keys, spatial_axis=0, prob=0.5),
            #RandFlipd(keys, spatial_axis=1, prob=0.5),
            #RandFlipd(keys, spatial_axis=2, prob=0.5),
        ])
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    dtype = (np.float32, np.uint8)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)

def get_ds(data_folder=".", width_cri=(192,192,16), 
           spacing=(1.25,1.25,5.0), splitter=None):
    data_path = Path(data_folder)

    images = sorted(glob.glob(f"{str(data_path)}/*_ct.nii.gz"))
    labels = sorted(glob.glob(f"{str(data_path)}/*_seg.nii.gz"))
    log.info(f"training: image/label ({len(images)}) folder: {data_folder}")

    keys = ("image", "label")
    if splitter is None:
        trn_frac, val_frac = 0.8, 0.2
        n_trn = int(trn_frac * len(images)) + 1
        n_val = len(images) - n_trn
        trn_iter = zip(images[:n_trn], labels[:n_trn])
        val_iter = zip(images[n_trn:], labels[n_trn:])
    else:
        trn_images, val_images = splitter(images)
        trn_labels, val_labels = splitter(labels)
        trn_iter = zip(trn_images, trn_labels)
        val_iter = zip(val_images, val_labels)

    trn_files = [{keys[0]: img, keys[1]: seg} for img, seg in trn_iter]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in val_iter]
    
    log.info(f"training: train {len(trn_files)} val {len(val_files)}, folder: {data_folder}")

    trn_transforms = get_xforms("trn", keys, spacing, width_cri)
    trn_ds = monai.data.CacheDataset(data=trn_files, transform=trn_transforms)

    val_transforms = get_xforms("val", keys, spacing, width_cri)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)

    return trn_ds, val_ds

