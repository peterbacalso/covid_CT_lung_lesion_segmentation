# Covid 19 CT Lung Lesion Segmentation

[Challenge Website](https://covid-segmentation.grand-challenge.org/COVID-19-20/)

# Overview

1. [Requirements](#requirements)

2. [Dependencies and installation](#dependencies-and-installation)

3. [Usages](#usages)

# Requirements

Tested with:

- `Ubuntu 18.04/20.04` | `Python 3.8-dev` | `CUDA 10.1`

GPU memory requirements:

- Default training pipeline requires _ GB memory

- Default inference pipeline requires _ GB memory

# Dependencies and Installation

**Pytorch**

Follow [Pytroch instructions](https://pytorch.org/get-started/locally/) for installation. Run the following code to verify:

`python -c 'import torch; print(torch.rand(4, 2, device="cuda"))'`

**Monai** - more info at https://docs.monai.io/en/latest/installation.html

`pip install monai`

**Nibabel** - more info at https://nipy.org/nibabel/

`pip install nibabel`

**DiskCache** - more info at https://pypi.org/project/diskcache/

`pip install diskcache`

**Learning Rate Finder** - more info at https://github.com/davidtvs/pytorch-lr-finder

`pip install torch-lr-finder`

**Fast Progress** - more info at https://github.com/fastai/fastprogress

`pip install fastprogress`

# Usages

Clone this repo and follow instructions in `run_net_clean.ipynb`. This requires installation of [jupter notebook or jupyter lab](https://jupyter.org/install). Check `run_net.ipynb` to see what the output should look like.

## Training (and validation at every 5th epoch)

Generate a metadata file and prepare a cache of preprocessed data. This will create a metadata/df_meta.fth file as well as a cache/ folder in the current directory.

`python prepcache.py --data-path "COVID-19-20_v2/Train"`

Use lr finder to find a good learning rate. This will run the model through a hundered batches which take a few minutes to complete on the gpu.

`python modules.util.lr_finder.py` 

Begin training. Validation will run every 5 epochs and a copy of the model parameters will be saved under saved-models/ folder. The best model based on F1 score is saved with the prefix `.best.state`

`python training.py --lr 0.001`

## Inference
Prepare the metadata and cache files. Note that this will overwrite the metadatafile generated from the train set if it was created previously

`python prepcache.py --data-path "COVID-19-20_TestSet"`

You can specify a UID from a CT scan based on the number found in the filename.
For example, pass in *0180_0* for *volume-covid19-A-0180_0_ct.nii.gz*

`python inference.py '0180_0' --data-path "COVID-19-20_v2/Train"`

To run inference on all of the files, simply pass the --run-all flag as shown below

`python inference.py --data-path "COVID-19-20_v2/Train" --run-all`





