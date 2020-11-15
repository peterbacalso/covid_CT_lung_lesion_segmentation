import torch
import numpy as np

from collections import namedtuple

WindowTuple = namedtuple('WindowInfo', ['width', 'level'])

lung_window = WindowTuple(1500, -600)
mediastinal_window = WindowTuple(350, 50)

def window_image(img, window_name):
    if window_name is not None:
        if window_name == 'lung':
            window = lung_window
        elif window_name == 'mediastinal':
            window = mediastinal_window
        img_min = window.level - window.width//2
        img_max = window.level + window.width//2
    else:
        img_min = -1000
        img_max = 1000
    if type(img).__name__=='Tensor':
        return img.clamp(img_min, img_max)
    return img.clip(img_min, img_max)

def normalize_image(image, window_name):
    if window_name is None:
        bins_path = 'bins.pkl'
    elif window_name == 'lung':
        bins_path = 'bins_lung_window.pkl'

    with open(bins_path, 'rb') as handle:
        bins = pickle.load(handle)

    return hist_scaled(img, bins)


def list_stride_splitter(array, val_stride):
    assert val_stride > 0, val_stride
    return [item for i, item in enumerate(array) if i % val_stride > 0], array[::val_stride]

def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module

# helper functions below are slightly modified code snippets from
# https://github.com/fastai/fastai2/blob/master/fastai2/medical/imaging.py

def array_freqhist_bins(img, n_bins=100):
    "A numpy based function to split the range of pixel values into groups, such that each group has around the same number of pixels"
    imsd = np.sort(img.flatten())
    t = np.array([0.001])
    t = np.append(t, np.arange(n_bins)/n_bins+(1/2/n_bins))
    t = np.append(t, 0.999)
    t = (len(imsd)*t+0.5).astype(np.int)
    return np.unique(imsd[t])

def hist_scaled(img,  brks=None):
    if brks is None: brks = array_freqhist_bins(img)
    ys = np.linspace(0., 1., len(brks))
    x = img.flatten()
    x = np.interp(x, brks, ys)
    return torch.tensor(x).reshape(img.shape).clamp(0.,1.)


