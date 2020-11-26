import torch
import pickle
import numpy as np

from collections import namedtuple

WindowTuple = namedtuple('WindowInfo', ['width', 'level'])

lung_window = WindowTuple(1500, -600)
shifted_lung_window = WindowTuple(1500, -250)
mediastinal_window = WindowTuple(350, 50)

def window_image(img, window_name):
    if window_name is not None:
        if window_name == 'lung':
            window = lung_window
        elif window_name == 'mediastinal':
            window = mediastinal_window
        elif window_name == 'shifted_lung':
            window = shifted_lung_window
        img_min = window.level - window.width//2
        img_max = window.level + window.width//2
    else:
        img_min = -1000
        img_max = 1000
    if type(img).__name__=='Tensor':
        return img.clamp(img_min, img_max)
    return img.clip(img_min, img_max)


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

def find_borders(mask_list, thresh):
    left, right = 0, len(mask_list)-1
    while left != right and (mask_list[left] < thresh or mask_list[right] < thresh):
        if mask_list[left] < thresh:
            left += 1
        if left == right:
            break
        if mask_list[right] < thresh:
            right -= 1
        if left == right:
            break
    return left, right


