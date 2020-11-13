from collections import namedtuple

WindowTuple = namedtuple('WindowInfo', ['width', 'level'])

def window_image(img, window, img_type='tensor'):
    if window is not None:
        img_min = window.level - window.width//2
        img_max = window.level + window.width//2
    else:
        img_min = -1000
        img_max = 1000
    if img_type=='tensor':
        return img.clamp(img_min, img_max)
    return img.clip(img_min, img_max)

'''
import numpy as np

XYZTuple = namedtuple('XYZTuple', ['x', 'y', 'z'])

def irc2xyz(irc, affine):
    linear_transform = affine[:3, :3]
    offset = affine[:3, 3]
    xyz = linear_transform.dot(irc) + offset
    return XYZTuple(*xyz)


def xyz2irc(xyz, affine):
    affine_inverse = np.linalg.inv(affine)
    linear_transform = affine_inverse[:3, :3]
    offset = affine_inverse[:3, 3]
    irc = linear_transform.dot(xyz) + offset
    return IRCTuple(*irc)
'''

