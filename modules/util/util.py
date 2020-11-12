import numpy as np

from collections import namedtuple

from modules.util.logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

IRCTuple = namedtuple('IRCTuple', ['index', 'row', 'col'])
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
