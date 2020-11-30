import torch
import torch.nn.functional as F

def dice_coefficient(y_pred, y_true, epsilon=1, ndims=4):
    dim = [-i for i in reversed(range(1, ndims+1))]
    dice_correct = (y_pred * y_true).sum(dim=dim)
    dice_pred = y_pred.sum(dim=dim)
    dice_true = y_true.sum(dim=dim)

    dice_coef = (2 * dice_correct + epsilon) \
        / (dice_pred + dice_true + epsilon)
    return dice_coef

def dice_loss(y_pred, y_true, ndims=4):
    return 1 - dice_coefficient(y_pred, y_true, ndims=ndims)


def tversky_index(y_pred, y_true, alpha=.6, epsilon=1, ndims=3):
    dim = [-i for i in reversed(range(1, ndims+1))]
    tp = (y_pred * y_true).sum(dim=dim)
    fn = ((1-y_pred) * y_true).sum(dim=dim)
    fp = (y_pred * (1-y_true)).sum(dim=dim)
    beta = 1. - alpha
    ti = (tp + epsilon) / (tp + (alpha*fn) + (beta*fp) + epsilon)
    return ti

def tversky_loss(y_pred, y_true, ndims=4):
    return 1. - tversky_index(y_pred, y_true, ndims=ndims)


def cross_entropy_loss(y_pred, y_true, ndims=3):
    dim = [-i for i in reversed(range(1, ndims+1))]
    return F.cross_entropy(y_pred, y_true, reduction='none').mean(dim=dim)

