import torch

def dice_coefficient(y_true, y_pred, epsilon=1, ndims=3):
    dim = [-i for i in reversed(range(1, ndims+1))]
    dice_correct = (y_true * y_pred).sum(dim=dim)
    dice_true = y_true.sum(dim=dim)
    dice_pred = y_pred.sum(dim=dim)

    dice_coef = (2 * dice_correct + epsilon) \
        / (dice_true + dice_pred + epsilon)
    return dice_coef

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def tversky_index(y_true, y_pred, alpha=.6, epsilon=1, ndims=3):
    dim = [-i for i in reversed(range(1, ndims+1))]
    tp = (y_pred * y_true).sum(dim=dim)
    fn = ((1-y_pred) * y_true).sum(dim=dim)
    fp = (y_pred * (1-y_true)).sum(dim=dim)
    beta = 1. - alpha
    ti = (tp + epsilon) / (tp + (alpha*fn) + (beta*fp) + epsilon)
    return ti

def tversky_loss(y_true, y_pred):
    return 1. - tversky_index(y_true, y_pred)

'''
def focal_tversky_loss(y_true, y_pred, gamma=.75):
    t_loss = tversky_loss(y_true, y_pred)
    return torch.pow(t_loss, gamma)
'''

