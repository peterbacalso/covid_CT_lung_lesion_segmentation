import sys
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import SGD
from functools import partial
from torch_lr_finder import LRFinder
from torch.utils.data import DataLoader

from modules.dsets import TrainingCovid2dSegmentationDataset, collate_fn
from modules.model import CovidSegNetWrapper
from modules.util.util import list_stride_splitter
from modules.util.loss_funcs import dice_loss, cross_entropy_loss


model = CovidSegNetWrapper(
    in_channels=1,
    n_classes=2,
    depth=3,
    wf=4,
    padding=True)
model = model.cuda()

lr = 1e-7
momentum=.99
weight_decay=1e-4
optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


class LearningRateFinder:
    
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for the lr finder',
            default=2,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )
        parser.add_argument('--width-irc',
            nargs='+',
            help='Pass 3 values: Index, Row, Column',
            default=[16,128,128]
        )
        parser.add_argument(
            '--ct-window',
            help='Specify CT window: one of (None, lung, mediastinal, shifted_lung)',
            default='shifted_lung',
            type=str
        )
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        self.width_irc = tuple([int(axis) for axis in self.cli_args.width_irc])        
        self.model = self.init_model()
        self.loss_func = self.init_loss_func()
        self.train_dl = self.init_train_dl()
        self.optim = self.init_optim()
        
        self.lr_finder = LRFinder(self.model, self.optim, self.loss_func, device=self.device)
        
    def init_model(self):
        model = CovidSegNetWrapper(
            in_channels=1,
            n_classes=2,
            depth=3,
            wf=4,
            padding=True)

        if self.use_cuda:
            log.info("Using CUDA")
            model = model.to(self.device)
            
        return model
    
    def init_loss_func(self):
        def loss_func(pred_g, label_g):
            ce_loss = cross_entropy_loss(pred_g, torch.squeeze(label_g, dim=1).long())
            d_loss = dice_loss(torch.argmax(pred_g, dim=1, keepdim=True).float(), label_g)
            return ((ce_loss*.3 + d_loss*.7)).mean()
        return loss_func
    
    def init_train_dl(self):
        splitter = partial(list_stride_splitter, val_stride=5) # include every 5th datapoint in validation
        
        train_ds = TrainingCovid2dSegmentationDataset(
            is_valid=False, 
            window=self.cli_args.ct_window,
            splitter=splitter,
            width_irc=self.width_irc)
        
        train_dl = DataLoader(
            train_ds,
            batch_size=self.cli_args.batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn)
        
        return train_dl
    
    def init_optim(self, lr=1e-7, momentum=.99, weight_decay=1e-4):
        optim = SGD(self.seg_model.parameters(), lr=lr, 
                    momentum=momentum, weight_decay=weight_decay)
        return optim
    
    def main(self):
        self.lr_finder.range_test(train_loader=self.train_dl, end_lr=10, num_iter=100)
        self.lr_finder.plot()
        plt.show()
        self.lr_finder.reset()
    
        