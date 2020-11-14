import sys
import torch
import argparse

from functools import partial

from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# local imports
from module.models import UNetWrapper
from modules.util.logconf import logging
from module.util.util import list_stride_splitter
from module.util.augmentation import SegmentationAugmentation
from module.dsets import TrainingCovid2dSegmentationDataset, Covid2dSegmentationDataset

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class CovidSegmentationTrainingApp:

    def __init__(self, argv=None):
        if argv is None:
            argv = sys.argv[1:]

        parser = argparse.ArgumentParser(
            description='CLI to configure Covid training')
        parser.add_argument(
            '--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int
        )
        parser.add_argument(
            '--batch-size',
            help='Number of items to load per call to data loader',
            default=64,
            type=int
        )
        parser.add_argument(
            '--epochs',
            help='Total iterations to feed the entire dataset to the model',
            default=1,
            type=int
        )
        parser.add_argument(
            '--val-stride',
            help='Run validation at every val-stride-th epoch. Validation is always run at epoch 1',
            default=5,
            type=int
        )
        parser.add_argument(
            '--run-name',
            help='Name of run to display in wandb',
            default=None,
            type=str
        )
        parser.add_argument(
            '--augment-flip',
            help='Augment the training data by randomly flipping the data left-right, up-down, and front-back.',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augment-offset',
            help='Augment the training data by randomly offsetting the data slightly along the X, Y, and Z axes.',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augment-scale',
            help='Augment the training data by randomly increasing or decreasing the size of the candidate.',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augment-rotate',
            help='Augment the training data by randomly rotating the data around the head-foot axis.',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augment-noise',
            help='Augment the training data by randomly adding noise to the data.',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augmented',
            help='Augment the training data.',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--pad-type',
            help='Conv padding: one of (zero, reflect, replicate)',
            default='zero',
            type=str
        )
        parser.add_argument('--width-irc',
            nargs='+',
            help='Pass 3 values: Index, Row, Column',
            default=None,
            required=True
        )

        self.cli_args = parser.parse_args(argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        assert self.cli_args.pad_type in ('zero', 'reflect', 'replicate'), repr(self.cli_args.pad_type)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = .1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = .2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.seg_model, self.aug_model = self.init_model()
        self.optim = self.init_optim()
        self.train_dl, self.valid_dl = self.init_dl()
        self.scheduler = self.init_scheduler()
        self.loss_func = self.init_loss_func()
        self.total_training_samples_count = 0
        self.width_irc = tuple([int(axis) for axis in self.cli_args.width_irc])

    def init_model(self):
        seg_model = UNetWrapper(
            in_channels=self.width_irc[0],
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            pad_type=self.cli_args.pad_type,
            batch_norm=True,
            up_mode='upconv')

        aug_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                aug_model = nn.DataParallel(aug_model)
            seg_model = seg_model.to(self.device)
            aug_model = aug_model.to(self.device)

        return seg_model, aug_model

    def init_optim(self, lr=3e-3, momentum=.99):
        return SGD(self.seg_model.parameters(), lr=lr, 
                   momentum=momentum, weight_decay=1e-4)

    def init_loss_func(self):
        def dice_loss(pred_g, label_g, epsilon=1):
            dice_correct = (pred_g * label_g).sum(dim=[1,2,3])
            dice_label_g = label_g.sum(dim=[1,2,3])
            dice_pred_g = pred_g.sum(dim=[1,2,3])

            dice_ratio = (2 * dice_correct + epsilon) \
                / (dice_label_g + dice_pred_g + epsilon)
            return 1 - dice_ratio
        return dice_loss

    def init_dl(self):
        splitter = partial(list_stride_splitter, val_stride=10)

        train_ds = TrainingCovid2dSegmentationDataset(
            splitter=splitter,
            width_irc=self.width_irc)

        valid_ds = Covid2dSegmentationDataset(
            is_valid=True,
            splitter=splitter)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda)

        valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda)

        return train_dl, valid_dl

    def init_scheduler(self):
        # janky solution for finishing wandb run in jupyter
        epochs = 1 if self.cli_args.epochs == 0 else self.cli_args.epochs 

        return OneCycleLR(self.optim, max_lr=3e-1,
                          steps_per_epoch=len(self.train_dl),
                          epochs=epochs)

    def main(self):
        pass

 


