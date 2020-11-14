import sys
import torch
import wandb
import argparse

from functools import partial
from fastprogress.fastprogress import master_bar, progress_bar

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

METRICS_SIZE = 4
METRICS_LOSS_IDX = 0
METRICS_TP_IDX = 1
METRICS_FN_IDX = 2
METRICS_FP_IDX = 3

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

    def batch_loss(self, idx, batch, batch_size, metrics, thresh=.5):
        input_t, label_t, uid_list, slice_idx_list = batch

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.seg_model.training and self.augmentation_dict:
            input_g, label_g = self.aug_model(input_g, label_g)

        pred_g = self.seg_model(input_g)

        dice_loss = self.loss_func(prediction_g, label_g)
        fine_loss = self.loss_func(prediction_g*label_g, label_g)

        start_idx = idx * batch_size
        end_idx = start_idx + batch_size

        with torch.no_grad():
            pred_bool = (pred_g > thresh).to(torch.float32)

            tp = (pred_bool * label_g).sum(dim=[1,2,3])
            fn = (~pred_bool * label_g).sum(dim=[1,2,3])
            fp = (pred_bool * ~label_g).sum(dim=[1,2,3])

            metrics[METRICS_LOSS_IDX, start_idx:end_idx] = dice_loss
            metrics[METRICS_TP_IDX, start_idx:end_idx] = tp 
            metrics[METRICS_FN_IDX, start_idx:end_idx] = fn
            metrics[METRICS_FP_IDX, start_idx:end_idx] = fp 

        # we want to maximize recall so we give the false negatives 
        # a larger impact on the loss (8 times more)
        return dice_loss.mean() + fine_loss.mean() * 8


    def one_epoch(self, epoch, dl, mb, train=True):
        if train:
            self.seg_model.train()
            dl.dataset.shuffle()
            self.total_training_samples_count += len(dl.dataset)
        else:
            self.seg_model.eval()

        metrics = torch.zeros(METRICS_SIZE, len(dl.dataset), device=self.device)

        pb = progress_bar(enumerate(dl), total=len(dl), parent=mb)
        for i, batch in pb:
            if train:
                self.optim.zero_grad()
                loss = self.batch_loss(i, batch, dl.batch_size, metrics)
                loss.backward()
                self.optim.step()
                self.scheduler.step()
            else:
                with torch.no_grad():
                    self.batch_loss(i, batch, dl.batch_size, metrics)

        return metrics.to('cpu')

    def log_metrics(self, epoch, mode_str, metrics):
        log.info("E{} {}".format(
            epoch,
            type(self).__name__,
        ))

        metrics_a = metrics.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        all_label_count = sum_a[METRICS_TP_IDX] + sum_a[METRICS_FN_IDX]

        metrics_dict = {}
        metrics_dict[f'loss/{mode_str}'] = metrics_a[METRICS_LOSS_IDX].mean()

        metrics_dict[f'metrics_{mode_str}/miss_rate'] = \
            sum_a[METRICS_FN_IDX] / (all_label_count or 1)
        metrics_dict[f'metrics_{mode_str}/fp_to_label_ratio'] = \
            sum_a[METRICS_FP_IDX] / (all_label_count or 1)

        precision = metrics_dict[f'pr_{mode_str}/precision'] = \
            sum_a[METRICS_TP_IDX_SEG] \
            / ((sum_a[METRICS_TP_IDX_SEG] + sum_a[METRICS_FP_IDX_SEG]) or 1)
        recall = metrics_dict[f'pr_{mode_str}/recall'] = \
            sum_a[METRICS_TP_IDX_SEG] / (all_label_count or 1)

        metrics_dict[f'pr_{mode_str}/f1_score'] = \
            2 * (precision * recall) / ((precision + recall) or 1)

        if mode_str=='trn':
            log.info(("E{} {:8} "
                      + "{loss/trn:.4f} loss, "
                      + "{pr_trn/precision:.4f} precision, "
                      + "{pr_trn/recall:.4f} recall, "
                      + "{pr_trn/f1_score:.4f} f1 score"
                      + "{metrics_trn/miss_rate:.4f} miss rate"
                      + "{metrics_trn/fp_to_label_ratio:.4f} fp to label ratio"
            ).format(epoch, mode_str, **metrics_dict))
        else:
            log.info(("E{} {:8} "
                      + "{loss/val:.4f} loss, "
                      + "{pr_val/precision:.4f} precision, "
                      + "{pr_val/recall:.4f} recall, "
                      + "{pr_val/f1_score:.4f} f1 score"
                      + "{metrics_val/miss_rate:.4f} miss rate"
                      + "{metrics_val/fp_to_label_ratio:.4f} fp to label ratio"
            ).format(epoch, mode_str, **metrics_dict))

        wandb.log(metrics_dict, step=self.total_training_samples_count)

        return metrics_dict

    def log_images(self):
        pass

    def save_model(self):
        pass

    def main(self):
        if self.cli_args.run_name is None:
            wandb.init(project="covid19_seg")
        else:
            wandb.init(project="covid19_seg", name=self.cli_args.run_name)
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")
        best_score = 0.
        mb = master_bar(range(1, self.cli_args.epochs+1))
        mb.write(['epoch', 'loss/trn', 'loss/val',
                  'metrics_val/miss_rate', 'metrics_val/fp_to_label_ratio',
                  'pr_val/precision', 'pr_val/recall', 'pr_val/f1_score'], 
                 table=True)
        for epoch in mb:
            train_metrics = self.one_epoch(epoch, dl=self.train_dl, mb=mb)
            train_metrics_dict = self.log_metrics(epoch, mode_str='train',
                                                  metrics=train_metrics)
            if epoch == 1 or epoch % self.val_stride == 0:
                val_metrics = self.one_epoch(epoch, dl=self.valid_dl, 
                                             mb=mb, train=False)
                val_metrics_dict = self.log_metrics(epoch, mode_str='val',
                                                    metrics=val_metrics)
                # self.log_images(epoch, mode_str='val', dl=self.valid_dl)
                best_score = max(best_score, val_metrics_dict['pr_val/recall'])
                # self.save_model(epoch, val_metrics_dict['pr_val/recall']==best_score)
                mb.write([
                    epoch,
                    "{:.4f}".format(trn_metrics_dict['loss/trn']),
                    "{:.4f}".format(val_metrics_dict['loss/val']),
                    "{:.4f}".format(val_metrics_dict['pr_val/precision']),
                    "{:.4f}".format(val_metrics_dict['pr_val/recall']),
                    "{:.4f}".format(val_metrics_dict['pr_val/f1_score']),
                    "{:.4f}".format(val_metrics_dict['metrics_val/miss_rate']),
                    "{:.4f}".format(val_metrics_dict['metrics_val/fp_to_label_ratio'])
                ], table=True)


 


