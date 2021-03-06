import sys
import torch
import shutil
import hashlib
import datetime
import argparse
import numpy as np

from pathlib import Path
from functools import partial
from monai.inferers import SlidingWindowInferer
from fastprogress.fastprogress import master_bar, progress_bar

from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# local imports
from modules.util.logconf import logging
from modules.model import CovidSegNetWrapper
from modules.util.util import list_stride_splitter
from modules.util.loss_funcs import dice_loss, cross_entropy_loss, surface_dist
from modules.dsets import (TrainingCovid2dSegmentationDataset, collate_fn,
                          Covid2dSegmentationDataset, get_ct)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

METRICS_SIZE = 9
METRICS_LOSS_IDX = 0
METRICS_TP_IDX = 1
METRICS_FN_IDX = 2
METRICS_FP_IDX = 3
METRICS_CE_LOSS_IDX = 4
METRICS_DICE_LOSS_IDX = 5
METRICS_MSD_IDX = 6
METRICS_RMSD_IDX = 7
METRICS_HAUSD_IDX = 8

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
            default=2,
            type=int
        )
        parser.add_argument(
            '--epochs',
            help='Total iterations to feed the entire dataset to the model',
            default=600,
            type=int
        )
        parser.add_argument(
            '--steps-per-epoch',
            help='default 320',
            default=320,
            type=int
        )
        parser.add_argument(
            '--val-cadence',
            help='Run validation at every val-cadence-th epoch. Validation will always run at epoch 1',
            default=5,
            type=int
        )
        parser.add_argument(
            '--depth',
            help='UNet Depth',
            default=3,
            type=int
        )
        parser.add_argument(
            '--lr',
            help='Learning rate',
            default=0.1,
            type=float
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
            default=True
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
        parser.add_argument('--model-path',
            help="Path to the saved segmentation model for resuming training",
            nargs='?',
            default=None
        )
        parser.add_argument('--freeze',
            help='Set flag to freeze model during transfer learning.',
            action='store_true',
            default=False
        )

        self.cli_args = parser.parse_args(argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        if self.cli_args.ct_window is not None:
            assert self.cli_args.ct_window in ('lung', 'mediastinal', 'shifted_lung')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = .2
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = .1
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.

        self.width_irc = tuple([int(axis) for axis in self.cli_args.width_irc])
        self.seg_model = self.init_model()
        self.train_dl, self.valid_dl = self.init_dl()
        self.dice_loss, self.ce_loss = self.init_loss_func()
        self.sliding_window = self.init_sliding_window()
        self.total_training_samples_count = 0
        self.batch_count = 0

        self.optim = self.init_optim()
        self.scheduler = self.init_scheduler()

    def init_model(self):
        seg_model = CovidSegNetWrapper(
            in_channels=1,
            n_classes=2,
            depth=self.cli_args.depth,
            wf=4,
            padding=True)
        
        if self.cli_args.model_path is not None:
            model_dict = torch.load(self.cli_args.model_path)
            seg_model.load_state_dict(model_dict['model_state'])
            if self.cli_args.freeze:
                for i, top_layer in enumerate(self.seg_model.children()):
                    if i == 1:
                        for j, layer in enumerate(top_layer.children()):
                            if j < 2:
                                for param in layer.parameters():
                                    param.requires_grad = False

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
            seg_model = seg_model.to(self.device)

        return seg_model

    def init_optim(self):
        optim = SGD(self.seg_model.parameters(), lr=self.cli_args.lr, 
                    momentum=.99, weight_decay=1e-4)
        if self.cli_args.model_path is not None:
            model_dict = torch.load(self.cli_args.model_path)
            optim.load_state_dict(model_dict['optimizer_state'])
        return optim
        
    def init_scheduler(self):
        scheduler = OneCycleLR(self.optim, max_lr=3e-1,
                               steps_per_epoch=len(self.train_dl),
                               epochs=self.cli_args.epochs)
        if self.cli_args.model_path is not None:
            model_dict = torch.load(self.cli_args.model_path)
            scheduler.load_state_dict(model_dict['scheduler_state'])
        return scheduler

    def init_loss_func(self):
        def dice_loss_func(pred_g, label_g, epsilon=1):
            d_loss = dice_loss(torch.argmax(pred_g, dim=1, keepdim=True).float(),
                               label_g)
            return d_loss
        def ce_loss_func(pred_g, label_g, epsilon=1):
            ce_loss = cross_entropy_loss(pred_g,
                                         torch.squeeze(label_g, dim=1).long())
            return ce_loss
        return dice_loss_func, ce_loss_func

    def init_dl(self):
        splitter = partial(list_stride_splitter, val_stride=10)

        train_ds = TrainingCovid2dSegmentationDataset(
            steps_per_epoch=self.cli_args.steps_per_epoch,
            window=self.cli_args.ct_window,
            is_valid=False,
            splitter=splitter,
            width_irc=self.width_irc)

        valid_ds = Covid2dSegmentationDataset(
            window=self.cli_args.ct_window,
            is_valid=True,
            splitter=splitter)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn=collate_fn)

        valid_dl = DataLoader(
            valid_ds,
            batch_size=1,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda)

        return train_dl, valid_dl

    def batch_loss(self, idx, batch, batch_size, metrics, 
                   is_train=True, thresh=.5): 
        ct_t, mask_t, spacings = batch

        ct_g = ct_t.to(self.device, non_blocking=True)
        mask_g = mask_t.to(self.device, non_blocking=True)

        if is_train:
            pred_g = self.seg_model(ct_g)
        else:
            pred_g = self.sliding_window(ct_g, self.seg_model)

        dice_loss = self.dice_loss(pred_g, mask_g)
        ce_loss = self.ce_loss(pred_g, mask_g)
        dice_ce_loss = (dice_loss * .7) + (ce_loss * .3)                

        if is_train:
            start_idx = idx * batch_size*3
            end_idx = start_idx + batch_size*3
        else:
            start_idx = idx * batch_size
            end_idx = start_idx + batch_size

        with torch.no_grad():
            pred_max = torch.argmax(pred_g, dim=1, keepdim=True)
            pred_bool = pred_max > thresh
            mask_bool = mask_g > thresh
            
            if not is_train:
                mean_dists = []
                rms_dists = []
                haus_dists = []
                for pred, mask, spacing in zip(pred_max,mask_g,spacings):
                    pred_a = pred.cpu().detach().numpy()
                    mask_a = mask.cpu().detach().numpy()
                    spacing_a = spacing.cpu().detach().numpy()
                    sd = surface_dist(pred_a, mask_a, spacing_a)
                    mean_dists.append(torch.tensor(sd.mean()))
                    rms_dists.append(torch.tensor(np.sqrt((sd**2).mean())))
                    haus_dists.append(torch.tensor(sd.max()))
                metrics[METRICS_MSD_IDX, start_idx:end_idx] = torch.tensor(mean_dists)
                metrics[METRICS_RMSD_IDX, start_idx:end_idx] = torch.tensor(rms_dists)
                metrics[METRICS_HAUSD_IDX, start_idx:end_idx] = torch.tensor(haus_dists)

            tp = (pred_bool * mask_bool).sum(dim=[-4,-3,-2,-1])
            fn = (~pred_bool * mask_bool).sum(dim=[-4,-3,-2,-1])
            fp = (pred_bool * ~mask_bool).sum(dim=[-4,-3,-2,-1])

            metrics[METRICS_LOSS_IDX, start_idx:end_idx] = dice_ce_loss
            metrics[METRICS_TP_IDX, start_idx:end_idx] = tp 
            metrics[METRICS_FN_IDX, start_idx:end_idx] = fn
            metrics[METRICS_FP_IDX, start_idx:end_idx] = fp 
            metrics[METRICS_CE_LOSS_IDX, start_idx:end_idx] = ce_loss
            metrics[METRICS_DICE_LOSS_IDX, start_idx:end_idx] = dice_loss

        self.batch_count += 1

        loss = dice_ce_loss.mean()
        return loss

    def init_sliding_window(self):
        roi_size = (self.width_irc[0], self.width_irc[1], self.width_irc[2])
        return SlidingWindowInferer(roi_size=roi_size,
                                    sw_batch_size=1,
                                    overlap=.2)

    def one_epoch(self, epoch, dl, mb, train=True):
        if train:
            self.seg_model.train()
            dl.dataset.shuffle()
            self.total_training_samples_count += len(dl.dataset)
            metrics = torch.zeros(METRICS_SIZE, len(dl.dataset)*3, device=self.device)
        else:
            self.seg_model.eval()
            metrics = torch.zeros(METRICS_SIZE, len(dl.dataset), device=self.device)


        pb = progress_bar(enumerate(dl), total=len(dl), parent=mb)
        for i, batch in pb:
            if train:
                self.optim.zero_grad()
                loss = self.batch_loss(i, batch, dl.batch_size, 
                                       metrics, is_train=train) 
                loss.backward()
                self.optim.step()
                self.scheduler.step()
            else:
                with torch.no_grad():
                    self.batch_loss(i, batch, dl.batch_size, metrics, 
                                    is_train=train)

        return metrics.to('cpu')

    def log_metrics(self, epoch, mode_str, metrics):
        log.info("E{} {}".format(
            epoch,
            type(self).__name__,
        ))

        metrics_a = metrics.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        mask_voxel_count = sum_a[METRICS_TP_IDX] + sum_a[METRICS_FN_IDX]

        metrics_dict = {}
        metrics_dict[f'loss/{mode_str}'] = metrics_a[METRICS_LOSS_IDX].mean()
        metrics_dict[f'dice_loss/{mode_str}'] = metrics_a[METRICS_DICE_LOSS_IDX].mean()
        metrics_dict[f'ce_loss/{mode_str}'] = metrics_a[METRICS_CE_LOSS_IDX].mean()

        if mode_str == 'val':
            metrics_dict[f'surface_dist_{mode_str}/mean'] = metrics_a[METRICS_MSD_IDX].mean()
            metrics_dict[f'surface_dist_{mode_str}/root_mean_squared'] = metrics_a[METRICS_RMSD_IDX].mean()
            metrics_dict[f'surface_dist_{mode_str}/hausdorff'] = metrics_a[METRICS_HAUSD_IDX].mean()

        metrics_dict[f'metrics_{mode_str}/miss_rate'] = \
            sum_a[METRICS_FN_IDX] / (mask_voxel_count or 1)
        metrics_dict[f'metrics_{mode_str}/fp_to_mask_ratio'] = \
            sum_a[METRICS_FP_IDX] / (mask_voxel_count or 1)

        precision = metrics_dict[f'pr_{mode_str}/precision'] = \
            sum_a[METRICS_TP_IDX] \
            / ((sum_a[METRICS_TP_IDX] + sum_a[METRICS_FP_IDX]) or 1)
        recall = metrics_dict[f'pr_{mode_str}/recall'] = \
            sum_a[METRICS_TP_IDX] / (mask_voxel_count or 1)

        metrics_dict[f'pr_{mode_str}/f1_score'] = \
            2 * (precision * recall) / ((precision + recall) or 1)

        if mode_str=='trn':
            log.info(("E{} {:8} "
                      + "{loss/trn:.4f} loss, "
                      + "{dice_loss/trn:.4f} dice loss, "
                      + "{ce_loss/trn:.4f} ce loss, "
                      + "{pr_trn/precision:.4f} precision, "
                      + "{pr_trn/recall:.4f} recall, "
                      + "{pr_trn/f1_score:.4f} f1 score "
                      + "{metrics_trn/miss_rate:.4f} miss rate "
                      + "{metrics_trn/fp_to_mask_ratio:.4f} fp to label ratio"
            ).format(epoch, mode_str, **metrics_dict))
        else:
            log.info(("E{} {:8} "
                      + "{loss/val:.4f} loss, "
                      + "{dice_loss/val:.4f} dice loss, "
                      + "{ce_loss/val:.4f} ce_loss, "
                      + "{pr_val/precision:.4f} precision, "
                      + "{pr_val/recall:.4f} recall, "
                      + "{pr_val/f1_score:.4f} f1 score "
                      + "{metrics_val/miss_rate:.4f} miss rate "
                      + "{metrics_val/fp_to_mask_ratio:.4f} fp to label ratio"
                      + "{surface_dist_val/mean:.4f} mean distance"
                      + "{surface_dist_val/root_mean_squared:.4f} root mean squared distance"
                      + "{surface_dist_val/hausdorff:.4f} hausdorff distance"
            ).format(epoch, mode_str, **metrics_dict))

        return metrics_dict

    def save_model(self, epoch, is_best=False):
        folder_path = Path(f'saved-models/')
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path/f'{self.time_str}.{self.total_training_samples_count}.state'
        model = self.seg_model
        if isinstance(model, nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optim.state_dict(),
            'optimizer_name': type(self.optim).__name__,
            'scheduler_state': self.scheduler.state_dict(),
            'scheduler_name': type(self.scheduler).__name__,
            'epoch': epoch,
            'total_training_samples_count': self.total_training_samples_count,
            'depth': self.cli_args.depth,
            'window': self.cli_args.ct_window
        }

        torch.save(state, file_path)
        log.info(f'Saved model params to {file_path}')

        if is_best:
            best_path = Path(f'saved-models/{self.time_str}.best.state')
            shutil.copy(file_path, best_path)
            log.info(f'Saved model params to {best_path}')

        # output sha1 of model saved 
        # this can help with debugging if file names get mixed up
        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    def main(self):

        log.info(f"Starting {type(self).__name__}, {self.cli_args}")
        best_score = 0.
        mb = master_bar(range(1, self.cli_args.epochs+1))
        mb.write(['epoch', 'loss/trn', 'dice_loss/trn', 'ce_loss/trn', 
                  'loss/val', 'dice_loss/val', 'ce_loss/val',
                  'metrics_val/miss_rate', 'metrics_val/fp_to_mask_ratio',
                  'pr_val/precision', 'pr_val/recall', 'pr_val/f1_score',
                  'surface_dist_val/mean', 'surface_dist_val/root_mean_squared',
                  'surface_dist_val/hausdorff'], 
                 table=True)
        for epoch in mb:
            trn_metrics = self.one_epoch(epoch, dl=self.train_dl, mb=mb)
            trn_metrics_dict = self.log_metrics(epoch, mode_str='trn',
                                                  metrics=trn_metrics)
            if epoch == 1 or epoch % self.cli_args.val_cadence== 0:
                val_metrics = self.one_epoch(epoch, dl=self.valid_dl, mb=mb, train=False)
                val_metrics_dict = self.log_metrics(epoch, mode_str='val',
                                                    metrics=val_metrics)
                best_score = max(best_score, val_metrics_dict['pr_val/f1_score'])
                self.save_model(epoch, val_metrics_dict['pr_val/f1_score']==best_score)
                mb.write([
                    epoch,
                    "{:.4f}".format(trn_metrics_dict['loss/trn']),
                    "{:.4f}".format(trn_metrics_dict['dice_loss/trn']),
                    "{:.4f}".format(trn_metrics_dict['ce_loss/trn']),
                    "{:.4f}".format(val_metrics_dict['loss/val']),
                    "{:.4f}".format(val_metrics_dict['dice_loss/val']),
                    "{:.4f}".format(val_metrics_dict['ce_loss/val']),
                    "{:.4f}".format(val_metrics_dict['metrics_val/miss_rate']),
                    "{:.4f}".format(val_metrics_dict['metrics_val/fp_to_mask_ratio']),
                    "{:.4f}".format(val_metrics_dict['pr_val/precision']),
                    "{:.4f}".format(val_metrics_dict['pr_val/recall']),
                    "{:.4f}".format(val_metrics_dict['pr_val/f1_score']),
                    "{:.4f}".format(val_metrics_dict['surface_dist_val/mean']),
                    "{:.4f}".format(val_metrics_dict['surface_dist_val/root_mean_squared']),
                    "{:.4f}".format(val_metrics_dict['surface_dist_val/hausdorff'])
                ], table=True)


if __name__=='__main__':
    CovidSegmentationTrainingApp().main()

