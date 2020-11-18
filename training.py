import sys
import torch
import wandb
import shutil
import hashlib
import datetime
import argparse
import numpy as np

from pathlib import Path
from functools import partial
from fastprogress.fastprogress import master_bar, progress_bar

from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# local imports
from modules.model import UNetWrapper
from modules.util.logconf import logging
from modules.util.util import list_stride_splitter
from modules.util.augmentation import SegmentationAugmentation
from modules.dsets import (TrainingCovid2dSegmentationDataset, 
                          Covid2dSegmentationDataset, get_ct)

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
            '--steps-per-epoch',
            help='default 10000',
            default=10000,
            type=int
        )
        parser.add_argument(
            '--val-cadence',
            help='Run validation at every val-cadence-th epoch. Validation will always run at epoch 1',
            default=1,
            type=int
        )
        parser.add_argument(
            '--recall-priority',
            help='Prioritize recall over precision by (int) times more',
            default=0,
            type=int
        )
        parser.add_argument(
            '--depth',
            help='UNet Depth',
            default=3,
            type=int
        )
        parser.add_argument(
            '--project-name',
            help='Name of project to save in wandb',
            default='covid19_seg',
            type=str
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
            default=[7,60,60]
        )
        parser.add_argument(
            '--ct-window',
            help='Specify CT window: one of (none, lung, mediastinal, dist)',
            default=None,
            type=str
        )

        self.cli_args = parser.parse_args(argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        if self.cli_args.epochs == 0:
            wandb.init(project=self.cli_args.project_name, name=self.cli_args.run_name)
            raise
        if self.cli_args.run_name is None:
            wandb.init(project=self.cli_args.project_name, config=self.cli_args)
        else:
            wandb.init(project=self.cli_args.project_name, 
                       config=self.cli_args,
                       name=self.cli_args.run_name)

        assert self.cli_args.pad_type in ('zero', 'reflect', 'replicate'), repr(self.cli_args.pad_type)
        if self.cli_args.ct_window is not None:
            assert self.cli_args.ct_window in ('lung', 'mediastinal', 'dist')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = .3
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = .2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.width_irc = tuple([int(axis) for axis in self.cli_args.width_irc])
        self.seg_model, self.aug_model = self.init_model()
        wandb.watch(self.seg_model) # apparently magic
        self.optim = self.init_optim()
        self.train_dl, self.valid_dl = self.init_dl()
        self.scheduler = self.init_scheduler()
        self.loss_func = self.init_loss_func()
        self.total_training_samples_count = 0
        self.batch_count = 0

    def init_model(self):
        seg_model = UNetWrapper(
            in_channels=self.width_irc[0],
            n_classes=1,
            depth=self.cli_args.depth,
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

    def init_optim(self, lr=1e-3, momentum=.99):
        #return SGD(self.seg_model.parameters(), lr=lr, 
        #           momentum=momentum, weight_decay=1e-4)
        return Adam(self.seg_model.parameters())

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
            steps_per_epoch=self.cli_args.steps_per_epoch,
            window=self.cli_args.ct_window,
            splitter=splitter,
            width_irc=self.width_irc)

        valid_ds = Covid2dSegmentationDataset(
            window=self.cli_args.ct_window,
            is_valid=True,
            splitter=splitter,
            width_irc=self.width_irc)

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
        return OneCycleLR(self.optim, max_lr=3e-1,
                          steps_per_epoch=len(self.train_dl),
                          epochs=self.cli_args.epochs)

    def batch_loss(self, idx, batch, batch_size, metrics, thresh=.5):
        hu, mask, uid_list, slice_idx_list = batch

        hu_g = hu.to(self.device, non_blocking=True)
        mask_g = mask.to(self.device, non_blocking=True)

        if self.seg_model.training and self.augmentation_dict:
            hu_g, mask_g = self.aug_model(hu_g, mask_g)

        pred_g = self.seg_model(hu_g)

        dice_loss = self.loss_func(pred_g, mask_g)
        fine_loss = self.loss_func(pred_g*mask_g, mask_g)

        start_idx = idx * batch_size
        end_idx = start_idx + batch_size

        with torch.no_grad():
            pred_bool = pred_g > thresh
            mask_bool = mask_g > thresh

            tp = (pred_bool * mask_bool).sum(dim=[1,2,3])
            fn = (~pred_bool * mask_bool).sum(dim=[1,2,3])
            fp = (pred_bool * ~mask_bool).sum(dim=[1,2,3])

            metrics[METRICS_LOSS_IDX, start_idx:end_idx] = dice_loss
            metrics[METRICS_TP_IDX, start_idx:end_idx] = tp 
            metrics[METRICS_FN_IDX, start_idx:end_idx] = fn
            metrics[METRICS_FP_IDX, start_idx:end_idx] = fp 

        self.batch_count += 1

        # we want to maximize recall so we give the false negatives 
        # a larger impact on the loss (recall_priority times more)
        return dice_loss.mean() + (fine_loss.mean() * self.cli_args.recall_priority)


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

        mask_voxel_count = sum_a[METRICS_TP_IDX] + sum_a[METRICS_FN_IDX]

        metrics_dict = {}
        metrics_dict[f'loss/{mode_str}'] = metrics_a[METRICS_LOSS_IDX].mean()

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
                      + "{pr_trn/precision:.4f} precision, "
                      + "{pr_trn/recall:.4f} recall, "
                      + "{pr_trn/f1_score:.4f} f1 score "
                      + "{metrics_trn/miss_rate:.4f} miss rate "
                      + "{metrics_trn/fp_to_mask_ratio:.4f} fp to label ratio"
            ).format(epoch, mode_str, **metrics_dict))
        else:
            log.info(("E{} {:8} "
                      + "{loss/val:.4f} loss, "
                      + "{pr_val/precision:.4f} precision, "
                      + "{pr_val/recall:.4f} recall, "
                      + "{pr_val/f1_score:.4f} f1 score "
                      + "{metrics_val/miss_rate:.4f} miss rate "
                      + "{metrics_val/fp_to_mask_ratio:.4f} fp to label ratio"
            ).format(epoch, mode_str, **metrics_dict))

        wandb.log(metrics_dict, step=self.total_training_samples_count)

        return metrics_dict

    def log_images(self, epoch, mode_str, dl, thresh=.5):
        self.seg_model.eval()
        uid_list = dl.dataset.coords.groupby(by='uid').first()\
            .reset_index().sort_values(by='uid')[:8]
        mask_list = []
        for i, row in uid_list.iterrows():
            uid, *_ = row
            ct = get_ct(uid)

            for slice_idx in range(6):
                if mode_str=='trn':
                    ct_slice = dl.dataset.getitem_cropslice(row)
                else:
                    ct_idx = slice_idx * (ct.hu.shape[0] - 1) // 5
                    ct_slice = dl.dataset.getitem_fullslice(uid, ct_idx)

                hu, mask, *_ = ct_slice 
                hu_g = hu.to(self.device).unsqueeze(0)
                mask_g = mask.to(self.device).unsqueeze(0)

                pred_g = self.seg_model(hu_g)[0]
                pred_bool = pred_g.cpu().detach().numpy()[0] > thresh 
                mask_bool = mask_g.cpu().numpy()[0][0] > thresh

                # normalize range 
                hu[:-1,:,:] /= 2000 # range -0.5 to 0.5
                hu[:-1,:,:] += .5   # range 0 to 1
                if mode_str == 'trn':
                    hu_slice = hu[slice_idx % hu.shape[0]].numpy()
                else:
                    hu_slice = hu[dl.dataset.context_slice_count].numpy()

                mask_data = np.zeros_like(pred_bool.squeeze()).astype(np.int)
                mask_data += 1 * pred_bool & mask_bool # true positives
                mask_data += 2 * (~pred_bool & mask_bool) # false negatives 
                mask_data += 3 * (pred_bool & ~mask_bool) # false positives

                class_labels = {
                    1: "True Positive",
                    2: "False Negative",
                    3: "False Positive"
                }

                truth_mask = np.zeros_like(mask_bool)
                truth_mask += mask_bool # ground truth
                truth_mask = ~truth_mask
                truth_mask = truth_mask.astype(np.int)
                truth_labels = {0: "Lesion"}

                image = np.expand_dims(hu_slice.squeeze(), axis=-1)
                mask_img = wandb.Image(image, masks={
                  "predictions": {
                      "mask_data": mask_data,
                      "class_labels": class_labels
                  },
                  "groud_truth": {
                      "mask_data": truth_mask,
                      "class_labels": truth_labels
                  }
                })
                mask_list.append(mask_img)
        wandb.log({f"predictions_{mode_str}": mask_list},
                  step=self.total_training_samples_count)

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
            'epoch': epoch,
            'total_training_samples_count': self.total_training_samples_count,
            'in_channels': self.width_irc[0],
            'depth': self.cli_args.depth,
            'pad_type': self.cli_args.pad_type,
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
        mb.write(['epoch', 'loss/trn', 'loss/val',
                  'metrics_val/miss_rate', 'metrics_val/fp_to_mask_ratio',
                  'pr_val/precision', 'pr_val/recall', 'pr_val/f1_score'], 
                 table=True)
        for epoch in mb:
            trn_metrics = self.one_epoch(epoch, dl=self.train_dl, mb=mb)
            trn_metrics_dict = self.log_metrics(epoch, mode_str='trn',
                                                  metrics=trn_metrics)
            if epoch == 1 or epoch % self.cli_args.val_cadence== 0:
                val_metrics = self.one_epoch(epoch, dl=self.valid_dl, 
                                             mb=mb, train=False)
                val_metrics_dict = self.log_metrics(epoch, mode_str='val',
                                                    metrics=val_metrics)
                self.log_images(epoch, mode_str='trn', dl=self.train_dl)
                self.log_images(epoch, mode_str='val', dl=self.valid_dl)
                best_score = max(best_score, val_metrics_dict['pr_val/f1_score'])
                self.save_model(epoch, val_metrics_dict['pr_val/f1_score']==best_score)
                mb.write([
                    epoch,
                    "{:.4f}".format(trn_metrics_dict['loss/trn']),
                    "{:.4f}".format(val_metrics_dict['loss/val']),
                    "{:.4f}".format(val_metrics_dict['metrics_val/miss_rate']),
                    "{:.4f}".format(val_metrics_dict['metrics_val/fp_to_mask_ratio']),
                    "{:.4f}".format(val_metrics_dict['pr_val/precision']),
                    "{:.4f}".format(val_metrics_dict['pr_val/recall']),
                    "{:.4f}".format(val_metrics_dict['pr_val/f1_score'])
                ], table=True)


if __name__=='__main__':
    CovidSegmentationTrainingApp().main()


