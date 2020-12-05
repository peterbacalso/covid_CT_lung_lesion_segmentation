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
from modules.util.augmentation import SegmentationAugmentation
from modules.util.loss_funcs import dice_loss, cross_entropy_loss
from modules.dsets import (TrainingCovid2dSegmentationDataset, collate_fn,
                          Covid2dSegmentationDataset, get_ct)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

METRICS_SIZE = 6
METRICS_LOSS_IDX = 0
METRICS_TP_IDX = 1
METRICS_FN_IDX = 2
METRICS_FP_IDX = 3
METRICS_CE_LOSS_IDX = 4
METRICS_DICE_LOSS_IDX = 5

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
            default=1,
            type=int
        )
        parser.add_argument(
            '--steps-per-epoch',
            help='default 160',
            default=160,
            type=int
        )
        parser.add_argument(
            '--val-cadence',
            help='Run validation at every val-cadence-th epoch. Validation will always run at epoch 1',
            default=1,
            type=int
        )
        parser.add_argument(
            '--depth',
            help='UNet Depth',
            default=4,
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
        parser.add_argument('--width-irc',
            nargs='+',
            help='Pass 3 values: Index, Row, Column',
            default=[7,192,192]
        )
        parser.add_argument(
            '--ct-window',
            help='Specify CT window: one of (none, lung, mediastinal, shifted_lung)',
            default=None,
            type=str
        )
        parser.add_argument(
            '--notes',
            help='notes to log in wandb',
            default='',
            type=str
        )
        parser.add_argument('--model-path',
            help="Path to the saved segmentation model for resuming training",
            nargs='?',
            default=None
        )

        self.cli_args = parser.parse_args(argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        if self.cli_args.epochs == 0:
            wandb.init(project=self.cli_args.project_name, 
                       notes=self.cli_args.notes, name=self.cli_args.run_name)
            raise
        if self.cli_args.run_name is None:
            wandb.init(project=self.cli_args.project_name, 
                       notes=self.cli_args.notes, config=self.cli_args)
        else:
            wandb.init(project=self.cli_args.project_name, 
                       notes=self.cli_args.notes,
                       config=self.cli_args,
                       name=self.cli_args.run_name)

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
        #self.seg_model, self.aug_model = self.init_model()
        self.seg_model = self.init_model()
        wandb.watch(self.seg_model) # apparently magic
        self.train_dl, self.valid_dl = self.init_dl()
        self.dice_loss, self.ce_loss = self.init_loss_func()
        self.sliding_window = self.init_sliding_window()
        self.total_training_samples_count = 0
        self.batch_count = 0

        if self.cli_args.model_path is not None:
            self.resume_training()

        self.optim = self.init_optim()
        self.scheduler = self.init_scheduler()

    def init_model(self):
        seg_model = CovidSegNetWrapper(
            in_channels=1,
            n_classes=2,
            depth=self.cli_args.depth,
            wf=4,
            padding=True)

        #aug_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                #aug_model = nn.DataParallel(aug_model)
            seg_model = seg_model.to(self.device)
            #aug_model = aug_model.to(self.device)

        return seg_model #, aug_model

    def init_optim(self, lr=3e-2, momentum=.99):
        return SGD(self.seg_model.parameters(), lr=lr, 
                   momentum=momentum, weight_decay=1e-4)
        #return Adam(self.seg_model.parameters(), lr=lr)

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
        splitter = partial(list_stride_splitter, val_stride=5)

        train_ds = TrainingCovid2dSegmentationDataset(
            steps_per_epoch=self.cli_args.steps_per_epoch,
            window=self.cli_args.ct_window,
            augmentation_dict=self.augmentation_dict,
            is_valid=False,
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
            pin_memory=self.use_cuda,
            collate_fn=collate_fn)

        valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_size//2,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda)

        return train_dl, valid_dl

    def init_scheduler(self):
        return OneCycleLR(self.optim, max_lr=3e-1,
                          steps_per_epoch=len(self.train_dl),
                          epochs=self.cli_args.epochs)

    def resume_training(self):
        model_dict = torch.load(self.cli_args.model_path)
        self.seg_model.load_state_dict(model_dict['model_state'])
        #self.optim.load_state_dict(model_dict['optimizer_state'])


    def batch_loss(self, idx, batch, batch_size, metrics, 
                   is_train=True, thresh=.5, get_sample=False):
        ct_t, mask_t = batch

        ct_g = ct_t.to(self.device, non_blocking=True)
        mask_g = mask_t.to(self.device, non_blocking=True)

        #if self.seg_model.training and self.augmentation_dict:
            #ct_g, mask_g = self.aug_model(ct_g, mask_g)

        pred_g = self.sliding_window(ct_g, self.seg_model)

        dice_loss = self.dice_loss(pred_g, mask_g)
        ce_loss = self.ce_loss(pred_g, mask_g)
        dice_ce_loss = (ce_loss + dice_loss) * .5

        if is_train:
            start_idx = idx * batch_size * 3
            end_idx = start_idx + batch_size * 3
        else:
            start_idx = idx * batch_size
            end_idx = start_idx + batch_size

        with torch.no_grad():
            pred_max = torch.argmax(pred_g, dim=1, keepdim=True)
            pred_bool = pred_max > thresh
            mask_bool = mask_g > thresh

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
        if get_sample:
            return loss, ct_g.detach().cpu(), \
                mask_g.detach().cpu(), pred_max.detach().cpu()
        return loss, None, None, None

    def init_sliding_window(self):
        roi_size = (self.width_irc[0], self.width_irc[1], self.width_irc[2])
        return SlidingWindowInferer(roi_size=roi_size,
                                    sw_batch_size=1,
                                    overlap=.5)

    def one_epoch(self, epoch, dl, mb, train=True):
        if train:
            self.seg_model.train()
            dl.dataset.shuffle()
            self.total_training_samples_count += len(dl.dataset)*3
            metrics = torch.zeros(METRICS_SIZE, len(dl.dataset)*3, device=self.device)
        else:
            self.seg_model.eval()
            metrics = torch.zeros(METRICS_SIZE, len(dl.dataset), device=self.device)


        pb = progress_bar(enumerate(dl), total=len(dl), parent=mb)
        for i, batch in pb:
            if train:
                self.optim.zero_grad()
                loss, ct_t, mask_t, pred_t = self.batch_loss(
                    i, batch, dl.batch_size, metrics, 
                    is_train=train, get_sample=(i==len(dl)-1))
                loss.backward()
                self.optim.step()
                self.scheduler.step()
            else:
                with torch.no_grad():
                    _, ct_t, mask_t, pred_t = self.batch_loss(
                        i, batch, dl.batch_size, metrics, 
                        is_train=train, get_sample=(i==len(dl)-1))

        return metrics.to('cpu'), ct_t, mask_t, pred_t

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
            ).format(epoch, mode_str, **metrics_dict))

        wandb.log(metrics_dict, step=self.total_training_samples_count)

        return metrics_dict

    def log_images(self, epoch, mode_str, dl, ct_t, mask_t, pred_t, thresh=.5):
        mask_list = []
        for slice_idx in range(ct_t.shape[-3]):
            pred_bool = pred_t.numpy()[0][0][slice_idx] > thresh 
            mask_bool = mask_t.numpy()[0][0][slice_idx] > thresh
            ct_slice = ct_t.numpy()[0][0][slice_idx]

            mask_data = np.zeros_like(pred_bool).astype(np.int)
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

            image = np.expand_dims(ct_slice.squeeze(), axis=-1)
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
                  'pr_val/precision', 'pr_val/recall', 'pr_val/f1_score'], 
                 table=True)
        for epoch in mb:
            trn_metrics, ct_t_trn, mask_t_trn, pred_t_trn = self.one_epoch(
                epoch, dl=self.train_dl, mb=mb)
            trn_metrics_dict = self.log_metrics(epoch, mode_str='trn',
                                                  metrics=trn_metrics)
            if epoch == 1 or epoch % self.cli_args.val_cadence== 0:
                val_metrics, ct_t_val, mask_t_val, pred_t_val = self.one_epoch(
                    epoch, dl=self.valid_dl, mb=mb, train=False)
                val_metrics_dict = self.log_metrics(epoch, mode_str='val',
                                                    metrics=val_metrics)
                self.log_images(epoch, mode_str='trn', dl=self.train_dl, 
                                ct_t=ct_t_trn, mask_t=mask_t_trn, 
                                pred_t=pred_t_trn)
                self.log_images(epoch, mode_str='val', dl=self.valid_dl,
                                ct_t=ct_t_val, mask_t=mask_t_val, 
                                pred_t=pred_t_val)
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
                    "{:.4f}".format(val_metrics_dict['pr_val/f1_score'])
                ], table=True)


if __name__=='__main__':
    CovidSegmentationTrainingApp().main()


