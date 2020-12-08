import sys
import torch
import argparse
import numpy as np
import nibabel as nib

from pathlib import Path
from scipy import ndimage
from monai.inferers import SlidingWindowInferer

from torch import nn
from torch.utils.data import DataLoader

# local imports
from modules.model import UNet3dWrapper, CovidSegNetWrapper
from modules.util.logconf import logging
from modules.dsets import Covid2dSegmentationDataset, get_ct

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

Path.ls = lambda x: [o.name for o in x.iterdir()]

class CovidInferenceApp:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for inference',
            default=1,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )
        parser.add_argument('--model-path',
            help="Path to the saved segmentation model",
            nargs='?',
            required=True
        )
        parser.add_argument('--data-path',
            help="Path to the data to infer",
            nargs='?',
            required=True
        )
        parser.add_argument('--run-all',
            help='Run over all data rather than a single CT.',
            action='store_true',
            default=False,
        )
        parser.add_argument('uid',
            nargs='?',
            default=None,
            help="CT UID to use.",
        )
        parser.add_argument('--width-irc',
            nargs='+',
            help='Pass 3 values: Index, Row, Column',
            default=[12,192,192]
        )

        self.cli_args = parser.parse_args(sys_argv)
        if not (bool(self.cli_args.uid) ^ self.cli_args.run_all):
            raise Exception("One and only one of uid and --run-all should be given")
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.window = None

        if not self.cli_args.model_path:
            raise Exception("Path to segmentation model should be given")

        self.width_irc = tuple([int(axis) for axis in self.cli_args.width_irc])
        self.model = self.init_model()
        self.sliding_window = self.init_sliding_window()

    def init_model(self):
        log.debug(self.cli_args.model_path)
        model_dict = torch.load(self.cli_args.model_path)

        self.window = model_dict['window']
        
        '''
        model = UNet3dWrapper(
            in_channels=1,
            n_classes=1,
            depth=model_dict['depth'],
            wf=4,
            padding=True,
            pad_type=model_dict['pad_type'],
            batch_norm=True,
            up_mode='upconv')
        '''

        model = CovidSegNetWrapper(
            in_channels=1,
            n_classes=2,
            depth=model_dict['depth'],
            wf=4,
            padding=True)

        model.load_state_dict(model_dict['model_state'])
        model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(self.device)

        return model

    def init_sliding_window(self):
        roi_size = (self.width_irc[0], self.width_irc[1], self.width_irc[2])
        return SlidingWindowInferer(roi_size=roi_size,
                                    sw_batch_size=1,
                                    overlap=.2)

    def init_dl(self, uid):
        ds = Covid2dSegmentationDataset(
            uid=uid,
            window=self.window)
            #width_irc=self.width_irc)
        dl = DataLoader(
            ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() \
                                                   if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda)
        return dl

    def segment_ct(self, ct, uid):
        with torch.no_grad():
            dl = self.init_dl(uid)
            for ct_t,_,__ in dl:
                log.info(f"input shape {ct_t.squeeze().shape}")
                ct_g = ct_t.to(self.device)
                preds = self.sliding_window(ct_g, self.model)
                preds = torch.argmax(preds, dim=1, keepdim=True).float()
                n = 1.0
                for dims in [[-2], [-1]]:
                    flip_ct_g = torch.flip(ct_g, dims=dims)
                    flip_pred = self.sliding_window(flip_ct_g, self.model)
                    flip_pred = torch.argmax(flip_pred, dim=1, keepdim=True).float()
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            preds = preds.detach().cpu().squeeze().numpy()
            mask = preds > .5
            log.info(f"output shape {mask.shape}")
        return mask

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        output_folder_path= Path(f'submission/')
        output_folder_path.mkdir(parents=True, exist_ok=True)

        if self.cli_args.uid:
            uid_set = set(self.cli_args.uid.split(','))
        elif self.cli_args.run_all:
            dataset_path = Path(self.cli_args.data_path)
            file_list = dataset_path.ls()
            uid_list = [fname[18:23] if len(fname) > 31 else fname[18:21] \
                        for fname in file_list]
            uid_set = set(uid_list)

        for uid in uid_set:
            ct = get_ct(uid, self.cli_args.data_path)
            mask_a = self.segment_ct(ct, uid)
            mask_a = mask_a.astype(np.float64)
            nifti_mask = nib.Nifti1Image(mask_a.T, affine=ct.affine)
            nib.save(nifti_mask, output_folder_path/f'{uid}.nii.gz')

if __name__=='__main__':
    CovidInferenceApp().main()




