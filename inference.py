import sys
import torch
import argparse
import numpy as np

from scipy import ndimage

from torch import nn
from torch.utils.data import DataLoader

# local imports
from modules.model import UNetWrapper
from modules.dsets import Covid2dSegmentationDataset 

class CovidInferenceApp:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
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

        self.cli_args = parser.parse_args(sys_argv)
        if not (bool(self.cli_args.uid) ^ self.cli_args.run_all):
            raise Exception("One and only one of uid and --run-all should be given")
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.window = None

        if not self.cli_args.model_path:
            raise Exception("Path to segmentation model should be given")

        self.model = self.init_model()

    def init_model(self):
        log.debug(self.cli_args.model_path)
        model_dict = torch.load(self.cli_args.model_path)

        self.window = model_dict['window']
        
        model = UNetWrapper(
            in_channels=model_dict['in_channels'],
            n_classes=1,
            depth=model_dict['depth'],
            wf=4,
            padding=True,
            pad_type=model_dict['pad_type'],
            batch_norm=True,
            up_mode='upconv')

        model.load_state_dict(model_dict['model_state'])
        model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(self.device)

        return model

    def init_dl(self, uid):
        ds = Covid2dSegmentationDataset(
            uid=uid,
            window=self.window, 
            is_full_ct=True)
        dl = DataLoader(
            ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() \
                                                   if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda)
        return dl

    def segment_ct(self, ct, uid):
        with torch.no_grad():
            output = np.zeros_like(ct.hu, dtype=np.float32)
            dl = self.init_dl(uid)
            for hu, _, _, slice_idx_list in dl:
                hu = hu.to(self.device)
                pred = self.model(hu)

                for i, slice_idx in enumerate(slice_idx_list):
                    output[slice_idx] = pred[i].cpu().numpy()

            mask = output > .5
            eroded_mask = ndimage.binary_erosion(mask)

        return mask, eroded_mask

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        if self.cli_args.uid:
            uid_set = set(self.cli_args.uid.split(','))
        for uid in uid_set:
            ct_paths = sorted(glob.glob(f'{str(dataset_path)}/*/*-0{uid}_*.nii.gz'))


