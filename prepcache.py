import sys
import argparse
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from modules.dsets import PrepcacheCovidDataset
from modules.util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

Path.ls = lambda x: [o.name for o in x.iterdir()]

class CovidPrepCacheApp:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=20,
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
        parser.add_argument('--data-path',
            help="Path to the data to train",
            nargs='?',
            required=True
        )

        self.cli_args = parser.parse_args(sys_argv)
        
        if self.cli_args.data_path is not None:
            data_path = Path(self.cli_args.data_path)
            ct_list = data_path.ls()
            file_names = [ct[:23] if len(ct) > 31 else ct[:21] for ct in ct_list if 'seg' not in ct]
            uid_list = [ct[18:23] if len(ct) > 22 else ct[18:21] for ct in file_names]
            fnames = pd.Series(file_names)
            ct_fnames = list((data_path/fnames).astype(str) + '_ct.nii.gz')
            mask_fnames = list((data_path/fnames).astype(str) + '_seg.nii.gz')
            assert len(uid_list) == len(ct_fnames) == len(mask_fnames), repr([len(uid_list), len(ct_fnames), len(mask_fnames)])
            df_meta = pd.DataFrame({'uid': uid_list,
                                    'ct_fname': ct_fnames,
                                    'mask_fname': mask_fnames})
            meta_path = Path(f'metadata/')
            meta_path.mkdir(parents=True, exist_ok=True)
            df_meta.to_feather(meta_path/'df_meta.fth')
            log.info("Creating metadata folder")
            with pd.option_context('display.max_rows', 10):
                display(df_meta)
            with open(".env", "w") as file:
                file.write(f'datasets_path="{self.cli_args.data_path}"')
                log.info("Creating data_path environment variable in new .env file")

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        width_irc = tuple([int(axis) for axis in self.cli_args.width_irc])
        ds = PrepcacheCovidDataset(width_irc=width_irc)
        self.prep_dl = DataLoader(
            ds,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers
        )

        for _ in tqdm(enumerate(self.prep_dl), total=len(self.prep_dl)):
            pass


if __name__ == '__main__':
    CovidPrepCacheApp().main()

