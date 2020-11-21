import sys
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader

from modules.dsets import PrepcacheCovidDataset
from modules.util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class CovidPrepCacheApp:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=200,
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
            default=[7,60,60]
        )

        self.cli_args = parser.parse_args(sys_argv)

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

