"""
EDIT NOTICE

File edited from original in https://github.com/castorini/hedwig
by Bernal Jimenez Gutierrez (jimenezgutierrez.1@osu.edu)
in May 2020
"""

import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="PyTorch deep learning models for document classification")

    parser.add_argument('--no-cuda', action='store_false', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--data-dir', default=os.path.join(os.pardir, 'hedwig-data', 'datasets'))
    parser.add_argument('--training_file',default='train.tsv')
    parser.add_argument('--dev_file',default='dev.tsv')
    parser.add_argument('--test_file',default='test.tsv')

    return parser
