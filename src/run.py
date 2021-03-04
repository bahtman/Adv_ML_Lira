import torch
import argparse
import logging
from dataset import TS_dataset
import os

PARSER = argparse.ArgumentParser(description='runModel.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Actions
PARSER.add_argument('--train', action='store_true', help='Train a new or restored model.')
PARSER.add_argument('--generate', action='store_true', help='Generate samples from a model.')
PARSER.add_argument('--cuda', type=int, help='Which cuda device to use')
PARSER.add_argument('--seed', type=int, default=1, help='Random seed.')

# File paths
PARSER.add_argument('--data_dir', default=None, help='Location of dataset.')
PARSER.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
PARSER.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')

# Training parameters
PARSER.add_argument('--n_labeled', type=int, default=3000, help='Number of training examples in the dataset')
PARSER.add_argument('--batch_size', type=int, default=100)
PARSER.add_argument('--time-steps', type=int, default=10, help='Size of sliding window in time series')
PARSER.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
PARSER.add_argument('--lr', type=float, default=3e-4, help='Learning rate')

import torch.nn as nn


class MockModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        logging.info('boi')


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        format="%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s",
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    ARGS = PARSER.parse_args()
    # Logging parameters for debugging purposes
    logging.info(ARGS)

    if ARGS.data_dir and not os.path.isfile(ARGS.data_dir):
        logging.error(f'{ARGS.data_dir} doesn\'t exists')
        exit(1)

    if not os.path.isdir(ARGS.output_dir):
        os.makedirs(ARGS.output_dir)

    ARGS.device = torch.device('cuda:{}'.format(ARGS.cuda)
                               if torch.cuda.is_available() and ARGS.cuda is not None
                               else 'cpu')
    if ARGS.device.type == 'cuda':
        torch.cuda.manual_seed(ARGS.seed)

    if ARGS.time_steps is None:
        dataset = TS_dataset(ARGS.data_dir, ARGS.time_steps)
    else:
        dataset = TS_dataset(ARGS.data_dir)

    model = MockModel(ARGS).to(ARGS.device)
