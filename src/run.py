import torch
import argparse
import logging
from dataset import TS_dataset
from network import RecurrentAutoencoder
from TrainScript import train_model
from torch.utils.data import DataLoader, random_split

import os

PARSER = argparse.ArgumentParser(description='runModel.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Actions
PARSER.add_argument('--train', action='store_true', help='Train a new or restored model.')
PARSER.add_argument('--generate', action='store_true', help='Generate samples from a model.')
PARSER.add_argument('--cuda', type=int, help='Which cuda device to use')
PARSER.add_argument('--seed', type=int, default=1, help='Random seed.')

# File paths
PARSER.add_argument('--data_dir', default=None, help='Location of dataset.')
PARSER.add_argument('--output_dir', default='./results/')
PARSER.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')

# Training parameters
PARSER.add_argument('--n_labeled', type=int, default=3000, help='Number of training examples in the dataset')
PARSER.add_argument('--batch_size', type=int, default=100)
PARSER.add_argument('--time-steps', type=int, default=10, help='Size of sliding window in time series')
PARSER.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
PARSER.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
PARSER.add_argument('--latent_dim', type=int, default=2, help='Learning rate')
PARSER.add_argument('--embedding_dim', type=int, default=64, help='Learning rate')

import torch.nn as nn



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
        dataset = TS_dataset(ARGS.data_dir)
    else:
        dataset = TS_dataset(ARGS.data_dir, ARGS.time_steps)
    #Have to be based upon dataset
    n_features = 1 
    seq_len = 10


    model = RecurrentAutoencoder(seq_len, n_features,ARGS).to(ARGS.device)
    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)
    n_train = int(len(dataset) - n_val)
    train, val = random_split(dataset, [n_train, n_val])
    
    # make datasets iterable
    n_val = int(len(val)*0.5)
    n_test = int(len(val)-n_val)
    val, test = random_split(val, [n_val, n_test])
    train_loader = DataLoader(train, batch_size=ARGS.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val, batch_size=ARGS.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test, batch_size=ARGS.batch_size, shuffle=False, num_workers=0, drop_last=False)

    if ARGS.train:
        trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=ARGS.n_epochs
        )
