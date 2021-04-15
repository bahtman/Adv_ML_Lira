import torch
import argparse
import logging
from dataset import TS_dataset

from TrainScript import train_model
from torch.utils.data import DataLoader, random_split

import os

PARSER = argparse.ArgumentParser(description='runModel.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Actions
PARSER.add_argument('--train', action='store_true', help='Train a new or restored model.')
PARSER.add_argument('--generate', action='store_true', help='Generate samples from a model.')
PARSER.add_argument('--cuda', type=int, help='Which cuda device to use')
PARSER.add_argument('--seed', type=int, default=1, help='Random seed.')
PARSER.add_argument('--model', type=int, default=1, help='Choose model for training if "1" the model will be a VAE, if "2" the model will be a normal LSTM.')

# File paths
PARSER.add_argument('--data_dir', default='./Data/gm_data.pickle', help='Location of dataset.')
PARSER.add_argument('--output_dir', default='./results/')
PARSER.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
PARSER.add_argument('--dataset', default='generated', help='Which dataset to use. [generated|GM]')

# Training parameters
PARSER.add_argument('--n_labeled', type=int, default=3000, help='Number of training examples in the dataset')
PARSER.add_argument('--batch_size', type=int, default=32)
PARSER.add_argument('--time-steps', type=int, default=10, help='Size of sliding window in time series')
PARSER.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
PARSER.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
PARSER.add_argument('--latent_dim', type=int, default=2, help='Latent dim')
PARSER.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
PARSER.add_argument('--amount_of_plots', type = int, default = 6, help = 'The amount of inputs sequences and their respective reconstructions to be plotted')

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

    ARGS.device = torch.device('cuda'
                               if torch.cuda.is_available()
                               else 'cpu')
    if ARGS.device.type == 'cuda':
        torch.cuda.manual_seed(ARGS.seed)
    else: 
        torch.manual_seed(ARGS.seed)
    if ARGS.dataset == 'generated':
        logging.info('Using generated dataset')
        dataset = TS_dataset(timesteps=ARGS.time_steps)
        n_features = 1
        seq_len = ARGS.time_steps
    elif ARGS.dataset == 'GM':
        logging.info('Using GreenMobility dataset')
        columns= ['acc.xyz.z']
        dataset = TS_dataset(ARGS.data_dir,ARGS.time_steps,columns=columns, logger=logging)
        n_features= len(columns)
        seq_len = ARGS.time_steps
    else:
        raise Exception(f"{ARGS.dataset} is not defined")

    if ARGS.model == 1:
        from network import VAE
        model = VAE(n_features, ARGS).to(ARGS.device)
        logging.info("A VAE model type structure will be used for training")
    elif ARGS.model == 2:
        from NormalLSTM import RecurrentAutoencoderLSTM
        model = RecurrentAutoencoderLSTM(seq_len, n_features, ARGS.embedding_dim, ARGS.latent_dim).to(ARGS.device)
        logging.info("A normal LSTM model type structure will be used for training")
 
    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    
    # make datasets iterable
    n_val = int(len(val)*0.5)
    n_test = int(len(val)-n_val)
    val, test = random_split(val, [n_val, n_test])

    train_loader = DataLoader(train, batch_size=ARGS.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val, batch_size=ARGS.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    if ARGS.train:
        trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        ARGS
        )

    if ARGS.generate:
        from ReconstructionPlotScript import Reconstruct_function
        fig, axs = Reconstruct_function(trained_model, test_loader, ARGS.amount_of_plots, ARGS)
