import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.distributions as d
import logging

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
        )

    def forward(self, x):
        rnn_features, h = self.rnn1(x)
        rnn_features, h = self.rnn2(rnn_features)
        return rnn_features, h


class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, input_dim=64):
        super().__init__()
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.out = nn.Linear(
            in_features=input_dim,
            out_features=2,
            bias=False
        )

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
        )



    def forward(self, z):
        z.rsample()

        x, (_, _) = self.rnn1(x)
        x, (_, state_final) = self.rnn2(x)
        x = self.out(x)

        return x


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, latent_dim=2):
        super().__init__()
        logging.info('boi')
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(seq_len, n_features, embedding_dim)  # .to(device)
        self.decoder = Decoder(seq_len, n_features, embedding_dim)  # .to(device)
        self.mu_log_sigma = nn.Linear(embedding_dim * seq_len, 2*latent_dim)

    def forward(self, x):
        x, (_, current_state) = self.encoder(x)
        x_flattened = x.reshape(-1, self.embedding_dim * self.seq_len)
        mu, log_sigma = self.mu_log_sigma(x_flattened).chunk(2, dim=-1)
        z = d.Normal(mu, log_sigma.exp())

        x_hat = self.decoder(z)
        return x_hat
