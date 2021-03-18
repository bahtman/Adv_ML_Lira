import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.distributions as d
import logging

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim):
        super().__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim
        )

    def forward(self, x):
        rnn_features, h = self.rnn1(x)
        rnn_features, h = self.rnn2(rnn_features)
        return rnn_features, h


class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, latent_dim, embedding_dim):
        super().__init__()
        self.hidden_dim = 2 * embedding_dim
        self.embedding_dim, self.n_features, self.latent_dim, self.seq_len =  embedding_dim, n_features, latent_dim, seq_len

        self.linear = nn.Linear(
            in_features=self.latent_dim,
            out_features= self.embedding_dim * self.seq_len
        )

        self.rnn1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.n_features
        )



    def forward(self, z):
        x = z.rsample()

        x = self.linear(x)
        x = x.reshape(self.seq_len, -1, self.embedding_dim)
        x, (_, _) = self.rnn1(x)
        x, (_, state_final) = self.rnn2(x)

        return x


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features,ARGS):
        super().__init__()
        logging.info('boi')
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = ARGS.embedding_dim
        self.latent_dim = ARGS.latent_dim

        self.encoder = Encoder(self.seq_len, self.n_features, self.embedding_dim )  # .to(device)
        self.decoder = Decoder(self.seq_len, self.n_features, self.latent_dim, self.embedding_dim)  # .to(device)
        self.mu_log_sigma = nn.Linear(self.embedding_dim * self.seq_len, 2*self.latent_dim)

    def forward(self, x):
        x, (_, current_state) = self.encoder(x)
        x_flattened = x.reshape(-1, self.embedding_dim * self.seq_len)
        mu, log_sigma = self.mu_log_sigma(x_flattened).chunk(2, dim=-1)
        z = d.Normal(mu, log_sigma.exp())

        x_hat = self.decoder(z)
        return x_hat
