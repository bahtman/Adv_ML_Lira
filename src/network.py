import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.distributions as d
import logging
class VAE(nn.Module):
    def __init__(self, n_features,ARGS):
        super().__init__()
        logging.info('Starting VAE')
        self.seq_len, self.n_features = ARGS.time_steps, n_features
        self.embedding_dim, self.hidden_dim = ARGS.embedding_dim, 2 * ARGS.embedding_dim
        self.latent_dim = ARGS.latent_dim


        self.p_z = d.Normal(torch.tensor(0., device=ARGS.device), torch.tensor(1., device=ARGS.device))
        
        self.mu_log_sigma = nn.Linear(2*self.embedding_dim * self.seq_len, 2*self.latent_dim)
        self.genLin = nn.Linear(in_features=self.latent_dim, out_features= self.embedding_dim * self.seq_len)
    
    def posterior(self, x):
        rnn1 = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim, bidirectional=True)
        rnn2 = nn.LSTM(input_size=self.hidden_dim*2, hidden_size=self.embedding_dim, bidirectional=True)
        rnn_features, h = rnn1(x)
        rnn_features, h = rnn2(rnn_features)
        x_flattened = rnn_features.reshape(-1, 2 * self.embedding_dim * self.seq_len)
        mu, log_sigma = self.mu_log_sigma(x_flattened).chunk(2, dim=-1)
        return d.Normal(mu, log_sigma.exp())



    def generative(self, z):
        rnn1 = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, bidirectional=True)
        rnn2 = nn.LSTM(input_size=self.hidden_dim*2, hidden_size=self.n_features)


        x = self.genLin(z)
        x = x.reshape(self.seq_len, -1, self.embedding_dim)
        x, h = rnn1(x)
        x, h = rnn2(x)
        
        x_flattened = x.reshape(self.seq_len,-1, self.n_features)

        return d.Bernoulli(logits=x_flattened)



    def forward(self, x):
        q_z_x = self.posterior(x) #Posterior

        p_z = self.p_z #Prior in batch size

        z = q_z_x.rsample() #Sample posterior with reparametrization trick

        p_x_z = self.generative(z) #Generate x_hat

        return x, z, p_z, q_z_x, p_x_z
