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
        self.n_layers = ARGS.n_layers
        self.batch_size = ARGS.batch_size
        self.device = ARGS.device
        self.p_z = d.Normal(torch.tensor(0., device=ARGS.device), torch.tensor(1., device=ARGS.device))
        if ARGS.bidir:
            self.mu_log_sigma = nn.Linear(2*self.embedding_dim, 2*self.latent_dim)
            self.genLin = nn.Linear(in_features=self.latent_dim, out_features= self.embedding_dim)
            self.rnn1 = nn.LSTM(input_size=self.n_features, hidden_size=self.embedding_dim,num_layers=self.n_layers, bidirectional=True)
            self.rnn3 = nn.LSTM(input_size=2*self.embedding_dim, hidden_size=self.n_features,num_layers=self.n_layers, bidirectional=True)
            
        else:
            self.mu_log_sigma = nn.Linear(self.embedding_dim, 2*self.latent_dim)
            self.genLin = nn.Linear(in_features=self.latent_dim, out_features= self.embedding_dim)
            self.rnn1 = nn.LSTM(input_size=self.n_features, hidden_size=self.embedding_dim,num_layers=self.n_layers, bidirectional=False)
            self.rnn3 = nn.LSTM(input_size=self.n_features, hidden_size=self.embedding_dim,num_layers=self.n_layers, bidirectional=False)

        self.hidden_to_output = nn.Linear(self.embedding_dim, self.n_features)
        self.decoder_inputs = torch.zeros(self.seq_len, self.batch_size, self.n_features, requires_grad=True, device=ARGS.device)
        self.c0 = torch.zeros(self.n_layers, self.batch_size, self.embedding_dim, requires_grad=True, device=ARGS.device)
    

    
    def posterior(self, x):

        rnn_features, (h,c) = self.rnn1(x)
        h = h[-1,:,:]
        x_flattened = h.reshape(-1,self.embedding_dim)
        mu, log_sigma = self.mu_log_sigma(x_flattened).chunk(2, dim=-1)
        return d.Normal(mu, log_sigma.exp())



    def generative(self, z):
        x = self.genLin(z)
        h_0 = torch.stack([x for _ in range(self.n_layers)])
        decoder_inputs = torch.zeros(self.seq_len, h_0.shape[1], self.n_features, requires_grad=True, device=self.device)
        c0 = torch.zeros(self.n_layers, h_0.shape[1], self.embedding_dim, requires_grad=True, device=self.device)
        decoder_output, _ = self.rnn3(decoder_inputs, (h_0, c0))
        x = self.hidden_to_output(decoder_output)
        x_flattened = x.reshape(self.seq_len,-1, self.n_features)
        
        return x_flattened#d.Bernoulli(logits=x_flattened)



    def forward(self, x):
        q_z_x = self.posterior(x) #Posterior

        p_z = self.p_z #Prior in batch size

        z = q_z_x.rsample() #Sample posterior with reparametrization trick

        p_x_z = self.generative(z) #Generate x_hat

        return x, z, p_z, q_z_x, p_x_z
