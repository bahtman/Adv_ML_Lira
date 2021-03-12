import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.distributions as d
import logging
from torch.distributions import Bernoulli
import math 
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution, Normal
from TestNetwork import *
from TrainScript import *
from src.dataset import *
import argparse
import logging
from src.dataset import TS_dataset
from src.network import *
import os


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
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
    def __init__(self, seq_len, n_features, latent_dim, embedding_dim=64):
        super().__init__()
        self.hidden_dim = 2 * embedding_dim
        self.embedding_dim, self.n_features, self.latent_dim, self.seq_len =  embedding_dim, n_features, latent_dim, seq_len

        self.linear = nn.Linear(
            in_features=self.latent_dim * 2,
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


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon() # <- your code
        
    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        dist = Normal(self.mu, self.sigma)
        return dist.log_prob(z)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features,ARGS):
        super().__init__()
        logging.info('boi')
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = ARGS.embedding_dim
        self.latent_dim = ARGS.latent_dim

        self.encoder = Encoder(self.seq_len, self.n_features, self.embedding_dim )  # .to(device)
        self.decoder = Decoder(self.seq_len, self.n_features, self.embedding_dim , self.latent_dim)  # .to(device)
        self.mu_log_sigma = nn.Linear(self.embedding_dim * self.seq_len, 2*self.latent_dim)

    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
    
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_logits = self.decoder(z)
        px_logits = px_logits.view(-1, *self.input_shape) # reshape the output
        return Bernoulli(logits=px_logits)
        

    def forward(self, x):# -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}

    def forward(self, x):
        x, (_, current_state) = self.encoder(x)
        x_flattened = x.reshape(-1, self.embedding_dim * self.seq_len)
        mu, log_sigma = self.mu_log_sigma(x_flattened).chunk(2, dim=-1)
        z = d.Normal(mu, log_sigma.exp())

        x_hat = self.decoder(z)
        return x_hat


def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta
        
    def forward(self, model:nn.Module, x:Tensor) -> Tensor:# -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px - kl
        beta_elbo = log_px - self.beta*kl
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs


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
    
    


torch.manual_seed(0)
torch.manual_seed(torch.initial_seed())
data = TS_dataset()
seq_len = len(data[0][0])#10 for vores data
n_features = len(data[0][0][0])#1 for vores data
model = RecurrentAutoencoder(seq_len, n_features,ARGS).to(ARGS.device)
#model = RecurrentAutoencoder(seq_len, n_features, 64)

vi = VariationalInference(beta=1.0)
loss, diagnostics, outputs = vi(model, data)
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")