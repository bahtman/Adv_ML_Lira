import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from .base import BaseEstimator
from torch.utils.data import DataLoader
import torch.utils.data
from torch.nn import Linear
from torch.autograd import Variable
from utils.distributions import log_Normal_diag, log_Normal_standard
import math
import os
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean#, self.latent_logvar

class Decoder(nn.Module):
    """Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM', prior = "vampprior"):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class VRAE(BaseEstimator, nn.Module):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries
    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss / any custom loss which inherits from `_Loss` class
    :param boolean cuda: to be run on GPU or not
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param dload: Download directory where models are to be dumped
    :param plot_loss: Wether to plot losses during training
    """
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM',
                 n_epochs=5, dropout_rate=0., optimizer='Adam', loss='MSELoss',
                 cuda=False, clip=True, max_grad_norm=5, dload='.',plot_loss=True, prior = 'vampprior'):

        super(VRAE, self).__init__()

        self.plot_loss = plot_loss
        self.dtype = torch.FloatTensor
        self.use_cuda = cuda

        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False


        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor


        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype)
        #self.means = nn.Linear(sequence_length, hidden_size)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.prior = prior

        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = dload

        if self.use_cuda:
            self.cuda()

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')

        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=False)
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=False)

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder
        :param x:input tensor
        :return: the decoded output, latent vector
        """
        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded = self.decoder(latent)

        return x_decoded, latent

    def _rec(self, x_decoded, x, loss_fn):
        """
        Compute the loss given output x decoded, input x and the specified loss function
        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar
        # KL
        z_q = self.reparameterize(latent_mean, latent_logvar)
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, latent_mean, latent_logvar, dim=1)
        kl_loss = -(log_p_z - log_q_z)
        #log_q_z = log_Normal_diag(z_q,z_q_mean, z_q_logvar, dim=1)
        #kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)

        return kl_loss + recon_loss, recon_loss, kl_loss
    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        x = self.q_z_layers(x)

        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar

    def log_p_z(self, z):
        if self.prior == 'standard':
            log_prior = log_Normal_standard(z, dim=1)
        elif self.prior == 'vampprior':
            # z - MB x M
            C = self.hidden_size

            # calculate params
            #X = self.means(self.idle_input)

            # calculate params for given data
            latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

            # expand z
            z_expand = z.unsqueeze(1)
            means = latent_mean.unsqueeze(0)
            logvars = latent_logvar.unsqueeze(0)

            a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
            a_max, _ = torch.max(a, 1)  # MB x 1

            # calculte log-sum-exp
            log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1

        else:
            raise Exception('Wrong name of the prior!')

        return log_prior
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.use_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def compute_loss(self, X):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration
        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)

        x_decoded, _ = self(x)
        loss, recon_loss, kl_loss = self._rec(x_decoded, x.detach(), self.loss_fn)
        loss, recon_loss, kl_loss = loss.mean(), recon_loss.mean(), kl_loss.mean()
        return loss, recon_loss, kl_loss, x


    def _train(self, train_loader, val_loader,epoch):
        """
        For each epoch, given the batch_size, run this function batch_size * num_of_batches number of times
        :param train_loader:input train loader with shuffle
        :return:
        """
        self.train()

        t = 0
        losses, recon_losses, kl_losses = [],[],[]
        #total_norm= []
        with tqdm(total=len(train_loader), desc='epoch {} of {}'.format(epoch+1, self.n_epochs)) as pbar:
            for t, X in enumerate(train_loader):

                # Index first element of array to return tensor
                X = X[0]

                # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
                X = X.permute(1,0,2)

                self.optimizer.zero_grad()
                loss, recon_loss, kl_loss, _ = self.compute_loss(X)
                loss.backward()
                losses.append(loss.cpu().detach().numpy())
                recon_losses.append(recon_loss.cpu().detach().numpy())
                kl_losses.append(kl_loss.cpu().detach().numpy())
            
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)
                '''
                for p in self.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm.append(param_norm.item())
                '''


                self.optimizer.step()

                #if (t + 1) % 20 == 0:
                #    print('Batch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' % (t + 1, np.mean(losses),
                #                                                                        np.mean(recon_losses), np.mean(kl_losses)))
                pbar.set_postfix(loss='{:.3f}'.format(np.mean(losses)))
                pbar.update()

        #print("Average grad norm: ",np.mean(total_norm))
        print('Average loss: {:.4f}'.format(np.mean(losses)))
        self.eval()

        t = 0
        val_losses = []
        for t, X in enumerate(val_loader):

            # Index first element of array to return tensor
            X = X[0]

            # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
            X = X.permute(1,0,2)

            loss, recon_loss, kl_loss, _ = self.compute_loss(X)
            val_losses.append(loss.cpu().detach().numpy())


        print('Average loss: {:.4f}'.format(np.mean(val_losses)))
        return np.mean(losses),np.mean(recon_losses), np.mean(kl_losses), np.mean(val_losses)


    def fit(self, train_dataset,val_dataset, save = False):
        """
        Calls `_train` function over a fixed number of epochs, specified by `n_epochs`
        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """

        train_loader = DataLoader(dataset = train_dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=True)
        val_loader = DataLoader(dataset = val_dataset,
                                    batch_size = self.batch_size,
                                    shuffle = False,
                                    drop_last=True)
        losses, recon_losses, kl_losses, val_losses = [],[],[], []
        for i in range(self.n_epochs):
            print('Epoch: %s' % i)

            loss, recon, kl, val_loss = self._train(train_loader,val_loader,i)
            losses.append(loss)
            recon_losses.append(recon)
            kl_losses.append(kl)
            val_losses.append(val_loss)
            y_pred, _ = self.detect_outlier(val_dataset)
            y_true = val_dataset.labels
            fpr, tpr, thresholds = roc_curve(y_true,y_pred)
            auc_ = auc(fpr,tpr)
            wandb.log({
            "train_loss":loss,
            "train_recon": recon,
            "train_kl": kl,
            "val_loss":val_loss,
            "auc":auc_
            })

        self.is_fitted = True
        if self.plot_loss:
            with torch.no_grad():
                fig, axs = plt.subplots(3,figsize=(15,15))
                axs[0].plot(losses, label = 'Elbo loss for train')
                axs[0].plot(val_losses, label = 'Elbo loss for val')
                axs[1].plot(recon_losses, label = 'Reconstruction loss')
                axs[2].plot(kl_losses, label = 'Kl divergence')

                axs[0].legend()
                axs[1].legend()
                axs[2].legend()
                for ax in axs.flat:
                    ax.set(xlabel='epoch', ylabel='loss')

                fig.savefig('loss_plots.png')

        if save:
            self.save('model.pth')


    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function
        :param x: input batch tensor
        :return: intermediate latent vector
        """
        return self.lmbd(
                    self.encoder(
                        Variable(x.type(self.dtype), requires_grad = False)
                    )
        ).cpu().data.numpy()

    def _batch_reconstruct(self, x, tensor=False):
        """
        Passes the given input tensor into encoder, lambda and decoder function
        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)
        if tensor:
            return x_decoded
        return x_decoded.cpu().data.numpy()

    def reconstruct(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit
        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = True,
                                 drop_last=True) # Don't shuffle for test_loader

        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    x_decoded_each = self._batch_reconstruct(x)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded.dump(self.dload + '/z_run.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')

    def detect_outlier(self, dataset, amount_of_samplings=1, threshhold = 1500):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit
        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader

        with torch.no_grad():
            tmp = np.zeros(len(dataset))
            for i, x in enumerate(test_loader):
                x = x[0]
                x = x.permute(1, 0, 2).type(self.dtype)
                _x = Variable(x, requires_grad = False)
                # Run the batch through the encoder and decoder. 
                # latent_mean and latent_logvar comes from latent space from encoder, after being run through the 
                # Lambda class. 
                x_recon, latent = self(_x)
                
                for l in range(amount_of_samplings):
                    # Draw batch_size*L samples from z ~ N(mu_z, sigma_z)
                    std = torch.exp(0.5 * self.lmbd.latent_logvar)
                    latent_space_samples = torch.normal(self.lmbd.latent_mean, std)
                    x_recon_batch = self.decoder(latent_space_samples)
                    for j in range(self.batch_size):
                        x_single = x[:,j,:]
                        x_recon_single = x_recon_batch[:,j,:]
                        # Measure loss between reconstruction and sample and call this "reconstruction probability"
                        tmp[i*self.batch_size+j] += self.loss_fn(x_recon_single, x_single)
            
            tmp /= amount_of_samplings
            # Marks the sample as an outlier if reconstruction probability > \alpha
            indices_outlier = np.where(dataset.labels == 1)[0]
            indices = np.where(dataset.labels == 0)[0]
            asd = np.array(range(len(dataset)))
            plt.scatter(asd[indices_outlier], tmp[indices_outlier], label='1')
            plt.scatter(asd[indices], tmp[indices], label='0')
            plt.legend()
            plt.savefig('anomalies.png')
            plt.show()
            plt.close()
            anomalies = [1 if x > threshhold else 0 for x in tmp]
        return tmp, anomalies
    def transform(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit
        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                z_run = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    z_run_each = self._batch_transform(x)
                    z_run.append(z_run_each)

                z_run = np.concatenate(z_run, axis=0)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False):
        """
        Combines the `fit` and `transform` functions above
        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle file
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        :return: None
        """
        PATH = self.dload + '/' + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        if self.use_cuda:
            self.load_state_dict(torch.load(PATH))
        else:
            self.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))