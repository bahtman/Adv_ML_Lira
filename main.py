from vrae.vrae import VRAE
from vrae.utils import *
import numpy as np
import torch
from vrae.dataset import *

import plotly
from torch.utils.data import DataLoader, random_split


hidden_size = 90
hidden_layer_depth = 2
latent_length = 20
batch_size = 32
learning_rate = 0.0005
n_epochs = 50
dropout_rate = 0.2
optimizer = 'Adam' # options: ADAM, SGD
cuda = True if torch.cuda.is_available() else False # options: True, False
#cuda = False
print_every=30
clip = True # options: True, False
max_grad_norm=5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'LSTM' # options: LSTM, GRU


columns= ['acc.xyz.z']
seq_len = 369
dataset = TS_dataset(timesteps=seq_len,columns=columns)
n_features= len(columns)

val_percent = 0.1
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])


vrae = VRAE(sequence_length=seq_len,
            number_of_features = n_features,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer, 
            cuda = cuda,
            print_every=print_every, 
            clip=clip, 
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block)

vrae.fit(train)
x_decoded = vrae.reconstruct(val)


with torch.no_grad():  
    n_plots = 5
    x,label = val[0:n_plots]
    print(x.shape)  
    print(x_decoded.shape)
    fig, axs = plt.subplots(n_plots, figsize = (15,15))
    for i in range(n_plots):
        axs[i].plot(x[i,:,0], label = 'Input data')
        axs[i].plot(x_decoded[:,i,0], label = 'Reconstructed data')
        axs[i].legend()


    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='y-value :)')

    fig.savefig('generated_samples.png')