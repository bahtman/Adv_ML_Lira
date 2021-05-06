from vrae.vrae import VRAE
from vrae.utils import *
from vrae.SplitDataSet import Split_Function
import numpy as np
import torch
from vrae.dataset import *
import wandb
from dotmap import DotMap
import plotly
from torch.utils.data import DataLoader, random_split


columns= ['acc.xyz.z']
seq_len = 369
train = TS_dataset(timesteps=seq_len,columns=columns,type='train')
val = TS_dataset(timesteps=seq_len,columns=columns,type='val')
test = TS_dataset(timesteps=seq_len,columns=columns,type='test')
n_features= len(columns)


hyperparameter_defaults = dict(
        hidden_size = 115, 
        hidden_layer_depth = 2,
        latent_length = 75,
        batch_size = 32,
        learning_rate = 0.0001649,
        n_epochs = 15,
        dropout_rate = 0.7426,
        max_grad_norm=31
        )
config = DotMap(hyperparameter_defaults)


wandb.init(config = hyperparameter_defaults,project="VRAE")
config = wandb.config
args = DotMap(dict(
    seq_len=seq_len,
    n_features = n_features,
    batch_size = config.batch_size,
    n_epochs = config.n_epochs,
    optimizer = 'Adam',
    clip = True,
    loss = 'MSELoss',
    block = 'LSTM',    
    datafile = f'{os.path.dirname(os.path.abspath(__file__))}/data',
    seed = 42,
    results_file = 'result.txt',
    output_dir = 'results',
    visualize=False,
    
    train = True,
    detectOutliers = False,
    prior='vampprior'
))
args.device = True if torch.cuda.is_available() else False
torch.manual_seed(args.seed)
#os.mkdir(args.output_dir)
if args.device == True : torch.cuda.manual_seed(args.seed)

vrae = VRAE(sequence_length=args.seq_len,
            number_of_features = args.n_features,
            hidden_size = config.hidden_size, 
            hidden_layer_depth = config.hidden_layer_depth,
            latent_length = config.latent_length,
            batch_size = args.batch_size,
            learning_rate = config.learning_rate,
            n_epochs = args.n_epochs,
            dropout_rate = config.dropout_rate,
            optimizer = args.optimizer, 
            cuda = args.device,
            clip=args.clip, 
            max_grad_norm=config.max_grad_norm,
            loss = args.loss,
            block = args.block,
            plot_loss = args.visualize,
            prior = args.prior)

if args.train:
    print('Training VRAE model')
    vrae.fit(train,val, save=True)
elif args.prior == 'standard':
    vrae.load('vrae/models/model_gaus.pth')
elif args.prior == 'vampprior':
    vrae.load('vrae/models/model_vamp.pth')

if args.detectOutliers: 
    print("Detecting outliers")
    from vrae.detect import detect
    detect(vrae, test, val, device=args.device)

if args.visualize:
    print("Visualizing validation set with VRAE")
    x_decoded = vrae.reconstruct(test)

    with torch.no_grad():  
        n_plots = 4
        x,label = val[0:n_plots]
        fig, axs = plt.subplots(n_plots, figsize = (15,15))
        for i in range(n_plots):
            axs[i].plot(x[i,:,0], label = 'Input data')
            axs[i].plot(x_decoded[:,i,0], label = 'Reconstructed data')
            axs[i].title.set_text(f'Label is {label[i]}')


        for ax in axs.flat:
            ax.set(xlabel='', ylabel='')
        wandb.log({f"Reconstructions":fig})
        fig.savefig('generated_samples.png')
