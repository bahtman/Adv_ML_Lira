import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from LossFunction import elbo_loss

def Reconstruct_function(model, test_dataset, train_diagnostics, val_diagnostics, ARGS):
    Reconstructions_for_plotting = []
    with torch.no_grad():
        test_dataset_batch = iter(test_dataset)
        fig, axs = plt.subplots(4)
        sample = test_dataset_batch.next()
        x, y = sample
        x, y = x.float().to(ARGS.device), y.float().to(ARGS.device)
        x = x.permute(1, 0, 2)
        x, z, p_z, q_z_x, p_x_z = model(x)
        p_x_z = p_x_z.mean
        x,p_x_z = x.cpu(), p_x_z.cpu()
        axs[0].plot(x[:,0,0], label = 'Input data')
        axs[0].plot(p_x_z[:,0,0], label = 'Reconstructed data')
        print(train_diagnostics['elbo'])
        print(train_diagnostics['log_px'].shape)
        axs[1].plot(train_diagnostics['elbo'], label = 'elbo loss for training data')
        axs[1].plot(val_diagnostics['elbo'], label = 'elbo loss for validation data')
        axs[2].plot(train_diagnostics['log_px'], label = 'log_px|z loss for training data')
        axs[2].plot(val_diagnostics['log_px'], label = 'log_px|z loss for validation data')
        axs[3].plot(train_diagnostics['kl'], label = 'kl loss for training data')
        axs[3].plot(val_diagnostics['kl'], label = 'kl loss for validation data' )
        #legend til hvert plot
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[3].legend()
        #1 legend
        #plt.legend()n

        for ax in axs.flat:
            ax.set(xlabel='time', ylabel='y-value :)')
        fig.savefig(os.path.join(ARGS.output_dir, 'loss_plots.png'))

    return fig, axs