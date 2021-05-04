
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def Reconstruct_function(model, test_dataset, n_plots,ARGS):
    Reconstructions_for_plotting = []
    with torch.no_grad():
        test_dataset_batch = iter(test_dataset)
        fig, axs = plt.subplots(n_plots)
        for i in range(n_plots):
            sample = test_dataset_batch.next()
            x, y = sample
            x, y = x.float().to(ARGS.device), y.float().to(ARGS.device)
            x = x.permute(1, 0, 2)
            x, z, p_z, q_z_x, p_x_z = model(x)
            p_x_z = p_x_z#.mean
            x,p_x_z = x.cpu(), p_x_z.cpu()
            axs[i].plot(x[:,0,0], label = 'Input data')
            axs[i].plot(p_x_z[:,0,0], label = 'Reconstructed data')
            #legend til hvert plot
            axs[i].legend()
            #1 legend
            #plt.legend()

        for ax in axs.flat:
            ax.set(xlabel='time', ylabel='y-value :)')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
#        for ax in axs.flat:
#            ax.label_outer() 
        fig.savefig(os.path.join(ARGS.output_dir, 'generated_samples.png'))
    return fig, axs