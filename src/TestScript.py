
import numpy as np
import torch
import matplotlib.pyplot as plt

def test_function(model, test_dataset):
    Reconstructions_for_plotting = []
    test_dataset_batch = iter(test_dataset)
    fig, axs = plt.subplots(2)
    for i in range(2):
        sample = test_dataset_batch.next()
        x, y = sample
        x, y = x.float(), y.float()
        x = x.permute(1, 0, 2)
        x_recon, mu, log_var = model(x)
        axs[i].plot(x[:,i,0])
        axs[i].plot(x_recon[:,i,0])
    #axs[0, 0].plot(test_dataset)
    #axs[0, 0].plot(Reconstructions_for_plotting[0])
    #axs[0, 1].plot(test_dataset[1])
    #axs[0, 1].plot(Reconstructions_for_plotting[1])
    #axs[0, 2].plot(test_dataset[2])
    #axs[0, 2].plot(Reconstructions_for_plotting[2])
    #axs[0, 3].plot(test_dataset[3])
    #axs[0, 3].plot(Reconstructions_for_plotting[3])
    #axs[1,0].plot(test_dataseet[4])
    #axs[1,0].plot(Reconstructions_for_plotting[4])
    #axs[1,1].plot(test_dataset[5])
    #axs[1,1].plot(Reconstructions_for_plotting[5])
    #axs[1,2].plot(test_dataset[6])
    #axs[1,2].plot(Reconstructions_for_plotting[6])
    #axs[1,3].plot(test_dataset[7])
    #axs[1,3].plot(Reconstructions_for_plotting[7])
    #axs[2,0].plot(test_dataset[8])
    #axs[2,0].plot(Reconstructions_for_plotting[8])
    #axs[2,1].plot(test_dataset[9])
    #axs[2,1].plot(Reconstructions_for_plotting[9])
    #axs[2,2].plot(test_dataset[10])
    #axs[2,2].plot(Reconstructions_for_plotting[10])
    #axs[2,3].plot(test_dataset[11])
    #axs[2,3].plot(Reconstructions_for_plotting[11])

    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='y-value :)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()  
    return fig, axs