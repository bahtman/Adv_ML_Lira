import logging
import matplotlib.pyplot as plt
import numpy as np

def detect(model, test_dataset, device):
    losses = model.detect_outlier(test_dataset)    
    print("Losses stats:")
    print('mean', np.mean(losses))
    print('std', np.std(losses))
    print('min', np.min(losses))
    print('max', np.max(losses))
    print('median', np.median(losses))

    f, ax = plt.subplots()
    cdict = { 1: 'red', -1: 'blue' }
    print(1, labels.count(1))
    print(-1, labels.count(-1))

    losses = np.array(losses)
    x_list = np.array(x_list)
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(x_list[ix], losses[ix], c = cdict[g], label = g, s=10)

    ax.set_xlabel('mean of x')
    ax.set_ylabel('log_px')
    ax.legend()
    plt.show()


