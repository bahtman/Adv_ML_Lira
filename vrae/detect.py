import logging
import matplotlib.pyplot as plt
import numpy as np

def detect(model, test_dataset, device):
    anomalies = model.detect_outlier(test_dataset, amount_of_samplings=1)    

    cdict = { 1: 'red', 0: 'blue' }
    labels = test_dataset.labels
    data = test_dataset.all_data

    print(dict(zip(*np.unique(labels, return_counts=True))))
    print(dict(zip(*np.unique(anomalies, return_counts=True))))

    f, ax = plt.subplots()
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter([ix], losses[ix], c = cdict[g], label = g, s=10)

    ax.set_xlabel('mean of x')
    ax.set_ylabel('log_px')
    ax.legend()
    plt.show()


