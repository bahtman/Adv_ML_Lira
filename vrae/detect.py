import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def detect(model, test_dataset, device):
    anomalies = model.detect_outlier(test_dataset, amount_of_samplings=10)    

    cdict = { 1: 'red', 0: 'blue' }
    labels = test_dataset.labels
    data = test_dataset.all_data

    accuracy = accuracy_score(labels, anomalies)
    print(accuracy)
    print(dict(zip(*np.unique(labels, return_counts=True))))
    print(dict(zip(*np.unique(anomalies, return_counts=True))))




