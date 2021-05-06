import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, auc

def detect(model, test_dataset, val_dataset, device):
    losses, _ = model.detect_outlier(val_dataset, amount_of_samplings=16)    
    fpr, tpr, thresholds_val = roc_curve(val_dataset.labels, losses)
    roc_auc = auc(fpr, tpr)
    j_score = tpr - fpr 
    threshold = thresholds_val[np.argmax(j_score)]
    _, anomalies = model.detect_outlier(test_dataset, amount_of_samplings=16, threshhold=threshold)    

    labels = test_dataset.labels
    accuracy = accuracy_score(labels, anomalies)
    print(threshold)
    print(accuracy*100)