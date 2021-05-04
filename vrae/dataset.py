from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import pickle5 as pickle
from sklearn.preprocessing import StandardScaler
from random import sample


class TS_dataset(Dataset):

    def __init__(self, datafile=None, timesteps=10,columns=['acc.xyz.z'],type='train'):
        
        self.timesteps = timesteps
        if datafile:
            data = pickle.load(open(datafile, 'rb'))
            data['labels'] = data['IRI_mean'].apply(lambda x: 1 if x <= 2 else -1)
            #data = data[data.labels==1]
            self.columns = columns
            self.data = data
            self.process_gm_re()
        else:
            data = pickle.load(open("./Data/synth_data.pickle", 'rb'))
            self.all_data = data['data']
            self.labels = data['labels']
        anomaly_y = self.all_data[self.labels==-1]
        anomaly_n = self.all_data[self.labels==1]
        label_y = self.labels[self.labels==-1]
        label_n = self.labels[self.labels==1]
        indices = sample(range(anomaly_n.shape[0]),int(anomaly_n.shape[0]*0.8))
        train_data, train_label = anomaly_n[indices,:,:], label_n[indices]
        rest_data, rest_label = np.delete(anomaly_n,indices,axis=0), np.delete(label_n,indices)
        rest_data = np.concatenate((rest_data,anomaly_y))
        rest_label = np.concatenate((rest_label,label_y))

        indices = sample(range(rest_data.shape[0]),int(rest_data.shape[0]*0.5))
        val_data, val_label = rest_data[indices,:,:], rest_label[indices]
        test_data, test_label = np.delete(rest_data,indices,axis=0), np.delete(rest_label,indices)

        if type=='train':
            self.all_data = train_data
            self.labels = train_label
        elif type=='val':
            self.all_data = val_data
            self.labels = val_label
        elif type == 'test':
            self.all_data = test_data
            self.labels = test_label


    def process_gm_re(self):
        array_data = np.vstack(self.data.iloc[:,1].values)
        standscaler = StandardScaler()
        self.all_data = standscaler.fit_transform(array_data)
        self.all_data = np.expand_dims(self.all_data, axis=2)
        self.labels = self.data['labels'].values
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.all_data[idx, :, :]
        label = self.labels[idx]

        return data, label
