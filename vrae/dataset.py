from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from random import sample


class TS_dataset(Dataset):

    def __init__(self, datafile=None, timesteps=10,columns=['acc.xyz.z'],type='train'):
        
        self.timesteps = timesteps
        if datafile:
            data = pickle.load(open(datafile, 'rb'))
            self.all_data = data['data']
            self.labels = data['labels']
        else:
            data = pickle.load(open("./Data/synth_data.pickle", 'rb'))
            self.all_data = data['data']
            self.all_data = np.expand_dims(self.all_data,2)
            self.labels = data['labels']
        anomaly_y = self.all_data[self.labels==1]
        anomaly_n = self.all_data[self.labels==0]
        label_y = self.labels[self.labels==1]
        label_n = self.labels[self.labels==0]
        indices = sample(range(anomaly_n.shape[0]),int(anomaly_n.shape[0]*0.6))
        train_data, train_label = anomaly_n[indices,:,:], label_n[indices]
        rest_data, rest_label = np.delete(anomaly_n,indices,axis=0), np.delete(label_n,indices)
        rest_data = np.concatenate((rest_data,anomaly_y))
        rest_label = np.concatenate((rest_label,label_y))

        indices = sample(range(rest_data.shape[0]),int(rest_data.shape[0]*0.5))
        val_data, val_label = rest_data[indices,:,:], rest_label[indices]
        test_data, test_label = np.delete(rest_data,indices,axis=0), np.delete(rest_label,indices)
        idx = np.random.permutation(len(test_label))
        test_data, test_label = test_data[idx,:,:], test_label[idx]
        

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
        d_array  = standscaler.fit_transform(array_data)
        d_array  = np.expand_dims(d_array , axis=2)
        # Init output arrays
        label_array  = self.data['labels'].values
        self.all_data = np.array([])
        self.labels = np.array([])
        for index in range(d_array.shape[0] - self.timesteps + 1):
            this_array = d_array[index:index + self.timesteps].reshape((-1, self.timesteps, len(self.columns)))
            timesteps_label = label_array[index:index + self.timesteps]
            if np.any(timesteps_label == 1):  # If any single observation in snippet is defect, the snippet is defect.
                this_label = 1
            else:
                this_label = 0
            if self.all_data.shape[0] == 0:
                self.all_data = this_array
                self.labels = this_label
            else:
                self.all_data = np.concatenate([self.all_data, this_array], axis=0)
                self.labels = np.append(self.labels, this_label)


    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.all_data[idx, :, :]
        label = self.labels[idx]
        return data, label
