from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import pickle5 as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TS_dataset(Dataset):

    def __init__(self, datafile=None, timesteps=10,columns=['acc.xyz.z']):
        
        self.timesteps = timesteps
        if datafile:
            data = pickle.load(open(datafile, 'rb'))
            data['labels'] = data['IRI_mean'].apply(lambda x: 1 if x <= 2 else -1)
            data = data[data.labels==1]
            #self.columns = columns
            self.data = data
            self.process_gm_re()
        else:
            data = pickle.load(open("./Data/synth_data.pickle", 'rb'))
            self.all_data = data['data']
            self.labels = data['labels']
    def process_gm_re(self):
        array_data = np.vstack(self.data.iloc[:,1].values)
        standscaler = StandardScaler()
        self.all_data = standscaler.fit_transform(array_data)
        self.all_data = np.expand_dims(self.all_data, axis=2)
        self.labels = self.data['labels'].values
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        test_percent = 0.1
        val_percent = 0.2

        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.all_data[idx, :, :]
        label = self.labels[idx]
        train, test_val, train_label, test_val_label = train_test_split(data, label, test_size = 0.3, random_state=42)
        val, test, val_label, test_label = train_test_split(test_val, test_val_label, test_size = 0.33, random_state = 42)
        train = train[train_label == 1]
        train_label = train_label[train_label == 1]
        return train, train_label, val, val_label, test, test_label
