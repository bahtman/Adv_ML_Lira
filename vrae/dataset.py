from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import pickle5 as pickle
from sklearn.preprocessing import StandardScaler


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
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.all_data[idx, :, :]
        label = self.labels[idx]
        return data, label
