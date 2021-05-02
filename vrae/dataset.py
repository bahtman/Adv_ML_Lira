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
            #data = data[(data['IRI']<=2) | (data['IRI']>=4)]
            data['labels'] = data['IRI_mean'].apply(lambda x: 1 if x <= 2 else -1)
            data = data[data.labels==1]
            #self.columns = columns
            self.data = data
            self.process_gm_re()
            #self.process_gm()
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
    def process_gm(self):
        self.all_data = np.array([])
        self.labels = np.array([])
        self.data.apply(self.gm_apply,axis=1)
            
    def gm_apply(self,row):
        label = row['labels']
        row = row[self.columns]
        data_np = np.zeros((row.iloc[0].shape[0],row.shape[0]))

        for i,col in enumerate(row):
            data_np[:,i] = col

        
        data_np = self.standscaler.fit_transform(data_np)
        data_np = self.mscaler.fit_transform(data_np)
        for index in range(data_np.shape[0] - self.timesteps + 1):
                this_array = data_np[index:index + self.timesteps].reshape((-1, self.timesteps, len(self.columns)))
                
                if self.all_data.shape[0] == 0:
                    self.all_data = this_array
                    self.labels = label
                else:
                    self.all_data = np.concatenate([self.all_data, this_array], axis=0)
                    self.labels = np.append(self.labels, label)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.all_data[idx, :, :]
        label = self.labels[idx]

        return data, label
