from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TS_dataset(Dataset):

    def __init__(self, datafile=None, timesteps=10, columns=['acc.xyz.z']):
        
        self.timesteps = timesteps
        if datafile:
            data = pickle.load(open(datafile, 'rb'))
            data = data[(data['IRI']<=2) | (data['IRI']>=4)]
            data['labels'] = data['IRI'].apply(lambda x: 1 if x <= 2 else -1)
            self.data_test = data
            self.data = data[data.labels==1]
            self.columns = columns
            self.process_gm()
        else:
            # Synthetic data
            import symengine
            import timesynth as ts
            # Initialize samplers
            time_sampler = ts.TimeSampler(stop_time=20)

            # Regular signal
            regular_time_samples = time_sampler.sample_regular_time(num_points=1000)
            reg_signalgen = ts.signals.Sinusoidal(frequency=0.25)
            noise = ts.noise.GaussianNoise(std=0.1)
            reg_timeseries = ts.TimeSeries(reg_signalgen, noise_generator=noise)

            # Defect signal
            irregular_time_samples = time_sampler.sample_irregular_time(num_points=2000, keep_percentage=50)
            irreg_signalgen = ts.signals.ar.AutoRegressive(ar_param=[0.9], sigma=0.5, start_value=[0.0])
            irreg_timeseries = ts.TimeSeries(irreg_signalgen, noise_generator=noise)

            # Generate regular
            samples, signals, errors = reg_timeseries.sample(regular_time_samples)
            samples -= 0.0
            samples[samples < 0] = 0

            # Generate defect
            defect_samples, defect_signals, defect_errors = irreg_timeseries.sample(irregular_time_samples)
            defect_samples -= 2.0
            defect_samples[defect_samples < 0] = 0

            # save as Dataframe
            data = pd.DataFrame(
                {'samples': samples + defect_samples, 'labels': [1 if x == 0 else -1 for x in defect_samples]})
            data = data[data.labels == 1]

        # Assume data column is always 'samples'
            self.columns = ['samples']
            self.data = data
            self.process_synth()
        
    def process_gm(self):
        self.all_data = np.array([])
        self.labels = np.array([])
        self.standscaler = StandardScaler()
        self.mscaler = MinMaxScaler(feature_range=(0, 1))
        
        self.data.apply(self.gm_apply,axis=1)
            
    def process_synth(self):

        # Normalization
        standscaler = StandardScaler()
        mscaler = MinMaxScaler(feature_range=(0, 1))
        self.data[self.columns] = standscaler.fit_transform(self.data[self.columns])
        self.data[self.columns] = mscaler.fit_transform(self.data[self.columns])

        # Init output arrays
        self.all_data = np.array([])
        self.labels = np.array([])

        # Extract data from dataframe
        d_array = self.data[self.columns].values
        label_array = self.data['labels'].values

        # Generate snippets with length equal to "timesteps"
        for index in range(self.data.shape[0] - self.timesteps + 1):
            this_array = d_array[index:index + self.timesteps].reshape((-1, self.timesteps, len(self.columns)))
            timesteps_label = label_array[index:index + self.timesteps]
            if np.any(timesteps_label == -1):  # If any single observation in snippet is defect, the snippet is defect.
                this_label = -1
            else:
                this_label = 1
            if self.all_data.shape[0] == 0:
                self.all_data = this_array
                self.labels = this_label
            else:
                self.all_data = np.concatenate([self.all_data, this_array], axis=0)
                self.labels = np.append(self.labels, this_label)
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
