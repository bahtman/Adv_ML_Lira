from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
#import pickle5 as pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split


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
            import symengine
            import timesynth as ts
            # Initialize samplers
            time_sampler = ts.TimeSampler(stop_time=20)

            # Regular signal
            regular_time_samples = time_sampler.sample_regular_time(num_points=1000)
            reg_signalgen = ts.signals.Sinusoidal(frequency=4)
            noise = ts.noise.GaussianNoise(std=0.1)
            reg_timeseries = ts.TimeSeries(reg_signalgen, noise_generator=noise)

            # Defect signal
            irregular_time_samples = time_sampler.sample_irregular_time(num_points=2000, keep_percentage=50)
            irreg_signalgen = ts.signals.ar.AutoRegressive(ar_param=[0.9], sigma=0.5, start_value=[0.0])
            irreg_timeseries = ts.TimeSeries(irreg_signalgen, noise_generator=noise)

            # Generate regular
            samples, signals, errors = reg_timeseries.sample(regular_time_samples)
            #samples -= 0.0
            #samples[samples < 0] = 0

            # Generate defect
            defect_samples, defect_signals, defect_errors = irreg_timeseries.sample(irregular_time_samples)
            defect_samples -= 2.0
            defect_samples[defect_samples < 0] = 0

            # save as Dataframe
            data = pd.DataFrame(
                {'samples': samples + defect_samples, 'labels': [1 if x == 0 else -1 for x in defect_samples]})
            data = data[data.labels == 1]
            #plt.plot(samples + defect_samples)
            #print(samples.shape)
            #plt.show()


        # Assume data column is always 'samples'
            self.columns = ['samples']
            self.data = data
            self.process_synth()
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
            
    def process_synth(self):

        # Normalization
        standscaler = StandardScaler()
        #self.data[self.columns] = standscaler.fit_transform(self.data[self.columns])

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
    def __splitdata__(self):
        print(self.all_data)
        print(self.labels)
        data = self.data
        val_percent = 0.2
        test_percent = 0.1
        n_val = int(length(data)*val_percent)
        n_test = int(length(data)*test_percent)
        n_train = int(length(data)-(n_val+n_test))
        train, val, test = random_split(data, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
        train = train[train['label'] == 1]
        print(train, val, test)
        return train, val, test