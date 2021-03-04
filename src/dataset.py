from torch.utils.data import Dataset
import torch
import symengine
import timesynth as ts
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler


class TS_dataset(Dataset):

    def __init__(self, datafile=None, timesteps=10):
        self.timesteps = timesteps
        if datafile:
            True
            #dataset = pickle.load(open(datafile,'rb'))
            #data = pd.DataFrame.from_dict(dataset)
            #self.columns = blabla
        else:
            ###Initialize samplers
            time_sampler = ts.TimeSampler(stop_time=20)
            
            #Regular signal
            regular_time_samples = time_sampler.sample_regular_time(num_points=1000)
            reg_signalgen = ts.signals.Sinusoidal(frequency=0.25)
            noise = ts.noise.GaussianNoise(std=0.1)
            reg_timeseries = ts.TimeSeries(reg_signalgen, noise_generator=noise)
            #Defect signal
            irregular_time_samples = time_sampler.sample_irregular_time(num_points=2000, keep_percentage=50)
            irreg_signalgen = ts.signals.ar.AutoRegressive(ar_param=[0.9], sigma=0.5, start_value=[0.0])
            irreg_timeseries = ts.TimeSeries(irreg_signalgen, noise_generator=noise)
            
            #Generate regular
            samples, signals, errors = reg_timeseries.sample(regular_time_samples)
            signals-=0.0
            signals[signals<0]=0
            #Generate defect
            defect_samples, defect_signals, defect_errors = irreg_timeseries.sample(irregular_time_samples)
            defect_signals-=2.0
            defect_signals[defect_signals<0]=0
            #save as Dataframe
            data = pd.DataFrame({'signal':signals+defect_signals,'labels':[1 if x== 0 else -1 for x in defect_signals] })
            #Data column
            self.columns = ['signal']

        self.data = data
        self.process_data()
        # print(signals)
        # print(self.all_data[200,:,:], self.labels[200])


    def process_data(self):
        #Normalization
        standscaler = StandardScaler()
        mscaler = MinMaxScaler(feature_range=(0,1))
        self.data[self.columns] = standscaler.fit_transform(self.data[self.columns])
        self.data[self.columns] = mscaler.fit_transform(self.data[self.columns])
        
        #Init output arrays
        self.all_data = np.array([])
        self.labels = np.array([])

        #Extract data from dataframe
        d_array = self.data[self.columns].values  
        label_array = self.data['labels'].values

        #Generate snippets with length equal to "timesteps"
        for index in range(self.data.shape[0]-self.timesteps+1):
            this_array = d_array[index:index+self.timesteps].reshape((-1,self.timesteps,len(self.columns)))
            timesteps_label = label_array[index:index+self.timesteps]
            if np.any(timesteps_label==-1): #If any single observation in snippet is defect, the snippet is defect.
                this_label = -1
            else:
                this_label = 1
            if self.all_data.shape[0] == 0:
                self.all_data = this_array
                self.labels = this_label                    
            else:
                self.all_data = np.concatenate([self.all_data,this_array],axis=0)
                self.labels = np.append(self.labels,this_label)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.all_data[idx,:,:]
        label = self.labels[idx]
        
        return data, label