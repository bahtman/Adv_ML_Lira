from torch.utils.data import Dataset
import torch
import symengine
import timesynth as ts
import matplotlib.pyplot as plt

time_sampler = ts.TimeSampler(stop_time=20)
# Sampling irregular time samples
irregular_time_samples = time_sampler.sample_irregular_time(num_points=2000, keep_percentage=50)
regular_time_samples = time_sampler.sample_regular_time(num_points=1000)
# Initializing Sinusoidal signal
signalgen = ts.signals.Sinusoidal(frequency=0.25)
# Initializing Gaussian noise
noise = ts.noise.GaussianNoise(std=0.1)
# Initializing TimeSeries class with the signal and noise objects
#timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
timeseries = ts.TimeSeries(signalgen, noise_generator=noise)
# Sampling using the irregular time samples
#samples, signals, errors = timeseries.sample(irregular_time_samples)
#%%
# Sampling using the irregular time samples
samples, signals, errors = timeseries.sample(regular_time_samples)
#samples, signals, errors = timeseries.sample(irregular_time_samples)
##%%
#plt.plot(errors)
# set negative to zero
signals-=0.0
signals[signals<0]=0

#%% create defect
signalgen = ts.signals.ar.AutoRegressive(ar_param=[0.9], sigma=0.5, start_value=[0.0])
timeseries = ts.TimeSeries(signalgen, noise_generator=noise)
defect_samples, defect_signals, defect_errors = timeseries.sample(irregular_time_samples)
defect_signals-=2.0
defect_signals[defect_signals<0]=0

labels = [1 if x== 0 else -1 for x in defect_signals]