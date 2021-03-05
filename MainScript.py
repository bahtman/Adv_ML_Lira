from TestNetwork import *
from TrainScript import *
from src.dataset import *
from torch.utils.data import random_split
import copy
data = TS_dataset()
seq_len = len(data[0][0])#10 for vores data
n_features = len(data[0][0][0])#1 for vores data
model = RecurrentAutoencoder(seq_len, n_features, 128)
#model = model.to(device)

val_percent = 0.1
n_val = int(len(data) * val_percent)
n_train = len(data) - n_val
train, val = random_split(data, [n_train, n_val])

model, history = train_model(
    model,
    train,
    val,
    n_epochs=150
)

