from TestNetwork import *
from TrainScript import *
from src.dataset import *
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from torch.autograd import Variable
import math
import copy
data = TS_dataset()
seq_len = len(data[0][0])#10 for vores data
n_features = len(data[0][0][0])#1 for vores data
model = RecurrentAutoencoder(seq_len, n_features, 64)
print(model)
#model = model.to(device)

val_percent = 0.1
n_val = int(len(data) * val_percent)
n_train = int(len(data) - n_val)
train, val = random_split(data, [n_train, n_val])
# make datasets iterable
train_loader = DataLoader(train, batch_size=2, shuffle=True, num_workers=0, drop_last=False)

n_val = int(len(val)*0.5)
n_test = int(len(val)-n_val)
val, test = random_split(val, [n_val, n_test])
val_loader = DataLoader(val, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
test_loader = DataLoader(test, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
model, history = train_model(
    model,
    train_loader,
    val_loader,
    n_epochs=10
)

