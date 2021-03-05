from Network import *
from TrainScript import *
from Sim_Data import *
print(labels)
model = RecurrentAutoencoder(seq_len, n_features, 128)
#model = model.to(device)
model, history = train_model(
    model,
    train_dataset,
    val_dataset,
    n_epochs=150
)

dataset = MyBasicDataset(dir_img)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])