from torch.utils.data import DataLoader, random_split
import torch

def Split_Function(dataset):
    val_percent = 0.2
    test_percent = 0.1
    n_val = int(len(dataset)*val_percent)
    n_test = int(len(dataset)*test_percent)
    n_train = int(len(dataset)-(n_val+n_test))
    train, val, test = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    train = train[train['label'] == 1]
    return train, val, test

