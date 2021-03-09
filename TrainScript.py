import numpy as np
import torch
import torch.nn as nn
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        train_dataset_batch = iter(train_dataset)
        for i in range(len(train_dataset)):
            sample=train_dataset_batch.next()
            optimizer.zero_grad()
            #sample = sample.to(device)
            seq_pred = model(sample[0].float())
            loss = criterion(seq_pred, sample[0].float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                #seq_true = seq_true.to(device)
                seq_pred = model(seq_true[0].float())
                loss = criterion(seq_pred, seq_true[0].float())
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history