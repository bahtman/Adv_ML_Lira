import numpy as np
import torch
import torch.nn as nn
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def elbo_loss(recon_x, x, mu, log_var):
    criterion = nn.L1Loss(reduction='sum').to(device)
    recon_loss = criterion(recon_x, x)
    # From https://arxiv.org/abs/1312.6114 Eq. (10)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + KLD

def train_model(model, train_dataset, val_dataset, n_epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = elbo_loss
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        train_dataset_batch = iter(train_dataset)
        for _ in range(len(train_dataset)):
            sample = train_dataset_batch.next()
            x, y = sample
            x, y = x.float(), y.float()
            x = x.permute(1, 0, 2)
            optimizer.zero_grad()
            x_recon, mu, log_var = model(x)
            loss = loss_func(x_recon, x, mu, log_var)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            val_dataset_batch = iter(val_dataset)
            for _ in range(len(val_dataset)):
                x, y = val_dataset_batch.next()
                x, y = x.float(), y.float()
                x = x.permute(1, 0, 2)
                x_recon, mu, log_var = model(x)
                loss = loss_func(x_recon, x, mu, log_var)
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
