import numpy as np
import torch
import torch.nn as nn
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def elbo_loss(x, z, p_z, p_x_z, q_z_x):
    kl = q_z_x.log_prob(z).sum(1) - p_z.log_prob(z).sum(1)
    loss =  p_x_z.log_prob(x).sum(0).sum(1) #- kl
    return -loss.mean()
          

def train_model(model, train_dataset, val_dataset,ARGS):
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)
    n_epochs = ARGS.n_epochs
    loss_func = elbo_loss
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    model.to(ARGS.device)
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        train_dataset_batch = iter(train_dataset)
        for _ in range(len(train_dataset)):
            sample = train_dataset_batch.next()
            x, y = sample
            x, y = x.float().to(ARGS.device), y.float().to(ARGS.device)
            x = x.permute(1, 0, 2)
            x, z, p_z, q_z_x, p_x_z = model(x)
            loss = loss_func(x, z, p_z, p_x_z,q_z_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            val_dataset_batch = iter(val_dataset)
            for _ in range(len(val_dataset)):
                x, y = val_dataset_batch.next()
                x, y = x.float().to(ARGS.device), y.float().to(ARGS.device)
                x = x.permute(1, 0, 2)
                x, z, p_z, q_z_x, p_x_z = model(x)
                loss = loss_func(x, z, p_z, p_x_z, q_z_x )
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
