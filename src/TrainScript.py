import numpy as np
import torch
import torch.nn as nn
import copy
import pandas as pd
from LossFunction import elbo_loss
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          

def train_model(model, train_dataset, val_dataset,ARGS):
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)
    n_epochs = ARGS.n_epochs
    loss_func = elbo_loss
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    column_names = ["elbo", "log_px", "kl"]
    train_diagnostics_df = pd.DataFrame(columns = column_names)
    val_diagnostics_df = pd.DataFrame(columns = column_names)
    temp_train_diagnostics = np.zeros(3)
    temp_val_diagnostics = np.zeros(3)
    model.to(ARGS.device)
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        #Initialization for plotting
        train_losses = []
        train_elbo_loss = []
        train_log_px_loss = []
        train_kl_loss = []

        train_dataset_batch = iter(train_dataset)
        for _ in range(len(train_dataset)):
            sample = train_dataset_batch.next()
            x, y = sample
            x, y = x.float().to(ARGS.device), y.float().to(ARGS.device)
            x = x.permute(1, 0, 2)
            x, z, p_z, q_z_x, p_x_z = model(x)
            loss, elbo, log_px, kl = loss_func(x, z, p_z, p_x_z,q_z_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #These three are just for plotting.
            train_elbo_loss.append(elbo.item())
            train_log_px_loss.append(log_px.item())
            train_kl_loss.append(kl.item())

            train_losses.append(loss.item())
        #Just for plotting
        temp_train_diagnostics[0]= np.mean(train_elbo_loss)
        temp_train_diagnostics[1]= np.mean(train_log_px_loss)
        temp_train_diagnostics[2]= np.mean(train_kl_loss)
        train_diagnostics_df.loc[epoch-1] = (temp_train_diagnostics)

        #Initialization for plotting
        val_losses = []
        val_elbo_loss = []
        val_log_px_loss = []
        val_kl_loss = []

        model = model.eval()
        with torch.no_grad():
            val_dataset_batch = iter(val_dataset)
            for _ in range(len(val_dataset)):
                x, y = val_dataset_batch.next()
                x, y = x.float().to(ARGS.device), y.float().to(ARGS.device)
                x = x.permute(1, 0, 2)
                x, z, p_z, q_z_x, p_x_z = model(x)
                loss, elbo, log_px, kl = loss_func(x, z, p_z, p_x_z, q_z_x )
                #De her 3 append er bare for plotting
                val_elbo_loss.append(elbo.item())
                val_log_px_loss.append(log_px.item())
                val_kl_loss.append(kl.item())

                val_losses.append(loss.item())
        #Igen de her 3 er bare for plotting
        temp_val_diagnostics[0]= np.mean(val_elbo_loss)
        temp_val_diagnostics[1]= np.mean(val_log_px_loss)
        temp_val_diagnostics[2]= np.mean(val_kl_loss)
        val_diagnostics_df.loc[epoch-1] = (temp_val_diagnostics)
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    plt.plot(p_x_z[:,:,0])
    #plt.plot(x[:,:,0])
    #print(samples.shape)
    plt.show()
    return model.eval(), history, train_diagnostics_df, val_diagnostics_df
