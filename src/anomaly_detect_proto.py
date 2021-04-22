import logging
from  LossFunction import elbo_loss
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def detect(model, test_dataloader, device):
    batch = iter(test_dataloader)
    model.eval()

    losses = []
    x_list = []
    labels = []
    for _ in range(len(test_dataloader)):
        sample = batch.next()
        x, y = sample
        x, y = x.float().to(device), y.float().to(device)
        x = x.permute(1, 0, 2)

        x, z, p_z, q_z_x, p_x_z = model(x)
        loss_mean, loss, log_px, kl = elbo_loss(x, z, p_z, p_x_z, q_z_x)
        losses.append(log_px.detach().item())
        x_list.append(x.detach().mean())
        labels.append(y.detach().item())
    print("Losses stats:")
    print('mean', np.mean(losses))
    print('std', np.std(losses))
    print('min', np.min(losses))
    print('max', np.max(losses))
    print('median', np.median(losses))

    f, ax = plt.subplots(figsize=(20, 20))

    ax.scatter(x_list, losses, label=labels)
    ax.set_xlabel('mean of x')
    ax.set_ylabel('log_px')
    plt.show()


