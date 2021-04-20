import logging
from  LossFunction import elbo_loss

def detect(model, test_dataloader, device):
    logging.info('detected outlier')
    batch = iter(test_dataloader)
    model.eval()
    for _ in range(len(test_dataloader)):
        sample = batch.next()
        x, y = sample
        x, y = x.float().to(device), y.float().to(device)
        x = x.permute(1, 0, 2)

        x, z, p_z, q_z_x, p_x_z = model(x)
        loss, elbo, log_px, kl = elbo_loss(x, z, p_z, p_x_z,q_z_x)
        print(log_px.to_numpy())



