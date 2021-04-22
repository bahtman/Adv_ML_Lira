import torch
import torch.distributions as d

def elbo_loss(x, z, p_z, p_x_z, q_z_x):
    p_z = d.Normal(torch.zeros_like(q_z_x.loc), torch.ones_like(q_z_x.scale))
    kl = d.kl_divergence(q_z_x, p_z).sum()
    log_px = -p_x_z.log_prob(x).sum(0).sum(1)
    loss =  log_px + kl
    return loss.mean(), loss, log_px, kl
          