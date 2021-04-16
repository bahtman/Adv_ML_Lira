def elbo_loss(x, z, p_z, p_x_z, q_z_x):
    kl = q_z_x.log_prob(z).sum(1) - p_z.log_prob(z).sum(1)
    log_px = p_x_z.log_prob(x).sum(0).sum(1)
    loss =  p_x_z.log_prob(x).sum(0).sum(1) - kl
    return -loss.mean(), loss, log_px, kl
          