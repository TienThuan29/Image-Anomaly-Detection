import torch
from torch import Tensor
from torch.functional import F


def vae_loss_function(
        x_hat: Tensor,
        x: Tensor,
        mu: Tensor,
        log_var: Tensor
):
    # Reconstruction loss (MSE)
    MSE = F.mse_loss(x_hat, x, reduction='sum')
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    loss = MSE + KLD
    return loss, MSE, KLD


def diffusion_loss_function(pred_noise, noise):
    return (noise - pred_noise).square().sum(dim=(1, 2, 3)).mean(dim=0)
