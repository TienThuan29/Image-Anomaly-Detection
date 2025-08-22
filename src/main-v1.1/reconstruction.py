from typing import Any
import torch
import numpy as np
from config import load_config

config = load_config()

# diffusion model config
_beta_start = config.diffusion_model.beta_start
_beta_end = config.diffusion_model.beta_end
_trajectory_steps = config.diffusion_model.num_timesteps
_test_trajectoy_steps = config.diffusion_model.test_trajectoy_steps
_skip = config.diffusion_model.skip
_eta = config.diffusion_model.eta

"""
    The reconstruction process
    :param y: the target image
    :param x: the input image
    :param seq: the sequence of denoising steps
    :param unet: the UNet model
    :param x0_t: the prediction of x0 at time step t
"""
class Reconstruction:

    def __init__(self, unet, device: torch.device) -> None:
        self.unet = unet
        self.device = device


    def __call__(self, x, y0, w) -> Any:
        def _compute_alpha(t):
            betas = np.linspace(_beta_start, _beta_end, _trajectory_steps, dtype=np.float64)
            betas = torch.tensor(betas).type(torch.float).to(self.device)
            beta = torch.cat([torch.zeros(1).to(self.device), betas], dim=0)
            beta = beta.to(self.device)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
            return a

        test_trajectoy_steps = torch.Tensor([_test_trajectoy_steps]).type(torch.int64).to(self.device).long()
        at = _compute_alpha(test_trajectoy_steps)
        xt = at.sqrt() * x + (1 - at).sqrt() * torch.randn_like(x).to(self.device)
        seq = range(0, _test_trajectoy_steps, _skip)

        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [xt]
            for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(n) * i).to(self.device)
                next_t = (torch.ones(n) * j).to(self.device)
                at = _compute_alpha(t.long())
                at_next = _compute_alpha(next_t.long())
                xt = xs[-1].to(self.device)
                self.unet = self.unet.to(self.device)

                # UNet dự đoán noise tại timestep t
                et = self.unet(xt, t)

                # Ảnh có điều kiện
                yt = at.sqrt() * y0 + (1 - at).sqrt() * et

                # Noise được điều chỉnh
                et_hat = et - (1 - at).sqrt() * w * (yt - xt)
                x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
                c1 = (
                        _eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
                xs.append(xt_next)
        return xs # a list





