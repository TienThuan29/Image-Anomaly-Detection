from typing import List
import torch
from config import load_config

config = load_config()
_strength = config.diffusion_model.reconstruction.strength # độ mạnh của noise được bắt đầu từ strength
_steps = config.diffusion_model.reconstruction.steps
_eta = config.diffusion_model.reconstruction.eta

class Reconstruction:

    def __init__(self, unet, gaussian_diffusion, device):
        self.unet = unet.to(device).eval()
        self.gaussian_diffusion = gaussian_diffusion
        self.device = device

    @torch.no_grad()
    def __call__(self, x):
        B = x.size(0)
        T = self.gaussian_diffusion.num_timesteps
        t0 = int(min(T - 1, max(1, int(_strength * T))))

        step = max(1, t0 // _steps)
        seq = list(range(t0, -1, -step))
        if seq[-1] != 0: seq.append(0)

        t_init = torch.full((B,), t0, device=self.device, dtype=torch.long)
        xt = self.gaussian_diffusion.q_sample(x, t_init, torch.randn_like(x))
        xs: List[torch.Tensor] = [xt]

        for ti, tj in zip(seq[:-1], seq[1:]):
            ti_t = torch.full((B,), ti, device=self.device, dtype=torch.long)
            tj_t = torch.full((B,), tj, device=self.device, dtype=torch.long)

            a_i = self.gaussian_diffusion._extract(self.gaussian_diffusion.alphas_cumprod, ti_t, xt.shape)
            a_j = self.gaussian_diffusion._extract(self.gaussian_diffusion.alphas_cumprod, tj_t, xt.shape)
            sqrt_one_minus_ai = self.gaussian_diffusion._extract(self.gaussian_diffusion.sqrt_one_minus_alphas_cumprod, ti_t, xt.shape)
            sqrt_one_minus_aj = self.gaussian_diffusion._extract(self.gaussian_diffusion.sqrt_one_minus_alphas_cumprod, tj_t, xt.shape)

            e = self.unet(xt, ti_t)

            # Ước lượng x0 tại t_i
            x0 = (xt - sqrt_one_minus_ai * e) / torch.sqrt(a_i + 1e-8)

            if _eta == 0.0:
                xt = torch.sqrt(a_j) * x0 + sqrt_one_minus_aj * e
            else:
                sigma = _eta * torch.sqrt(
                    (1 - a_j) / (1 - a_i + 1e-8) * (1 - a_i / (a_j + 1e-8) + 1e-8)
                )
                c = torch.sqrt(torch.clamp(1 - a_j - sigma ** 2, min=0.0))
                z = torch.randn_like(xt)
                xt = torch.sqrt(a_j) * x0 + c * e + sigma * z
            xs.append(xt)

        return xt

