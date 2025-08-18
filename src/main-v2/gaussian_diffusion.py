import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from diffusion_scheduler import get_named_beta_schedule


class GaussianDiffusion:

    def __init__(
            self,
            num_timesteps: int = 1000,
            beta_schedule: str = "linear",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        betas = get_named_beta_schedule(schedule_name=beta_schedule, num_timesteps=num_timesteps)
        # Convert to tensors and precompute values
        self.betas = torch.from_numpy(betas).float().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute frequently used values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior coefficients
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, arr: torch.Tensor, timesteps: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Efficient value extraction"""
        out = arr[timesteps]
        return out.view(-1, *((1,) * (len(shape) - 1))).expand(shape)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: add noise to clean data"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from noise prediction"""
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Compute posterior q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)

        return posterior_mean, posterior_variance

    def get_noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise level for analysis"""
        return 1.0 - self._extract(self.alphas_cumprod, t, (len(t), 1, 1))