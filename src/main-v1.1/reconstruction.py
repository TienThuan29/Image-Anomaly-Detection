from typing import List
import torch


"""
    The reconstruction process
    :param y: the target image
    :param x: the input image
    :param seq: the sequence of denoising steps
    :param unet: the UNet model
    :param x0_t: the prediction of x0 at time step t
"""
class Reconstruction:

    def __init__(self, unet, gaussian_diffusion, device):
        self.unet = unet.to(device).eval()
        self.gaussian_diffusion = gaussian_diffusion
        self.device = device

    @torch.no_grad()
    def __call__(self, x, y0, w):
        strength = 0.3 # độ mạnh của noise được bắt đầu từ
        steps = 70
        eta = 0.0

        B = x.size(0)
        # B = y0.size(0)
        T = self.gaussian_diffusion.num_timesteps
        t0 = int(min(T - 1, max(1, int(strength * T))))

        step = max(1, t0 // steps)
        seq = list(range(t0, -1, -step))
        if seq[-1] != 0: seq.append(0)

        # khởi tạo xt từ x đúng tại t0
        t_init = torch.full((B,), t0, device=self.device, dtype=torch.long)
        xt = self.gaussian_diffusion.q_sample(x, t_init, torch.randn_like(x))
        # xt = self.diff.q_sample(y0, t_init, torch.randn_like(y0))
        xs: List[torch.Tensor] = [xt]

        for ti, tj in zip(seq[:-1], seq[1:]):
            ti_t = torch.full((B,), ti, device=self.device, dtype=torch.long)
            tj_t = torch.full((B,), tj, device=self.device, dtype=torch.long)

            a_i = self.gaussian_diffusion._extract(self.gaussian_diffusion.alphas_cumprod, ti_t, xt.shape)
            a_j = self.gaussian_diffusion._extract(self.gaussian_diffusion.alphas_cumprod, tj_t, xt.shape)
            sqrt_one_minus_ai = self.gaussian_diffusion._extract(self.gaussian_diffusion.sqrt_one_minus_alphas_cumprod, ti_t, xt.shape)
            sqrt_one_minus_aj = self.gaussian_diffusion._extract(self.gaussian_diffusion.sqrt_one_minus_alphas_cumprod, tj_t, xt.shape)

            # UNet dự đoán noise tại ti
            e = self.unet(xt, ti_t)

            # Ước lượng x0 tại ti (chuẩn DDIM/DDPM)
            x0 = (xt - sqrt_one_minus_ai * e) / torch.sqrt(a_i + 1e-8)

            if eta == 0.0:
                # DDIM deterministic
                xt = torch.sqrt(a_j) * x0 + sqrt_one_minus_aj * e
            else:
                # DDIM stochastic
                sigma = eta * torch.sqrt(
                    (1 - a_j) / (1 - a_i + 1e-8) * (1 - a_i / (a_j + 1e-8) + 1e-8)
                )
                c = torch.sqrt(torch.clamp(1 - a_j - sigma ** 2, min=0.0))
                z = torch.randn_like(xt)
                xt = torch.sqrt(a_j) * x0 + c * e + sigma * z

            xs.append(xt)

        return xt




# from typing import Any, List
# import torch
# from diffusion_gaussian import GaussianDiffusion
# from config import load_config
#
# config = load_config()
#
# # diffusion model config
# # _beta_start = config.diffusion_model.beta_start
# # _beta_end = config.diffusion_model.beta_end
# # _trajectory_steps = config.diffusion_model.num_timesteps
# # _test_trajectoy_steps = config.diffusion_model.test_trajectoy_steps
# # _skip = config.diffusion_model.skip
# # _eta = config.diffusion_model.eta
#
# class Reconstruction:
#     def __init__(self, unet, diffusion: GaussianDiffusion,device: torch.device) -> None:
#         self.unet = unet
#         self.device = device
#         self.diffusion = diffusion
#         # Config parameters
#         self._test_trajectory_steps = config.diffusion_model.test_trajectoy_steps
#         self._skip = config.diffusion_model.skip
#         self._eta = config.diffusion_model.eta
#
#     def __call__(self, x, y0, w) -> Any:
#         """
#         Reconstruction process using GaussianDiffusion
#         :param x: the input image
#         :param y: the target image (y0)
#         :param w: weight parameter
#         """
#         # Khởi tạo với noise
#         test_trajectory_steps = torch.tensor([self._test_trajectory_steps]).long().to(self.device)
#
#         # Sử dụng q_sample từ GaussianDiffusion để tạo xt
#         xt = self.diffusion.q_sample(x, test_trajectory_steps)
#
#         # Tạo sequence cho denoising
#         seq = range(0, self._test_trajectory_steps, self._skip)
#
#         with torch.no_grad():
#             n = x.size(0)
#             seq_next = [-1] + list(seq[:-1])
#             xs = [xt]
#
#             for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
#                 t = (torch.ones(n) * i).long().to(self.device)
#                 next_t = (torch.ones(n) * j).long().to(self.device)
#
#                 # Sử dụng get_alpha_cumprod từ GaussianDiffusion
#                 at = self.diffusion.get_alpha_cumprod(t)
#                 at_next = self.diffusion.get_alpha_cumprod(next_t) if j >= 0 else torch.ones_like(at)
#
#                 xt = xs[-1].to(self.device)
#                 self.unet = self.unet.to(self.device)
#
#                 # UNet dự đoán noise tại timestep t
#                 et = self.unet(xt, t.float())
#
#                 # Ảnh có điều kiện - sử dụng q_sample
#                 yt = self.diffusion.q_sample(y0, t, et)
#
#                 # Noise được điều chỉnh
#                 et_hat = et - (1 - at).sqrt() * w * (yt - xt)
#
#                 # Predict x0 using GaussianDiffusion method
#                 x0_t = self.diffusion.predict_start_from_noise(xt, t, et_hat)
#
#                 # DDIM sampling coefficients
#                 c1 = self._eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
#                 c2 = ((1 - at_next) - c1 ** 2).sqrt()
#
#                 # Next step
#                 xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
#                 xs.append(xt_next)
#
#         return xs  # a list of denoised samples










# from typing import Any
# import torch
# import numpy as np
# from config import load_config
#
# config = load_config()
#
# # diffusion model config
# _beta_start = config.diffusion_model.beta_start
# _beta_end = config.diffusion_model.beta_end
# _trajectory_steps = config.diffusion_model.num_timesteps
# _test_trajectoy_steps = config.diffusion_model.test_trajectoy_steps
# _skip = config.diffusion_model.skip
# _eta = config.diffusion_model.eta
#
# """
#     The reconstruction process
#     :param y: the target image
#     :param x: the input image
#     :param seq: the sequence of denoising steps
#     :param unet: the UNet model
#     :param x0_t: the prediction of x0 at time step t
# """
# class Reconstruction:
#
#     def __init__(self, unet, device: torch.device) -> None:
#         self.unet = unet
#         self.device = device
#
#
#     def __call__(self, x, y0, w) -> Any:
#         def _compute_alpha(t):
#             betas = np.linspace(_beta_start, _beta_end, _trajectory_steps, dtype=np.float64)
#             betas = torch.tensor(betas).type(torch.float).to(self.device)
#             beta = torch.cat([torch.zeros(1).to(self.device), betas], dim=0)
#             beta = beta.to(self.device)
#             a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
#             return a
#
#         test_trajectoy_steps = torch.Tensor([_test_trajectoy_steps]).type(torch.int64).to(self.device).long()
#         at = _compute_alpha(test_trajectoy_steps)
#         xt = at.sqrt() * x + (1 - at).sqrt() * torch.randn_like(x).to(self.device)
#         seq = range(0, _test_trajectoy_steps, _skip)
#
#         with torch.no_grad():
#             n = x.size(0)
#             seq_next = [-1] + list(seq[:-1])
#             xs = [xt]
#             for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
#                 t = (torch.ones(n) * i).to(self.device)
#                 next_t = (torch.ones(n) * j).to(self.device)
#                 at = _compute_alpha(t.long())
#                 at_next = _compute_alpha(next_t.long())
#                 xt = xs[-1].to(self.device)
#                 self.unet = self.unet.to(self.device)
#
#                 # UNet dự đoán noise tại timestep t
#                 et = self.unet(xt, t)
#
#                 # Ảnh có điều kiện
#                 yt = at.sqrt() * y0 + (1 - at).sqrt() * et
#
#                 # Noise được điều chỉnh
#                 et_hat = et - (1 - at).sqrt() * w * (yt - xt)
#                 x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
#                 c1 = (
#                         _eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
#                 )
#                 c2 = ((1 - at_next) - c1 ** 2).sqrt()
#                 xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
#                 xs.append(xt_next)
#         return xs # a list

