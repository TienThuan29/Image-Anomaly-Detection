import torch
from torch import nn
from functools import partial
from inspect import isfunction
import numpy as np
from tqdm import tqdm
from diffusion.model.beta_schedule import make_beta_schedule
from diffusion.model.ucdir import UNetSeeInDark


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def identity(t, *args, **kwargs):
    return t

""" Abstract for diffusion model """
class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            image_size,
            channels,
            loss_type='l1',
            conditional=True
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError('No loss type matches')

    def set_new_noise_schedule(self, schedule: str, n_timestep: int, linear_start: float, linear_end: float, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule,
            n_timestep=n_timestep,
            linear_start=linear_start,
            linear_end=linear_end
        )  # 1e-6 1e-2 2k
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas  # 1- 1e-6 1 - 1e-2 2k
        alphas_cumprod = np.cumprod(alphas, axis=0)  # 1 - 4e-5
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(  # x0 keeped ratio 1 - 0.006
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # Dùng register_buffer để các tensor này trở thành một phần của state model,
        # nhưng không học được gradient (chỉ lưu trong checkpoint).
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / (alphas_cumprod + 1e-10))))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / (alphas_cumprod + 1e-10) - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def _extract(self, a, t, x_shape):
        device = a.device
        B = x_shape[0]
        if isinstance(t, int):
            out = a[t].expand(B)  # [B]
        else:
            if t.dim() == 0:
                t = t.view(1).expand(B)  # [B]
            t = t.to(device=device, dtype=torch.long)
            out = a.index_select(0, t)  # [B]
        return out.view(B, *([1] * (len(x_shape) - 1)))  # [B,1,1,1]

    # def predict_start_from_noise(self, x_t, t, noise):
    #     return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise
    def predict_start_from_noise(self, x_t, t, noise):
        # coeffs: [B,1,1,1]
        coeff1 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        coeff2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return coeff1 * x_t - coeff2 * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, kwargs={}):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level, **kwargs))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level, **kwargs))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, kwargs={}):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, kwargs=kwargs)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, kwargs={}):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i, kwargs=kwargs)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x, kwargs=kwargs)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    # based on p_mean_variance
    # def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False, kwargs={}):
    #     # model_output = self.model(x, t, x_self_cond)
    #
    #     batch_size = x.shape[0]
    #     noise_level = torch.FloatTensor(
    #         [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
    #     model_output = self.denoise_fn(torch.cat([x_self_cond, x], dim=1), noise_level, **kwargs)
    #
    #     maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
    #
    #     if self.objective == 'pred_noise':
    #         pred_noise = model_output
    #         x_start = self.predict_start_from_noise(x, t, pred_noise)
    #         x_start = maybe_clip(x_start)
    #
    #         if clip_x_start and rederive_pred_noise:
    #             pred_noise = self.predict_noise_from_start(x, t, x_start)
    #
    #     elif self.objective == 'pred_x0':
    #         x_start = model_output
    #         x_start = maybe_clip(x_start)
    #         pred_noise = self.predict_noise_from_start(x, t, x_start)
    #
    #     elif self.objective == 'pred_v':
    #         v = model_output
    #         x_start = self.predict_start_from_v(x, t, v)
    #         x_start = maybe_clip(x_start)
    #         pred_noise = self.predict_noise_from_start(x, t, x_start)
    #
    #     from collections import namedtuple
    #     ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
    #     return ModelPrediction(pred_noise, x_start)

    # based on p_mean_variance
    def model_predictions(
            self,
            x,
            t,
            x_self_cond=None,
            clip_x_start: bool = False,
            rederive_pred_noise: bool = False,
            kwargs=None,
    ):
        """
        x: [B, C, H, W]
        t: int | scalar LongTensor | LongTensor[B]
        x_self_cond: [B, C, H, W] hoặc None (sẽ dùng zeros_like(x) nếu None)
        """
        if kwargs is None:
            kwargs = {}

        device = x.device
        B = x.shape[0]

        # self-conditioning
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x)

        # Đảm bảo sqrt_alphas_cumprod_prev là Tensor trên đúng device/dtype
        if torch.is_tensor(self.sqrt_alphas_cumprod_prev):
            sqrt_prev = self.sqrt_alphas_cumprod_prev.to(
                device=device, dtype=self.alphas_cumprod.dtype
            )
        else:
            sqrt_prev = torch.as_tensor(
                self.sqrt_alphas_cumprod_prev, device=device, dtype=self.alphas_cumprod.dtype
            )

        # Chuẩn hoá t -> LongTensor[B] và lấy (t+1) để tra sqrt_prev
        if isinstance(t, int):
            idx = torch.full((B,), min(t + 1, self.num_timesteps), device=device, dtype=torch.long)
        else:
            # t là tensor
            if t.dim() == 0:
                t = t.view(1).expand(B)
            t = t.to(device=device, dtype=torch.long)
            idx = (t + 1).clamp_max(self.num_timesteps)

        # noise_level: [B, 1]
        noise_level = sqrt_prev.index_select(0, idx).unsqueeze(1)

        # Gộp điều kiện và gọi denoise_fn
        model_input = torch.cat([x_self_cond, x], dim=1)
        model_output = self.denoise_fn(model_input, noise_level, **kwargs)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        from collections import namedtuple
        ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, x_in, continous=False, kwargs={}):
        shape = x_in.shape
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))

        self.ddim_sampling_eta = 0.0  # 0-> ddim 1 -> ddpm
        self.sampling_timesteps = 100
        self.objective = 'pred_noise'
        batch, total_timesteps, sampling_timesteps, eta = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        initx = self.predictor(x_in)
        self.pre_initx = initx
        kwargs = {'guide': initx}

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_in
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=True,
                                                             rederive_pred_noise=False, kwargs=kwargs)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)

        ret = img if not continous else torch.stack(imgs, dim=1)

        # ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    # def p_losses(self, x_in, noise=None):
    #     x_start = x_in['HR']
    #     [b, c, h, w] = x_start.shape
    #     t = np.random.randint(1, self.num_timesteps + 1)
    #     continuous_sqrt_alpha_cumprod = torch.FloatTensor(
    #         np.random.uniform(
    #             self.sqrt_alphas_cumprod_prev[t - 1],
    #             self.sqrt_alphas_cumprod_prev[t],
    #             size=b
    #         )
    #     ).to(x_start.device)
    #     continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
    #         b, -1)
    #
    #     noise = default(noise, lambda: torch.randn_like(x_start))
    #     x_noisy = self.q_sample(
    #         x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
    #
    #     if not self.conditional:
    #         x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
    #     else:
    #         x_recon = self.denoise_fn(
    #             torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
    #
    #     loss = self.loss_func(noise, x_recon)
    #     return loss

    def p_losses(self, x_in, noise=None):
        pass

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


""" ResiGaussianGuideDY: Class diffusion tích hợp cả hai mạng (UNetSeeInDark + DY3h)"""
# pass initial predicter to guide dynamic model
class ResiGaussianGuideDY(GaussianDiffusion):
    def __init__(
            self,
            denoise_fn,
            image_size,
            channels: int, # 3
            loss_type: str, # l1
            # schedule: str,
            # n_timestep: int,
            # linear_start: float,
            # linear_end: float,
            conditional=True
    ):
        super(ResiGaussianGuideDY, self).__init__(denoise_fn, image_size, channels, loss_type, conditional)
        # Init 'Initial Predictor' - First unet block
        self.predictor = UNetSeeInDark()

    def p_losses(self, x_in, noise=None): # x_in : bao gồm cả ảnh HR (ảnh xịn) và ảnh SR (ảnh cùi)

        """
            Đưa ảnh cùi đầu vào mô hình Unet đơn giản đầu tiên
            Đầu ra là một tensor có shape giống với input ([B, 3, H, W]),
            nhưng chưa qua activation nên output là giá trị chưa chuẩn hóa
        """
        x_init = self.predictor(x_in['SR'])

        # Residual = Ảnh xịn - Dự đoán ban đầu từ unet_init_predictor
        """
            x_init: ảnh “ước lượng ban đầu” từ input suy biến (SR). Nó nắm cấu trúc/thấp tần tốt nhưng còn thiếu chi tiết cao tần.
            x_start: đúng nghĩa là “ảnh phần bù (residual)” giữa ảnh chuẩn HR và ước lượng ban đầu x_init.
            Đây chính là “những gì còn thiếu” mà mô hình bổ sung.
        """
        x_start = x_in['HR'] - x_init  # residule input

        """ Diffusion process """
        [b, c, h, w] = x_start.shape

        # lấy timestep
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        # noise esp
        noise = default(noise, lambda: torch.randn_like(x_start))

        # thêm noise vào phần bù
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise
        )

        # if not self.conditional:
        #     x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        # else:
        kwargs = {'guide': x_init}

        # Dự đoán noise với guidance
        # Input: concat ảnh cùi với noise residual
        x_recon = self.denoise_fn(
            torch.cat([x_in['SR'], x_noisy], dim=1),
            continuous_sqrt_alpha_cumprod, **kwargs
        )

        loss = self.loss_func(noise, x_recon) # L1 loss giữa noise thật và dự đoán
        return loss

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        initx = self.predictor(x_in)
        self.pre_initx = initx
        kwargs = {'guide': initx}
        return self.p_sample_loop(x_in, continous, kwargs=kwargs) + initx



