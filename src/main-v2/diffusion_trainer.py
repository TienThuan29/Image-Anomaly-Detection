import os
import math
import time
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from utils import (
    ConfigLoader,
    load_mvtec_train_dataset,
    load_mvtec_only_good_test_dataset,
    LossEarlyStopping,
)
from vae_model import VAEResNet
from diffusion_model import SwinIRDiffusion
from gaussian_diffusion import GaussianDiffusion


def set_seed(seed: int | None):
    if seed is None:
        return
    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_vae_pretrained(checkpoint_path: str, in_channels: int, latent_dim: int, out_channels: int,
                        device: torch.device) -> VAEResNet:
    model = VAEResNet(in_channels=in_channels, latent_dim=latent_dim, out_channels=out_channels).to(device)

    # Robust and safer loader
    try:
        import numpy as np
        safe_globals = [
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np.float32, np.float64, np.int32, np.int64,
            np.bool_, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64
        ]
        torch.serialization.add_safe_globals(safe_globals)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print(f"[INFO] Loaded VAE checkpoint with weights_only=True: {checkpoint_path}")
    except Exception as e:
        print(f"[WARN] Secure load failed, using fallback loader: {str(e)[:160]}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] VAE missing keys: {missing}, unexpected: {unexpected}")

    model.eval()
    return model


@torch.no_grad()
def diffusion_reverse(denoiser: SwinIRDiffusion, sched: GaussianDiffusion, x_t: torch.Tensor, t_start: int,
                      pred_type: str = 'epsilon') -> torch.Tensor:
    """
    Reverse diffusion from a given timestep t_start down to 0 using the model.
    pred_type: 'epsilon' if model predicts noise; 'start_x' if model predicts x0 directly.
    """
    device = x_t.device
    B = x_t.size(0)

    # Iterate t = t_start, ..., 0
    x_cur = x_t
    for t_scalar in range(t_start, -1, -1):
        t = torch.full((B,), t_scalar, dtype=torch.long, device=device)
        model_out = denoiser(x_cur, t)

        if pred_type == 'start_x':
            x0_pred = model_out
        else:
            eps_pred = model_out
            # x0 prediction from epsilon
            x0_pred = sched.predict_start_from_noise(x_cur, t, eps_pred)
        x0_pred = x0_pred.clamp(0.0, 1.0)

        # Posterior q(x_{t-1} | x_t, x0)
        mean, var = sched.q_posterior_mean_variance(x0_pred, x_cur, t)
        if t_scalar > 0:
            noise = torch.randn_like(x_cur)
            x_prev = mean + torch.sqrt(var) * noise
        else:
            x_prev = mean
        x_cur = x_prev

    return x_cur.clamp(0.0, 1.0)


class DiffusionPhase2Trainer:
    def __init__(self, config_path: str = "config.yml"):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config(config_path)
        self.data_cfg = self.config_loader.get_section("data")
        self.vae_cfg = self.config_loader.get_section("vae_model")
        self.diff_cfg = self.config_loader.get_section("diffusion_model")
        self.es_cfg = self.config_loader.get_section("early_stopping")

        self.category = self.data_cfg.get('category')

        # Device
        cuda_id = self.diff_cfg.get('cuda') if self.diff_cfg.get('cuda') is not None else self.vae_cfg.get('cuda', 0)
        self.device = torch.device(
            f"cuda:{cuda_id}" if (cuda_id is not None and cuda_id >= 0 and torch.cuda.is_available()) else "cpu")

        # AMP setup (disabled if not on CUDA/HIP)
        self.use_amp = bool(self.diff_cfg.get('amp', True)) and (self.device.type == 'cuda')
        self.amp_dtype = torch.float16
        if self.use_amp:
            print("[AMP] Mixed precision enabled (float16)")
        else:
            print("[AMP] Mixed precision disabled")

        # I/O dirs
        self.train_result_dir = os.path.join(
            self.diff_cfg.get('train_result_base_dir', './mvtec_phase2_diffusion/train_results/'), self.category)
        ensure_dir(self.train_result_dir)

        # Seed
        set_seed(self.diff_cfg.get('seed'))

        print(f"Category: {self.category}")
        print(f"Device:   {self.device}")
        print(f"Results:  {self.train_result_dir}")

        # Data (use diffusion batch_size)
        self.train_loader = load_mvtec_train_dataset(
            dataset_root_dir=self.data_cfg.get('mvtec_data_dir'),
            category=self.data_cfg.get('category'),
            image_size=self.data_cfg.get('image_size'),
            batch_size=self.diff_cfg.get('batch_size', 1),
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # VAE (frozen)
        self.vae = load_vae_pretrained(
            checkpoint_path=self.diff_cfg.get('phase1_model_path'),
            in_channels=self.diff_cfg.get('input_channels', 3),
            latent_dim=self.diff_cfg.get('z_dim', 256),
            out_channels=self.diff_cfg.get('output_channels', 3),
            device=self.device,
        )
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        # Model size from config with smaller defaults to avoid OOM
        embed_dim = self.diff_cfg.get('embed_dim', 64)
        depths = self.diff_cfg.get('depths', [4, 4, 4, 4])
        num_heads = self.diff_cfg.get('num_heads', [4, 4, 4, 4])
        window_size = self.diff_cfg.get('window_size', 8)
        use_checkpoint = bool(self.diff_cfg.get('use_checkpoint', True))

        # Diffusion denoiser model
        self.model = SwinIRDiffusion(
            img_size=self.diff_cfg.get('image_size', 256),
            in_chans=self.diff_cfg.get('input_channels', 3),
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            time_emb_dim=self.diff_cfg.get('z_dim', 256),
            use_checkpoint=use_checkpoint,
        ).to(self.device)

        # Scheduler
        self.sched = GaussianDiffusion(
            num_timesteps=self.diff_cfg.get('num_timesteps', 1000),
            beta_schedule=self.diff_cfg.get('beta_schedule', 'linear'),
            device=str(self.device),
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.diff_cfg.get('lr')),
            weight_decay=float(self.diff_cfg.get('weight_decay'))
        )

        # Early stopping on training loss
        self.early_stopper = None
        if self.es_cfg.get('is_early_stopping', True):
            self.early_stopper = LossEarlyStopping(
                patience=self.es_cfg.get('patience', 50),
                min_delta=self.es_cfg.get('min_delta', 0.0),
                smoothing_window=self.es_cfg.get('smoothing_window', 5),
                verbose=True,
            )

        # AMP GradScaler
        # Use new AMP API form
        self.scaler = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)

        # Logging
        self.global_step = 0

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_mse_recon_vs_orig = 0.0
        num_batches = 0

        # Create progress bar for batches
        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch:4d}",
                    unit="batch",
                    leave=False,
                    ascii=True,
                    ncols=120)

        for batch_idx, batch in enumerate(pbar):
            num_batches += 1
            images = batch['image'].to(self.device).clamp(0, 1)

            # Phase 1: VAE reconstruction (frozen)
            with torch.no_grad():
                recon = self.vae.reconstruct(images).clamp(0, 1)

            # Sample random timesteps per-image
            B = images.size(0)
            t = torch.randint(0, self.sched.num_timesteps, (B,), device=self.device, dtype=torch.long)

            # Add noise to the VAE reconstruction
            noise = torch.randn_like(recon)
            x_t = self.sched.q_sample(recon, t, noise=noise)

            model_mean_type = self.diff_cfg.get('model_mean_type', 'epsilon')

            # Forward under AMP
            if self.use_amp:
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                    eps_pred = self.model(x_t, t)
                    if model_mean_type == 'start_x':
                        loss = F.mse_loss(eps_pred, recon)
                    else:
                        loss = F.mse_loss(eps_pred, noise)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                eps_pred = self.model(x_t, t)
                if model_mean_type == 'start_x':
                    loss = F.mse_loss(eps_pred, recon)
                else:
                    loss = F.mse_loss(eps_pred, noise)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            loss_val = float(loss.detach().item())
            total_loss += loss_val

            # Occasionally compute a reverse pass MSE wrt original image (do a shorter reverse)
            mse_recon_vs_orig = 0.0
            if batch_idx % 5 == 0:  # Compute less frequently to save time
                with torch.no_grad():
                    t_eval_scalar = min(self.sched.num_timesteps - 1, int(self.sched.num_timesteps * 0.7))
                    t_eval = torch.full((B,), t_eval_scalar, dtype=torch.long, device=self.device)
                    xt_eval = self.sched.q_sample(recon, t_eval)
                    pred_type = self.diff_cfg.get('model_mean_type', 'epsilon')
                    x0_recovered = diffusion_reverse(self.model.eval(), self.sched, xt_eval, t_eval_scalar,
                                                     pred_type=pred_type)
                    mse_recon_vs_orig = F.mse_loss(x0_recovered, images).item()

            total_mse_recon_vs_orig += mse_recon_vs_orig

            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f'{loss_val:.6f}',
                'mse': f'{mse_recon_vs_orig:.6f}',
                'step': f'{self.global_step}'
            })

            self.global_step += 1

        pbar.close()

        avg_loss = total_loss / max(1, num_batches)
        avg_mse_rev_vs_orig = total_mse_recon_vs_orig / max(1, num_batches)
        return avg_loss, avg_mse_rev_vs_orig

    @torch.no_grad()
    def save_visuals(self, epoch: int):
        self.model.eval()
        batch = next(iter(self.train_loader))
        images = batch['image'].to(self.device).clamp(0, 1)
        recon = self.vae.reconstruct(images).clamp(0, 1)

        B = 1  # keep visuals light to avoid OOM
        images = images[:B]
        recon = recon[:B]

        # Forward noise at several steps
        sample_ts = [0, 50, 100, 250, 500, 750, self.sched.num_timesteps - 1]
        cols = [recon]

        # Show progress for forward diffusion steps
        print(f"[VISUAL] Generating forward diffusion panel for epoch {epoch}...")
        for t_scalar in tqdm(sample_ts[1:], desc="Forward steps", leave=False, ascii=True):
            t = torch.full((B,), t_scalar, dtype=torch.long, device=self.device)
            xt = self.sched.q_sample(recon, t).clamp(0, 1)
            cols.append(xt)

        panel = torch.cat(cols, dim=0)
        grid = make_grid(panel.cpu(), nrow=len(sample_ts), padding=2)
        save_image(grid, os.path.join(self.train_result_dir, f"epoch_{epoch:04d}_forward_panel.png"))

        # Reverse from a high noise level
        print(f"[VISUAL] Generating reverse diffusion for epoch {epoch}...")
        t_start = self.sched.num_timesteps - 1
        t = torch.full((B,), t_start, dtype=torch.long, device=self.device)
        xt = self.sched.q_sample(recon, t)
        pred_type = self.diff_cfg.get('model_mean_type', 'epsilon')

        # Add progress bar for reverse diffusion (optional, can be removed if too verbose)
        x0_rec = diffusion_reverse(self.model, self.sched, xt, t_start, pred_type=pred_type)

        pair = torch.cat([images.cpu(), recon.cpu(), x0_rec.cpu()], dim=0)
        save_image(make_grid(pair, nrow=B, padding=2),
                   os.path.join(self.train_result_dir, f"epoch_{epoch:04d}_pairs.png"))
        print(f"[VISUAL] Saved visualizations for epoch {epoch}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        filename = os.path.join(self.train_result_dir, f"{self.category}_diffusion_epoch_{epoch:04d}.pth")
        torch.save(ckpt, filename)
        if is_best:
            best_filename = os.path.join(self.train_result_dir, f"{self.category}_diffusion_best.pth")
            torch.save(ckpt, best_filename)
        print(f"[CKPT] Saved to {filename}")

    def run(self):
        epochs = self.diff_cfg.get('epochs', 500)
        best_loss = math.inf
        print(f"Start training for {epochs} epochs")

        # Create overall progress bar for epochs
        epoch_pbar = tqdm(range(1, epochs + 1),
                          desc="Training Progress",
                          unit="epoch",
                          ascii=True,
                          ncols=120)

        for epoch in epoch_pbar:
            epoch_start_time = time.time()

            train_loss, mse_rev_vs_orig = self.train_epoch(epoch)

            epoch_time = time.time() - epoch_start_time

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'loss': f'{train_loss:.6f}',
                'mse': f'{mse_rev_vs_orig:.6f}',
                'time': f'{epoch_time:.1f}s',
                'best': f'{best_loss:.6f}'
            })

            # Early stopping
            if self.early_stopper is not None:
                if self.early_stopper(train_loss):
                    print("\n[EARLY STOP] Early stopping triggered.")
                    self.save_checkpoint(epoch, is_best=False)
                    break

            # Checkpointing
            is_best = train_loss < best_loss
            if is_best:
                best_loss = train_loss
                epoch_pbar.set_postfix({
                    'loss': f'{train_loss:.6f}',
                    'mse': f'{mse_rev_vs_orig:.6f}',
                    'time': f'{epoch_time:.1f}s',
                    'best': f'{best_loss:.6f} ★'
                })

            if (epoch % self.diff_cfg.get('save_freq', 20) == 0) or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            # Visualizations
            if epoch % self.diff_cfg.get('sample_freq', 10) == 0:
                self.save_visuals(epoch)

        epoch_pbar.close()
        print(f"\n[COMPLETE] Training finished! Best loss: {best_loss:.6f}")


def main():
    trainer = DiffusionPhase2Trainer("config.yml")
    trainer.run()


if __name__ == "__main__":
    main()