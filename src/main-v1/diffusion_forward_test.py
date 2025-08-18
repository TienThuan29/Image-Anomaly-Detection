import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import glob
import numpy as np
import random
from utils import ConfigLoader
from vae_model import VAEResNet
from gaussian_diffusion import GaussianDiffusion

config_loader = ConfigLoader("config.yml")
config = config_loader.load_config()
data_config = config_loader.get_section("data")
vae_config = config_loader.get_section("vae_model")
diffusion_config = config_loader.get_section("diffusion_model")
early_stopping_config = config_loader.get_section("early_stopping")

category_name = data_config.get('category')
train_result_dir = vae_config.get('train_result_base_dir') + category_name
pretrained_save_dir = vae_config.get('pretrained_save_base_dir') + category_name

"""
Load VAE model for diffusion phase
"""


def load_vae(checkpoint_path: str, in_channels: int, latent_dim: int, out_channels: int, device: torch.device):
    model = VAEResNet(in_channels=in_channels, latent_dim=latent_dim, out_channels=out_channels).to(device)

    # Handle .pth checkpoint loading with proper security
    try:
        # Add safe globals for numpy objects commonly found in PyTorch checkpoints
        import numpy as np
        safe_globals = [
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np.float32, np.float64, np.int32, np.int64,
            np.bool_, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64
        ]

        # Add all safe globals at once
        torch.serialization.add_safe_globals(safe_globals)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print(f"[INFO] Successfully loaded {checkpoint_path} with weights_only=True")

    except Exception as e:
        # If weights_only=True still fails, use the fallback but warn about security
        print(f"[WARN] Secure loading failed, using fallback (potential security risk)")
        print(f"[WARN] Error details: {str(e)[:200]}...")  # Truncate long error messages
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both direct state_dict and nested checkpoint formats
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt  # Assume it's a direct state_dict

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing}, Unexpected keys: {unexpected}")

    model.eval()
    return model


def load_images_from_dir(img_dir: str, image_size: int, max_images: int | None = None):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    if max_images is not None:
        paths = paths[:max_images]
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0,1]
    ])
    imgs = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            imgs.append(tfm(im))
        except Exception as e:
            print(f"[SKIP] {p}: {e}")
    if len(imgs) == 0:
        raise RuntimeError(f"No images loaded from {img_dir}")
    x = torch.stack(imgs, dim=0)  # [B,3,H,W]
    return x, paths


def save_grid(tensors, nrow, out_path, caption=None):
    grid = make_grid(tensors, nrow=nrow, padding=2)
    save_image(grid, out_path)
    if caption:
        print(caption, "->", out_path)
    else:
        print("Saved ->", out_path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


@torch.no_grad()
def test_forward_diffusion(
        model: VAEResNet,
        diffusion: GaussianDiffusion,
        x0: torch.Tensor,
        sample_ts: list[int],
        out_dir: str,
        prefix: str = "diff"
):
    """
    Pipeline: x0 (ảnh gốc) -> VAE.reconstruct -> q_sample(recon, t)
    Lưu cả: x0_clean, recon_clean, và các mức nhiễu từ recon.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    # 1) Move data
    x0 = x0.to(device).clamp(0, 1)
    B = x0.size(0)

    # 2) Reconstruct qua VAE
    recon = model.reconstruct(x0).clamp(0, 1)

    # 3) Lưu riêng ảnh gốc & ảnh tái tạo (so sánh nhanh)
    save_image(x0.cpu(), os.path.join(out_dir, f"{prefix}_t=clean_x0.png"))
    save_image(recon.cpu(), os.path.join(out_dir, f"{prefix}_t=clean_recon.png"))

    # 4) Tạo panel: cột đầu = recon_clean, các cột sau = q_sample(recon, t)
    all_cols = []
    labels = []

    # cột 0: recon sạch
    all_cols.append(recon.cpu())
    labels.append("recon_clean")

    # các cột còn lại: diffuse từ recon
    for t_scalar in sample_ts:
        t = torch.full((B,), fill_value=t_scalar, dtype=torch.long, device=device)
        xt = diffusion.q_sample(recon, t).clamp(0, 1)
        all_cols.append(xt.cpu())
        labels.append(f"t={t_scalar}")

    # Ghép panel: mỗi hàng là 1 ảnh; mỗi cột là 1 mức t
    stacked = torch.cat(all_cols, dim=0)  # [(1+len(ts))*B, 3, H, W]
    ncol = len(sample_ts) + 1
    grid_path = os.path.join(out_dir, f"{prefix}_panel.png")
    grid = make_grid(stacked, nrow=ncol, padding=2)
    save_image(grid, grid_path)
    print(f"Columns: {', '.join(labels)} -> {grid_path}")

    # (tuỳ chọn) lưu từng cột
    for imgs, lab in zip(all_cols, labels):
        save_image(imgs, os.path.join(out_dir, f"{prefix}_{lab}.png"))


def main():
    set_seed(diffusion_config.get("seed"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load VAE
    model = load_vae(
        checkpoint_path=diffusion_config.get("phase1_model_path"),
        in_channels=diffusion_config.get("input_channels"),
        latent_dim=diffusion_config.get("z_dim"),
        out_channels=diffusion_config.get("output_channels"),
        device=device
    )

    # 2) Chuẩn bị x0
    img_dir = "/home/tienthuan29/workspaces/projects/AiResearch/ImageAnomalyDetection/datasets/mvtec_anomaly_detection/cable/train/good"
    out_dir = "./diffusion_forward_vis"
    x0, _ = load_images_from_dir(img_dir, diffusion_config.get("image_size"), max_images=1)

    diffusion = GaussianDiffusion(
        num_timesteps=diffusion_config.get("num_timesteps"),
        beta_schedule=diffusion_config.get("beta_schedule")
    )

    sample_ts = [0, 50, 100, 250, 500, 750, 999]

    test_forward_diffusion(
        model=model,
        diffusion=diffusion,
        x0=x0,
        sample_ts=sample_ts,
        out_dir=out_dir,
        prefix="vae_forward_diffusion"
    )


if __name__ == "__main__":
    main()
