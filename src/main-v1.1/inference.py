import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC
from config import load_config
from utils import load_mvtec_test_dataset
from vae_resnet_model import VAEResNet
from vae_unet_model import VAEUnet
from diffusion_model import UNetModel
from diffusion_gaussian import GaussianDiffusion


@torch.no_grad()
def load_vae_model(
    checkpoint_path: str,
    vae_name: str,
    in_channels: int,
    latent_dim: int,
    out_channels: int,
    backbone: str,
    dropout_p: float,
    image_size: int,
    device: torch.device,
):
    if vae_name == 'vae_resnet':
        model = VAEResNet(
            image_size=image_size,
            in_channels=in_channels,
            latent_dim=latent_dim,
            out_channels=out_channels,
            resnet_name=backbone,
            dropout_p=dropout_p,
        ).to(device)
    elif vae_name == 'vae_unet':
        model = VAEUnet(
            in_channels=in_channels,
            latent_dim=latent_dim,
            out_channels=out_channels,
        ).to(device)
    else:
        raise ValueError(f"Unknown vae model: {vae_name}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] VAE missing keys: {missing}, unexpected: {unexpected}")
    model.eval()
    return model


@torch.no_grad()
def load_diffusion_unet(
    checkpoint_path: str,
    image_size: int,
    in_channels: int,
    device: torch.device,
):
    model = UNetModel(
        img_size=image_size,
        base_channels=128,
        conv_resample=True,
        n_heads=1,
        n_head_channels=64,
        channel_mults=(1, 1, 2, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0,
        attention_resolutions="32,16,8",
        biggan_updown=True,
        in_channels=in_channels,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] Diffusion UNet missing keys: {missing}, unexpected: {unexpected}")
    model.eval()
    return model


def compute_anomaly_map(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    # x, x_recon: [B, C, H, W] in [0,1]
    # per-pixel mean absolute error across channels -> [B, H, W]
    return (x - x_recon).abs().mean(dim=1)


def normalize_maps_global(anomaly_maps: List[torch.Tensor]) -> List[torch.Tensor]:
    # Concatenate along batch dimension to compute global min-max
    stacked = torch.cat(anomaly_maps, dim=0)  # [N, H, W]
    amin = stacked.min()
    amax = stacked.max()
    if (amax - amin) < 1e-8:
        return [torch.zeros_like(m) for m in anomaly_maps]
    return [(m - amin) / (amax - amin) for m in anomaly_maps]


def eval_auroc_image(labels: List[int], scores: List[float]) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def eval_auroc_pixel(anomaly_maps: List[torch.Tensor], gt_masks: List[torch.Tensor]) -> float:
    # Normalize maps to [0,1] globally
    norm_maps = normalize_maps_global(anomaly_maps)
    # Flatten all
    pred = torch.cat([m.flatten() for m in norm_maps], dim=0).cpu()
    gt = torch.cat([g.flatten() for g in gt_masks], dim=0).cpu().bool()
    metric = AUROC(task="binary")
    return float(metric(pred, gt))


def to_label_list(labels_field) -> List[int]:
    if isinstance(labels_field, list):
        return [int(x) for x in labels_field]
    if torch.is_tensor(labels_field):
        if labels_field.ndim == 0:
            return [int(labels_field.item())]
        return [int(x) for x in labels_field.tolist()]
    return [int(labels_field)]


@torch.no_grad()
def run_evaluation():
    config = load_config(os.path.join(os.path.dirname(__file__), 'config.yml'))

    # general
    image_size = config.general.image_size
    batch_size = config.general.batch_size
    input_channels = config.general.input_channels
    output_channels = config.general.output_channels
    cuda_idx = config.general.cuda
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")

    # data
    category_name = config.data.category
    mvtec_data_dir = config.data.mvtec_data_dir

    # vae
    vae_name = config.vae_model.name if hasattr(config, 'vae_model') else config.vae.name
    z_dim = config.vae_model.z_dim if hasattr(config, 'vae_model') else config.diffusion_model.z_dim
    dropout_p = config.vae_model.dropout_p if hasattr(config, 'vae_model') else config.diffusion_model.dropout_p
    backbone = getattr(config.vae_model, 'backbone', 'resnet18') if hasattr(config, 'vae_model') else getattr(config.vae, 'backbone', 'resnet18')

    # testing paths
    diff_test_cfg = getattr(config, 'diffusion_testing', None)
    if diff_test_cfg is None:
        raise ValueError("diffusion_testing section not found in config.yml")

    diffusion_model_path = diff_test_cfg.diffusion_model_path
    vae_model_path = diff_test_cfg.diffusion_vae_model_path if hasattr(diff_test_cfg, 'diffusion_vae_model_path') else diff_test_cfg.vae_model_path
    if not diffusion_model_path or not os.path.exists(diffusion_model_path):
        raise FileNotFoundError(f"Invalid diffusion_model_path: {diffusion_model_path}")
    if not vae_model_path or not os.path.exists(vae_model_path):
        raise FileNotFoundError(f"Invalid vae_model_path: {vae_model_path}")

    # diffusion settings
    num_timesteps = config.diffusion_model.num_timesteps
    beta_schedule = config.diffusion_model.beta_schedule

    print(f"[INFO] Evaluating category={category_name} on device={device}")
    print(f"[INFO] VAE: {vae_name}, backbone={backbone}")
    print(f"[INFO] Loading VAE from: {vae_model_path}")
    vae = load_vae_model(
        checkpoint_path=vae_model_path,
        vae_name=vae_name,
        in_channels=input_channels,
        latent_dim=z_dim,
        out_channels=output_channels,
        backbone=backbone,
        dropout_p=dropout_p,
        image_size=image_size,
        device=device,
    )

    print(f"[INFO] Loading diffusion UNet from: {diffusion_model_path}")
    unet = load_diffusion_unet(
        checkpoint_path=diffusion_model_path,
        image_size=image_size,
        in_channels=input_channels,
        device=device,
    )
    gaussian = GaussianDiffusion(num_timesteps=num_timesteps, beta_schedule=beta_schedule, device=str(device))

    # dataset
    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=mvtec_data_dir,
        category=category_name,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # Accumulators
    labels_all: List[int] = []
    img_scores_all: List[float] = []
    maps_all: List[torch.Tensor] = []
    gts_all: List[torch.Tensor] = []

    # Fixed t for inference (can be tuned). Use moderately noisy step.
    infer_t = max(1, int(0.4 * num_timesteps))

    print("[INFO] Running inference...")
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)  # [B,1,H,W]
        labels = batch['label']           # list/int/tensor

        # VAE reconstruction
        recon_vae, _, _ = vae(images)

        # Add noise at step t and denoise via UNet to predict x0
        B = images.size(0)
        t = torch.full((B,), infer_t, dtype=torch.long, device=device)
        noise = torch.randn_like(recon_vae)
        x_t = gaussian.q_sample(recon_vae, t, noise)
        pred_noise = unet(x_t, t)
        recon = gaussian.predict_start_from_noise(x_t, t, pred_noise).clamp(0, 1)

        # Anomaly map and image-level score
        anomaly_map = compute_anomaly_map(images, recon)  # [B,H,W]
        image_scores = anomaly_map.view(B, -1).max(dim=1).values  # max pooling

        # Accumulate
        labels_all.extend(to_label_list(labels))
        img_scores_all.extend(image_scores.detach().cpu().tolist())
        maps_all.extend(list(anomaly_map.detach().cpu()))
        gts_all.extend(list(masks.detach().cpu().squeeze(1)))

    # Metrics
    img_auroc = eval_auroc_image(labels_all, img_scores_all)
    px_auroc = eval_auroc_pixel(maps_all, gts_all)

    print(f"\n[RESULT] Image AUROC: {img_auroc:.4f}")
    print(f"[RESULT] Pixel AUROC: {px_auroc:.4f}")


if __name__ == '__main__':
    run_evaluation()