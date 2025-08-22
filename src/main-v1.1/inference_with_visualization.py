import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC
import matplotlib.pyplot as plt
from torchvision.utils import save_image
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
        base_channels=32,
        n_heads=2,
        num_res_blocks=2,
        dropout=0.1,
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


def save_visualization_comparison(
    images: torch.Tensor,
    recon_vae: torch.Tensor,
    recon_diffusion: torch.Tensor,
    anomaly_maps: torch.Tensor,
    save_dir: str,
    batch_idx: int,
    category_name: str,
    max_images: int = 8
):
    """
    Save visualization comparing input, VAE output, diffusion output, and anomaly heat map.
    
    Args:
        images: Input images [B, C, H, W]
        recon_vae: VAE reconstruction [B, C, H, W]
        recon_diffusion: Diffusion reconstruction [B, C, H, W]
        anomaly_maps: Anomaly maps [B, H, W]
        save_dir: Directory to save visualizations
        batch_idx: Batch index for naming
        category_name: Category name for naming
        max_images: Maximum number of images to save
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Limit number of images to save
    num_images = min(images.size(0), max_images)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Input image
        img_input = images[i].cpu().permute(1, 2, 0).numpy()
        img_input = np.clip(img_input, 0, 1)
        axes[i, 0].imshow(img_input)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # VAE reconstruction
        img_vae = recon_vae[i].cpu().permute(1, 2, 0).numpy()
        img_vae = np.clip(img_vae, 0, 1)
        axes[i, 1].imshow(img_vae)
        axes[i, 1].set_title('VAE Reconstruction')
        axes[i, 1].axis('off')
        
        # Diffusion reconstruction
        img_diff = recon_diffusion[i].cpu().permute(1, 2, 0).numpy()
        img_diff = np.clip(img_diff, 0, 1)
        axes[i, 2].imshow(img_diff)
        axes[i, 2].set_title('Diffusion Reconstruction')
        axes[i, 2].axis('off')
        
        # Anomaly heat map
        anomaly_map = anomaly_maps[i].cpu().numpy()
        im = axes[i, 3].imshow(anomaly_map, cmap='hot', interpolation='nearest')
        axes[i, 3].set_title('Anomaly Heat Map')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{category_name}_batch_{batch_idx:03d}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save individual images using torchvision for better quality
    individual_dir = os.path.join(save_dir, f'batch_{batch_idx:03d}_individual')
    os.makedirs(individual_dir, exist_ok=True)
    
    for i in range(num_images):
        # Save input
        save_image(images[i:i+1], 
                  os.path.join(individual_dir, f'{category_name}_input_{i:02d}.png'), 
                  normalize=True)
        
        # Save VAE reconstruction
        save_image(recon_vae[i:i+1], 
                  os.path.join(individual_dir, f'{category_name}_vae_recon_{i:02d}.png'), 
                  normalize=True)
        
        # Save diffusion reconstruction
        save_image(recon_diffusion[i:i+1], 
                  os.path.join(individual_dir, f'{category_name}_diffusion_recon_{i:02d}.png'), 
                  normalize=True)
        
        # Save anomaly map as heat map
        anomaly_map = anomaly_maps[i:i+1].unsqueeze(1)  # Add channel dimension
        save_image(anomaly_map, 
                  os.path.join(individual_dir, f'{category_name}_anomaly_map_{i:02d}.png'), 
                  normalize=True)
    
    print(f"[VIS] Saved visualization for batch {batch_idx} to {save_path}")


def save_summary_visualization(
    labels_all: List[int],
    img_scores_all: List[float],
    maps_all: List[torch.Tensor],
    save_dir: str,
    category_name: str
):
    """
    Create and save summary visualizations including score distributions.
    
    Args:
        labels_all: List of labels (0=normal, 1=anomaly)
        img_scores_all: List of image-level anomaly scores
        maps_all: List of pixel-level anomaly maps
        save_dir: Directory to save visualizations
        category_name: Category name for naming
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy arrays
    labels_np = np.array(labels_all)
    scores_np = np.array(img_scores_all)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Image-level score distribution
    normal_scores = scores_np[labels_np == 0]
    anomaly_scores = scores_np[labels_np == 1]
    
    axes[0, 0].hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
    axes[0, 0].hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red', density=True)
    axes[0, 0].set_xlabel('Image-level Anomaly Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Image-level Anomaly Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Pixel-level score statistics
    pixel_scores = torch.cat([m.flatten() for m in maps_all], dim=0).cpu().numpy()
    axes[0, 1].hist(pixel_scores, bins=50, alpha=0.7, color='green', density=True)
    axes[0, 1].set_xlabel('Pixel-level Anomaly Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Pixel-level Anomaly Score Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Score statistics table
    stats_text = f"""
    Image-level Statistics:
    - Normal samples: {len(normal_scores)}
    - Anomaly samples: {len(anomaly_scores)}
    - Normal mean: {normal_scores.mean():.4f}
    - Anomaly mean: {anomaly_scores.mean():.4f}
    - Normal std: {normal_scores.std():.4f}
    - Anomaly std: {anomaly_scores.std():.4f}
    
    Pixel-level Statistics:
    - Total pixels: {len(pixel_scores)}
    - Mean: {pixel_scores.mean():.4f}
    - Std: {pixel_scores.std():.4f}
    - Min: {pixel_scores.min():.4f}
    - Max: {pixel_scores.max():.4f}
    """
    
    axes[1, 0].text(0.1, 0.9, stats_text, transform=axes[1, 0].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Score Statistics')
    
    # 4. ROC curve visualization (if we have enough data)
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels_np, scores_np)
        axes[1, 1].plot(fpr, tpr, color='blue', linewidth=2)
        axes[1, 1].plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curve')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data for ROC curve', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('ROC Curve')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{category_name}_summary_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[VIS] Saved summary analysis to {save_path}")


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
    category_name = config.diffusion_testing.category
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

    # visualization settings
    save_visualizations = getattr(diff_test_cfg, 'save_visualizations', True)
    max_visualization_batches = getattr(diff_test_cfg, 'max_visualization_batches', 5)
    max_images_per_batch = getattr(diff_test_cfg, 'max_images_per_batch', 8)

    # diffusion settings
    num_timesteps = config.diffusion_model.num_timesteps
    beta_schedule = config.diffusion_model.beta_schedule

    print(f"[INFO] Evaluating category={category_name} on device={device}")
    print(f"[INFO] VAE: {vae_name}, backbone={backbone}")
    print(f"[INFO] Loading VAE from: {vae_model_path}")
    if save_visualizations:
        print(f"[INFO] Visualizations enabled: saving {max_visualization_batches} batches with max {max_images_per_batch} images each")
    else:
        print(f"[INFO] Visualizations disabled")
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

    # Setup visualization directory
    vis_dir = os.path.join(os.path.dirname(diffusion_model_path), 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print("[INFO] Running inference...")
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
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

        # Save visualization for first few batches
        if save_visualizations and batch_idx < max_visualization_batches:
            save_visualization_comparison(
                images=images,
                recon_vae=recon_vae,
                recon_diffusion=recon,
                anomaly_maps=anomaly_map,
                save_dir=vis_dir,
                batch_idx=batch_idx,
                category_name=category_name,
                max_images=max_images_per_batch
            )

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
    
    # Save results summary
    results_summary = {
        'category': category_name,
        'image_auroc': img_auroc,
        'pixel_auroc': px_auroc,
        'total_samples': len(labels_all),
        'anomaly_samples': sum(labels_all),
        'normal_samples': len(labels_all) - sum(labels_all)
    }
    
    # Save summary to file
    summary_path = os.path.join(vis_dir, f'{category_name}_results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Category: {category_name}\n")
        f.write(f"Image AUROC: {img_auroc:.4f}\n")
        f.write(f"Pixel AUROC: {px_auroc:.4f}\n")
        f.write(f"Total Samples: {len(labels_all)}\n")
        f.write(f"Anomaly Samples: {sum(labels_all)}\n")
        f.write(f"Normal Samples: {len(labels_all) - sum(labels_all)}\n")
    
    print(f"[INFO] Results summary saved to {summary_path}")
    if save_visualizations:
        print(f"[INFO] Visualizations saved to {vis_dir}")
        # Save summary visualization
        save_summary_visualization(
            labels_all=labels_all,
            img_scores_all=img_scores_all,
            maps_all=maps_all,
            save_dir=vis_dir,
            category_name=category_name
        )


if __name__ == '__main__':
    run_evaluation() 