import os
import torch
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC
from torch.functional import F


def normalize_maps_global(anomaly_maps: List[torch.Tensor]) -> List[torch.Tensor]:
    stacked = torch.cat(anomaly_maps, dim=0)
    amin = stacked.min()
    amax = stacked.max()
    if (amax - amin) < 1e-8:
        return [torch.zeros_like(m) for m in anomaly_maps]
    return [(m - amin) / (amax - amin) for m in anomaly_maps]

def compute_anomaly_map(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return (x - x_recon).abs().mean(dim=1) # Tính mean giữa 3 channels

def eval_auroc_image(labels: List[int], scores: List[float]) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))

def eval_auroc_pixel(anomaly_maps: List[torch.Tensor], gt_masks: List[torch.Tensor]) -> float:
    norm_maps = normalize_maps_global(anomaly_maps)
    pred = torch.cat([m.flatten() for m in norm_maps], dim=0).cpu()
    gt = torch.cat([g.flatten() for g in gt_masks], dim=0).cpu().bool()
    metric = AUROC(task="binary")
    return float(metric(pred, gt))

def calc_image_score(anomaly_maps, image_score_type_name):
    if isinstance(anomaly_maps, list):
        if len(anomaly_maps) == 0:
            raise ValueError("anomaly_maps list is empty")
        first_shape = anomaly_maps[0].shape
        for i, tensor in enumerate(anomaly_maps):
            if tensor.shape != first_shape:
                raise ValueError(f"Tensor {i} has shape {tensor.shape}, expected {first_shape}")

        anomaly_map = torch.stack(anomaly_maps, dim=0)
    else:
        anomaly_map = anomaly_maps
    
    if image_score_type_name == 'max':
        return anomaly_map.amax(dim=(1, 2))
    elif image_score_type_name == 'mean':
        return anomaly_map.float().mean(dim=(1, 2))
    elif image_score_type_name == 'std':
        return anomaly_map.float().std(dim=(1, 2), unbiased=False)
    else:
        raise ValueError(f"Unsupported score type: {image_score_type_name}")



from SSIM_TM import StructuralSimilarityIndexMeasure as SSIM

def calc_ssim(x: torch.Tensor, x_recon: torch.Tensor, kernel_ens: list = [11], sigma_ens: list = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]) -> torch.Tensor:
    """
    Calculate SSIM-based anomaly map between original and reconstructed images.
    
    Args:
        x: Original images tensor of shape [B, C, H, W]
        x_recon: Reconstructed images tensor of shape [B, C, H, W]
        kernel_ens: List of kernel sizes for SSIM ensemble
        sigma_ens: List of sigma values for SSIM ensemble
        
    Returns:
        anomaly_map: SSIM-based anomaly map of shape [B, H, W]
    """
    # Ensure tensors are on the same device and have the same dtype
    x_recon = x_recon.type_as(x)
    
    # SSIM-ens: Create ensemble of SSIM metrics with different parameters
    SSIM_list = []
    for kernel in kernel_ens:
        for sigma in sigma_ens:
            ssim_met = SSIM(return_full_image=True, reduction='none', sigma=sigma, kernel_size=kernel)
            SSIM_list.append(ssim_met)

    # Calculate different SSIMs
    ssims, ssim_aggs = [], []
    for SSIM_ in SSIM_list:  # average over all SSIMs
        ssim_agg, ssim = SSIM_(x_recon, x)
        ssims.append(ssim)
        ssim_aggs.append(ssim_agg)
    
    # Calculate weights based on SSIM values (lower SSIM = higher weight for anomaly detection)
    weightings = [torch.exp(-x) for x in ssims]
    # Normalize weights
    weightings = [x / sum(weightings) for x in weightings]
    # Calculate weighted sum
    ssim = torch.stack(ssims).mul(torch.stack(weightings)).sum(0)
    
    # Calculate the anomaly map: (1 - SSIM) gives higher values for more dissimilar regions
    anomaly_map = 1 - ssim
    
    # Average across channels to get single-channel anomaly map
    anomaly_map = anomaly_map.mean(dim=1)  # [B, H, W]
    
    return anomaly_map


def compute_perceptual_anomaly_map(vae_model, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """
    Compute perceptual anomaly map using VAE encoder features.
    
    Args:
        vae_model: Pre-trained VAE model
        x: Original images tensor of shape [B, C, H, W]
        x_recon: Reconstructed images tensor of shape [B, C, H, W]
        
    Returns:
        anomaly_map: Perceptual anomaly map of shape [B, H, W]
    """
    with torch.no_grad():
        # Get features from VAE encoder (ResNet backbone)
        feat_orig = vae_model.encoder.features(x)  # [B, 512, H', W']
        feat_recon = vae_model.encoder.features(x_recon)  # [B, 512, H', W']
        
        # Compute feature difference
        feat_diff = (feat_orig - feat_recon).abs()
        
        # Average across channels to get single-channel feature map
        feat_anomaly = feat_diff.mean(dim=1)  # [B, H', W']
        
        # Upsample to match original image size
        anomaly_map = F.interpolate(
            feat_anomaly.unsqueeze(1), 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)  # [B, H, W]
        
        return anomaly_map


def compute_multiscale_perceptual_anomaly(vae_model, x: torch.Tensor, x_recon: torch.Tensor, 
                                        feature_weight: float = 1.0, latent_weight: float = 0.1) -> torch.Tensor:
    """
    Multi-scale perceptual anomaly detection using both feature and latent space.
    
    Args:
        vae_model: Pre-trained VAE model
        x: Original images tensor of shape [B, C, H, W]
        x_recon: Reconstructed images tensor of shape [B, C, H, W]
        feature_weight: Weight for feature-level anomaly
        latent_weight: Weight for latent-level anomaly
        
    Returns:
        anomaly_map: Multi-scale perceptual anomaly map of shape [B, H, W]
    """
    with torch.no_grad():
        # Get features at different scales
        feat_orig = vae_model.encoder.features(x)  # [B, 512, H', W']
        feat_recon = vae_model.encoder.features(x_recon)  # [B, 512, H', W']
        
        # Compute anomaly at feature level
        feat_anomaly = (feat_orig - feat_recon).abs().mean(dim=1)  # [B, H', W']
        
        # Also compute at latent space level
        mu_orig, _ = vae_model.encode(x)  # [B, latent_dim]
        mu_recon, _ = vae_model.encode(x_recon)  # [B, latent_dim]
        latent_anomaly = (mu_orig - mu_recon).abs().mean(dim=1)  # [B]
        
        # Combine both (weighted combination)
        combined_anomaly = (feature_weight * feat_anomaly + 
                           latent_weight * latent_anomaly.unsqueeze(-1).unsqueeze(-1))
        
        # Upsample to original size
        anomaly_map = F.interpolate(
            combined_anomaly.unsqueeze(1),
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H, W]
        
        return anomaly_map


def compute_combined_anomaly_map(vae_model, x: torch.Tensor, x_recon: torch.Tensor, 
                               pixel_weight: float = 0.5, perceptual_weight: float = 0.5,
                               use_multiscale: bool = True) -> torch.Tensor:
    """
    Combine pixel-level and perceptual anomaly maps.
    
    Args:
        vae_model: Pre-trained VAE model
        x: Original images tensor of shape [B, C, H, W]
        x_recon: Reconstructed images tensor of shape [B, C, H, W]
        pixel_weight: Weight for pixel-level anomaly (L1)
        perceptual_weight: Weight for perceptual anomaly
        use_multiscale: Whether to use multi-scale perceptual features
        
    Returns:
        anomaly_map: Combined anomaly map of shape [B, H, W]
    """
    with torch.no_grad():
        # Compute pixel-level anomaly (L1)
        pixel_anomaly = compute_anomaly_map(x, x_recon)  # [B, H, W]
        
        # Compute perceptual anomaly
        if use_multiscale:
            perceptual_anomaly = compute_multiscale_perceptual_anomaly(vae_model, x, x_recon)
        else:
            perceptual_anomaly = compute_perceptual_anomaly_map(vae_model, x, x_recon)
        
        # Normalize both maps to [0, 1] range
        pixel_anomaly_norm = (pixel_anomaly - pixel_anomaly.min()) / (pixel_anomaly.max() - pixel_anomaly.min() + 1e-8)
        perceptual_anomaly_norm = (perceptual_anomaly - perceptual_anomaly.min()) / (perceptual_anomaly.max() - perceptual_anomaly.min() + 1e-8)
        
        # Combine with weights
        combined_anomaly = pixel_weight * pixel_anomaly_norm + perceptual_weight * perceptual_anomaly_norm
        
        return combined_anomaly
