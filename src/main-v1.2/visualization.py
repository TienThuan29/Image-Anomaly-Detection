import os
from typing import Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    if image.min() < 0 or image.max() > 1:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return np.clip(image, 0, 1)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization."""
    # Handle different tensor dimensions
    if tensor.dim() == 4:  # [B, C, H, W]
        tensor = tensor.squeeze(0)  # Remove batch dimension

    if tensor.dim() == 3:  # [C, H, W]
        if tensor.shape[0] in [1, 3]:  # Channel first
            tensor = tensor.permute(1, 2, 0)  # [H, W, C]
    elif tensor.dim() == 2:  # [H, W] - grayscale
        tensor = tensor.unsqueeze(-1)  # [H, W, 1]

    # Convert to numpy
    img = tensor.detach().cpu().numpy()

    # Handle grayscale case
    if img.shape[-1] == 1:
        img = img.squeeze(-1)  # Remove single channel dimension

    # Normalize to [0, 1]
    img = normalize_image(img)

    return img

def create_heatmap(anomaly_map: torch.Tensor, target_height: int, target_width: int) -> np.ndarray:
    """Create heatmap from anomaly map with consistent size."""
    # Convert to numpy
    if anomaly_map.dim() == 3:  # [B,H,W] or [C,H,W]
        if anomaly_map.shape[0] == 1:  # Single batch or single channel
            heatmap = anomaly_map.squeeze(0).detach().cpu().numpy()
        else:  # Multiple batches/channels - take first
            heatmap = anomaly_map[0].detach().cpu().numpy()
    elif anomaly_map.dim() == 2:  # [H,W]
        heatmap = anomaly_map.detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected anomaly_map dimensions: {anomaly_map.shape}")

    # Normalize to [0, 1]
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = np.zeros_like(heatmap)

    # Resize to match target shape (width, height for cv2)
    if heatmap.shape != (target_height, target_width):
        heatmap = cv2.resize(heatmap, (target_width, target_height))

    # Apply colormap
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]  # Remove alpha channel

    return heatmap_colored


def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Overlay heatmap on original image."""
    # Ensure both images have same dimensions
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Handle grayscale input image
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)  # Convert to RGB
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)  # Convert to RGB

    # Ensure both are in same range [0, 1]
    image = normalize_image(image)
    heatmap = normalize_image(heatmap)

    # Create overlay
    overlay = alpha * heatmap + (1 - alpha) * image
    return np.clip(overlay, 0, 1)


def create_visualization_grid(
        input_images: torch.Tensor, vae_reconstructions: torch.Tensor,
        diffusion_reconstructions: torch.Tensor, anomaly_maps: torch.Tensor,
        gt_masks: torch.Tensor = None, max_images: int = 8
) -> plt.Figure:
    """Create a grid visualization of all components."""
    B = min(input_images.size(0), max_images)

    # Number of columns (input, vae, diffusion, heatmap, overlay, gt_mask if available)
    n_cols = 6 if gt_masks is not None else 5
    n_rows = B

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(B):
        # Convert input image
        input_img = tensor_to_numpy(input_images[i])
        target_height, target_width = input_img.shape[:2]

        # Input image
        axes[i, 0].imshow(input_img, cmap='gray' if len(input_img.shape) == 2 else None)
        axes[i, 0].set_title('Input Image', fontsize=10)
        axes[i, 0].axis('off')

        # VAE reconstruction
        vae_img = tensor_to_numpy(vae_reconstructions[i])
        # Resize if needed
        if vae_img.shape[:2] != (target_height, target_width):
            vae_img = cv2.resize(vae_img, (target_width, target_height))
        axes[i, 1].imshow(vae_img, cmap='gray' if len(vae_img.shape) == 2 else None)
        axes[i, 1].set_title('VAE Reconstruction', fontsize=10)
        axes[i, 1].axis('off')

        # Diffusion reconstruction
        diff_img = tensor_to_numpy(diffusion_reconstructions[i])
        # Resize if needed
        if diff_img.shape[:2] != (target_height, target_width):
            diff_img = cv2.resize(diff_img, (target_width, target_height))
        axes[i, 2].imshow(diff_img, cmap='gray' if len(diff_img.shape) == 2 else None)
        axes[i, 2].set_title('Diffusion Reconstruction', fontsize=10)
        axes[i, 2].axis('off')

        # Heatmap
        heatmap = create_heatmap(anomaly_maps[i], target_height, target_width)
        axes[i, 3].imshow(heatmap)
        axes[i, 3].set_title('Anomaly Heatmap', fontsize=10)
        axes[i, 3].axis('off')

        # Overlay
        overlay = overlay_heatmap_on_image(input_img, heatmap)
        axes[i, 4].imshow(overlay)
        axes[i, 4].set_title('Overlay', fontsize=10)
        axes[i, 4].axis('off')

        # Ground truth mask (if available)
        if gt_masks is not None:
            gt_mask = tensor_to_numpy(gt_masks[i])
            if gt_mask.shape[:2] != (target_height, target_width):
                gt_mask = cv2.resize(gt_mask, (target_width, target_height))
            axes[i, 5].imshow(gt_mask, cmap='gray')
            axes[i, 5].set_title('Ground Truth Mask', fontsize=10)
            axes[i, 5].axis('off')

    plt.tight_layout()
    return fig


def save_visualization_results(results: Dict, save_dir: str, category_name: str, batch_idx: int):
    """Save visualization results to files."""
    os.makedirs(save_dir, exist_ok=True)

    # Save individual components
    for i, (input_img, vae_img, diff_img, heatmap, overlay) in enumerate(zip(
            results['input_images'],
            results['vae_reconstructions'],
            results['diffusion_reconstructions'],
            results['heatmaps'],
            results['overlays']
    )):
        # Convert to PIL and save
        def to_pil(img_array):
            if len(img_array.shape) == 2:  # Grayscale
                return Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
            else:  # RGB
                return Image.fromarray((img_array * 255).astype(np.uint8), mode='RGB')

        input_pil = to_pil(input_img)
        vae_pil = to_pil(vae_img)
        diff_pil = to_pil(diff_img)
        heatmap_pil = to_pil(heatmap)
        overlay_pil = to_pil(overlay)

        # Save files
        input_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_input.png'))
        vae_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_vae.png'))
        diff_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_diffusion.png'))
        heatmap_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_heatmap.png'))
        overlay_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_overlay.png'))

        # Save ground truth if available
        if 'gt_masks' in results and i < len(results['gt_masks']):
            gt_pil = to_pil(results['gt_masks'][i])
            gt_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_gt_mask.png'))
