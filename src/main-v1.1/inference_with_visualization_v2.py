import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import cv2
from PIL import Image
from config import load_config
from utils import load_mvtec_test_dataset, load_vae, load_diffusion_unet
from reconstruction import Reconstruction
from diffusion_gaussian import GaussianDiffusion
from inference_v2 import (
    compute_anomaly_map,
    eval_auroc_image,
    eval_auroc_pixel,
    to_label_list,
    to_batch_tensor
)


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


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


def create_heatmap(
        anomaly_map: torch.Tensor,
        target_height: int,
        target_width: int
) -> np.ndarray:
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


def overlay_heatmap_on_image(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.6
) -> np.ndarray:
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
        input_images: torch.Tensor,
        vae_reconstructions: torch.Tensor,
        diffusion_reconstructions: torch.Tensor,
        anomaly_maps: torch.Tensor,
        gt_masks: torch.Tensor = None,
        max_images: int = 8
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


def save_visualization_results(
        results: Dict,
        save_dir: str,
        category_name: str,
        batch_idx: int
):
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


@torch.no_grad()
def run_inference_with_visualization():
    """Run inference with comprehensive visualization."""
    config = load_config("config.yml")

    # Set seed
    set_seed(config.general.seed)

    # General settings
    image_size = config.general.image_size
    batch_size = config.general.batch_size
    input_channels = config.general.input_channels
    output_channels = config.general.output_channels
    device = torch.device(f"cuda:{config.general.cuda}" if torch.cuda.is_available() else "cpu")

    # Data settings
    category_name = config.data.category
    mvtec_data_dir = config.data.mvtec_data_dir

    # VAE settings
    vae_name = config.vae_model.name
    z_dim = config.vae_model.z_dim
    dropout_p = config.vae_model.dropout_p
    backbone = config.vae_model.backbone

    # Diffusion settings
    num_timesteps = config.diffusion_model.num_timesteps
    beta_schedule = config.diffusion_model.beta_schedule
    w = config.diffusion_model.w

    # Testing paths
    diff_test_cfg = config.diffusion_testing
    diffusion_model_path = diff_test_cfg.diffusion_model_path
    vae_model_path = diff_test_cfg.diffusion_vae_model_path

    # Visualization settings
    save_visualizations = getattr(diff_test_cfg, 'save_visualizations', True)
    max_visualization_batches = getattr(diff_test_cfg, 'max_visualization_batches', 5)
    max_images_per_batch = getattr(diff_test_cfg, 'max_images_per_batch', 8)

    # Create output directories
    output_base_dir = os.path.join(diff_test_cfg.diffusion_test_result_base_dir, category_name)
    visualization_dir = os.path.join(output_base_dir, 'visualizations')
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"[INFO] Running inference with visualization for category: {category_name}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {output_base_dir}")

    # Load models
    print("[INFO] Loading VAE model...")
    vae_model = load_vae(
        checkpoint_path=vae_model_path,
        vae_name=vae_name,
        input_channels=input_channels,
        output_channels=output_channels,
        z_dim=z_dim,
        backbone=backbone,
        dropout_p=dropout_p,
        image_size=image_size,
        device=device
    )

    print("[INFO] Loading Diffusion model...")
    diffusion_model = load_diffusion_unet(
        checkpoint_path=diffusion_model_path,
        image_size=image_size,
        in_channels=input_channels,
        device=device
    )

    gaussian_diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule='linear')

    # Initialize reconstruction
    reconstruction = Reconstruction(diffusion_model, gaussian_diffusion, device)

    # Load test dataset
    print("[INFO] Loading test dataset...")
    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=mvtec_data_dir,
        category=category_name,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # Accumulators for evaluation
    labels_all: List[int] = []
    img_scores_all: List[float] = []
    maps_all: List[torch.Tensor] = []
    gts_all: List[torch.Tensor] = []

    # Visualization accumulators
    all_visualization_results = []

    print("[INFO] Starting inference...")
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)  # [B,1,H,W]
        labels = batch['label']

        print(f"\n[DEBUG] Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Labels: {labels}")

        # VAE reconstruction
        recon_vae, _, _ = vae_model(images)
        print(f"  VAE reconstruction shape: {recon_vae.shape}")

        # Diffusion reconstruction using guided reconstruction
        images_reconstructed = reconstruction(
            x=recon_vae,
            y0=images,
            w=w
        )

        # Convert list to tensor
        images_reconstructed = to_batch_tensor(images_reconstructed, images)
        print(f"  Diffusion reconstruction shape: {images_reconstructed.shape}")

        # Compute anomaly map and scores
        B = images.size(0)
        anomaly_map = compute_anomaly_map(images, images_reconstructed)  # [B,H,W] or [B,1,H,W]

        # Handle different anomaly map shapes
        if anomaly_map.dim() == 4 and anomaly_map.shape[1] == 1:
            anomaly_map = anomaly_map.squeeze(1)  # [B,H,W]

        print(f"  Anomaly map shape: {anomaly_map.shape}")

        # Calculate image scores
        # image_scores = anomaly_map.float().mean(dim=(1, 2))
        image_scores = anomaly_map.float().std(dim=(1, 2), unbiased=False)

        # Accumulate evaluation results
        labels_all.extend(to_label_list(labels))
        img_scores_all.extend(image_scores.detach().cpu().tolist())
        maps_all.extend(list(anomaly_map.detach().cpu()))
        gts_all.extend(list(masks.detach().cpu().squeeze(1)))

        # Prepare visualization data for this batch
        if batch_idx < max_visualization_batches:
            batch_results = {
                'input_images': [],
                'vae_reconstructions': [],
                'diffusion_reconstructions': [],
                'heatmaps': [],
                'overlays': [],
                'gt_masks': []
            }

            for i in range(min(B, max_images_per_batch)):
                # Convert tensors to numpy for visualization
                input_img = tensor_to_numpy(images[i])
                vae_img = tensor_to_numpy(recon_vae[i])
                diff_img = tensor_to_numpy(images_reconstructed[i])

                # Get target dimensions from input image
                target_height, target_width = input_img.shape[:2]

                # Ensure all images have same size
                if vae_img.shape[:2] != (target_height, target_width):
                    vae_img = cv2.resize(vae_img, (target_width, target_height))
                if diff_img.shape[:2] != (target_height, target_width):
                    diff_img = cv2.resize(diff_img, (target_width, target_height))

                # Create heatmap with consistent size
                heatmap = create_heatmap(anomaly_map[i], target_height, target_width)

                # Create overlay
                overlay = overlay_heatmap_on_image(input_img, heatmap)

                # Process ground truth mask
                gt_mask = tensor_to_numpy(masks[i]) if masks is not None else None
                if gt_mask is not None and gt_mask.shape[:2] != (target_height, target_width):
                    gt_mask = cv2.resize(gt_mask, (target_width, target_height))

                # Store results
                batch_results['input_images'].append(input_img)
                batch_results['vae_reconstructions'].append(vae_img)
                batch_results['diffusion_reconstructions'].append(diff_img)
                batch_results['heatmaps'].append(heatmap)
                batch_results['overlays'].append(overlay)
                if gt_mask is not None:
                    batch_results['gt_masks'].append(gt_mask)

            all_visualization_results.append(batch_results)

            # Save individual visualizations
            if save_visualizations:
                batch_vis_dir = os.path.join(visualization_dir, f'batch_{batch_idx:03d}')
                save_visualization_results(batch_results, batch_vis_dir, category_name, batch_idx)

    # Calculate evaluation metrics
    print("\n[INFO] Calculating evaluation metrics...")
    img_auroc = eval_auroc_image(labels_all, img_scores_all)
    px_auroc = eval_auroc_pixel(maps_all, gts_all)

    print(f"\n[RESULTS] Image AUROC: {img_auroc:.4f}")
    print(f"[RESULTS] Pixel AUROC: {px_auroc:.4f}")

    # Create comprehensive visualization grid
    print("\n[INFO] Creating visualization grid...")
    if all_visualization_results:
        # Create a sample visualization with first batch
        first_batch = all_visualization_results[0]
        n_samples = len(first_batch['input_images'])

        fig, axes = plt.subplots(n_samples, 6, figsize=(18, n_samples * 3))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Input image
            axes[i, 0].imshow(first_batch['input_images'][i],
                              cmap='gray' if len(first_batch['input_images'][i].shape) == 2 else None)
            axes[i, 0].set_title('Input Image', fontsize=10)
            axes[i, 0].axis('off')

            # VAE reconstruction
            axes[i, 1].imshow(first_batch['vae_reconstructions'][i],
                              cmap='gray' if len(first_batch['vae_reconstructions'][i].shape) == 2 else None)
            axes[i, 1].set_title('VAE Reconstruction', fontsize=10)
            axes[i, 1].axis('off')

            # Diffusion reconstruction
            axes[i, 2].imshow(first_batch['diffusion_reconstructions'][i],
                              cmap='gray' if len(first_batch['diffusion_reconstructions'][i].shape) == 2 else None)
            axes[i, 2].set_title('Diffusion Reconstruction', fontsize=10)
            axes[i, 2].axis('off')

            # Anomaly heatmap
            axes[i, 3].imshow(first_batch['heatmaps'][i])
            axes[i, 3].set_title('Anomaly Heatmap', fontsize=10)
            axes[i, 3].axis('off')

            # Overlay
            axes[i, 4].imshow(first_batch['overlays'][i])
            axes[i, 4].set_title('Overlay', fontsize=10)
            axes[i, 4].axis('off')

            # Ground truth mask
            if first_batch['gt_masks'] and i < len(first_batch['gt_masks']):
                axes[i, 5].imshow(first_batch['gt_masks'][i], cmap='gray')
                axes[i, 5].set_title('Ground Truth', fontsize=10)
            else:
                axes[i, 5].text(0.5, 0.5, 'No GT Mask', ha='center', va='center')
                axes[i, 5].set_title('Ground Truth', fontsize=10)
            axes[i, 5].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_base_dir, f'{category_name}_comprehensive_visualization.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Save evaluation results
    results_summary = {
        'category': category_name,
        'image_auroc': img_auroc,
        'pixel_auroc': px_auroc,
        'total_samples': len(labels_all),
        'config': {
            'image_size': image_size,
            'batch_size': batch_size,
            'num_timesteps': num_timesteps,
            'beta_schedule': beta_schedule,
            'w': w
        }
    }

    # Save results to file
    import json
    with open(os.path.join(output_base_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[INFO] Results saved to: {output_base_dir}")
    print(f"[INFO] Visualizations saved to: {visualization_dir}")

    return results_summary


if __name__ == "__main__":
    results = run_inference_with_visualization()
    print(f"\n[FINAL RESULTS] {results['category']}:")
    print(f"  Image AUROC: {results['image_auroc']:.4f}")
    print(f"  Pixel AUROC: {results['pixel_auroc']:.4f}")
    print(f"  Total samples: {results['total_samples']}")

# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from typing import List, Tuple, Dict
# import cv2
# from PIL import Image
# from config import load_config
# from utils import load_mvtec_test_dataset, load_vae, load_diffusion_unet
# from reconstruction import Reconstruction
# from diffusion_gaussian import GaussianDiffusion
# from inference_v2 import (
#     compute_anomaly_map,
#     eval_auroc_image,
#     eval_auroc_pixel,
#     to_label_list,
#     to_batch_tensor
# )
#
#
# def set_seed(seed):
#     """Set random seed for reproducibility."""
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#
# def normalize_image(image: torch.Tensor) -> torch.Tensor:
#     """Normalize image to [0, 1] range."""
#     if image.min() < 0 or image.max() > 1:
#         image = (image - image.min()) / (image.max() - image.min())
#     return image
#
#
# def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
#     """Convert tensor to numpy array for visualization."""
#     if tensor.dim() == 4:  # [B, C, H, W]
#         tensor = tensor.squeeze(0)  # Remove batch dimension
#     if tensor.dim() == 3:  # [C, H, W]
#         tensor = tensor.permute(1, 2, 0)  # [H, W, C]
#
#     # Convert to numpy and ensure proper range
#     img = tensor.detach().cpu().numpy()
#     img = normalize_image(torch.tensor(img)).numpy()
#
#     # Clip to valid range
#     img = np.clip(img, 0, 1)
#     return img
#
#
# def create_heatmap(anomaly_map: torch.Tensor, original_size: Tuple[int, int] = None) -> np.ndarray:
#     """Create heatmap from anomaly map."""
#     # Convert to numpy
#     heatmap = anomaly_map.detach().cpu().numpy()
#
#     # Normalize to [0, 1]
#     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
#
#     # Apply colormap
#     heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
#
#     # Resize if needed
#     if original_size:
#         heatmap_colored = cv2.resize(heatmap_colored, original_size)
#
#     return heatmap_colored
#
#
# def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
#     """Overlay heatmap on original image."""
#     # Ensure both are in same range [0, 1]
#     image = np.clip(image, 0, 1)
#     heatmap = np.clip(heatmap, 0, 1)
#
#     # Overlay
#     overlay = alpha * heatmap + (1 - alpha) * image
#     return np.clip(overlay, 0, 1)
#
#
# def create_visualization_grid(
#     input_images: torch.Tensor,
#     vae_reconstructions: torch.Tensor,
#     diffusion_reconstructions: torch.Tensor,
#     anomaly_maps: torch.Tensor,
#     gt_masks: torch.Tensor = None,
#     max_images: int = 8
# ) -> np.ndarray:
#     """Create a grid visualization of all components."""
#     B = min(input_images.size(0), max_images)
#
#     # Number of columns (input, vae, diffusion, heatmap, overlay, gt_mask if available)
#     n_cols = 5 if gt_masks is not None else 4
#     n_rows = B
#
#     # Create figure
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
#     if n_rows == 1:
#         axes = axes.reshape(1, -1)
#
#     for i in range(B):
#         # Input image
#         input_img = tensor_to_numpy(input_images[i])
#         axes[i, 0].imshow(input_img)
#         axes[i, 0].set_title('Input Image', fontsize=10)
#         axes[i, 0].axis('off')
#
#         # VAE reconstruction
#         vae_img = tensor_to_numpy(vae_reconstructions[i])
#         axes[i, 1].imshow(vae_img)
#         axes[i, 1].set_title('VAE Reconstruction', fontsize=10)
#         axes[i, 1].axis('off')
#
#         # Diffusion reconstruction
#         diff_img = tensor_to_numpy(diffusion_reconstructions[i])
#         axes[i, 2].imshow(diff_img)
#         axes[i, 2].set_title('Diffusion Reconstruction', fontsize=10)
#         axes[i, 2].axis('off')
#
#         # Heatmap
#         heatmap = create_heatmap(anomaly_maps[i])
#         axes[i, 3].imshow(heatmap)
#         axes[i, 3].set_title('Anomaly Heatmap', fontsize=10)
#         axes[i, 3].axis('off')
#
#         # Overlay
#         if gt_masks is not None:
#             overlay = overlay_heatmap_on_image(input_img, heatmap)
#             axes[i, 4].imshow(overlay)
#             axes[i, 4].set_title('Overlay', fontsize=10)
#             axes[i, 4].axis('off')
#
#     plt.tight_layout()
#     return fig
#
#
# def save_visualization_results(
#     results: Dict,
#     save_dir: str,
#     category_name: str,
#     batch_idx: int
# ):
#     """Save visualization results to files."""
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Save individual components
#     for i, (input_img, vae_img, diff_img, heatmap, overlay) in enumerate(zip(
#         results['input_images'],
#         results['vae_reconstructions'],
#         results['diffusion_reconstructions'],
#         results['heatmaps'],
#         results['overlays']
#     )):
#         # Convert to PIL and save
#         input_pil = Image.fromarray((input_img * 255).astype(np.uint8))
#         vae_pil = Image.fromarray((vae_img * 255).astype(np.uint8))
#         diff_pil = Image.fromarray((diff_img * 255).astype(np.uint8))
#         heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
#         overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
#
#         # Save files
#         input_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_input.png'))
#         vae_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_vae.png'))
#         diff_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_diffusion.png'))
#         heatmap_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_heatmap.png'))
#         overlay_pil.save(os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{i:02d}_overlay.png'))
#
#
# @torch.no_grad()
# def run_inference_with_visualization():
#     """Run inference with comprehensive visualization."""
#     config = load_config("config.yml")
#
#     # Set seed
#     set_seed(config.general.seed)
#
#     # General settings
#     image_size = config.general.image_size
#     batch_size = config.general.batch_size
#     input_channels = config.general.input_channels
#     output_channels = config.general.output_channels
#     device = torch.device(f"cuda:{config.general.cuda}" if torch.cuda.is_available() else "cpu")
#
#     # Data settings
#     category_name = config.data.category
#     mvtec_data_dir = config.data.mvtec_data_dir
#
#     # VAE settings
#     vae_name = config.vae_model.name
#     z_dim = config.vae_model.z_dim
#     dropout_p = config.vae_model.dropout_p
#     backbone = config.vae_model.backbone
#
#     # Diffusion settings
#     num_timesteps = config.diffusion_model.num_timesteps
#     beta_schedule = config.diffusion_model.beta_schedule
#     w = config.diffusion_model.w
#
#     # Testing paths
#     diff_test_cfg = config.diffusion_testing
#     diffusion_model_path = diff_test_cfg.diffusion_model_path
#     vae_model_path = diff_test_cfg.diffusion_vae_model_path
#
#     # Visualization settings
#     save_visualizations = getattr(diff_test_cfg, 'save_visualizations', True)
#     max_visualization_batches = getattr(diff_test_cfg, 'max_visualization_batches', 5)
#     max_images_per_batch = getattr(diff_test_cfg, 'max_images_per_batch', 8)
#
#     # Create output directories
#     output_base_dir = os.path.join(diff_test_cfg.diffusion_test_result_base_dir, category_name)
#     visualization_dir = os.path.join(output_base_dir, 'visualizations')
#     os.makedirs(output_base_dir, exist_ok=True)
#     os.makedirs(visualization_dir, exist_ok=True)
#
#     print(f"[INFO] Running inference with visualization for category: {category_name}")
#     print(f"[INFO] Device: {device}")
#     print(f"[INFO] Output directory: {output_base_dir}")
#
#     # Load models
#     print("[INFO] Loading VAE model...")
#     vae_model = load_vae(
#         checkpoint_path=vae_model_path,
#         vae_name=vae_name,
#         input_channels=input_channels,
#         output_channels=output_channels,
#         z_dim=z_dim,
#         backbone=backbone,
#         dropout_p=dropout_p,
#         image_size=image_size,
#         device=device
#     )
#
#     print("[INFO] Loading Diffusion model...")
#     diffusion_model = load_diffusion_unet(
#         checkpoint_path=diffusion_model_path,
#         image_size=image_size,
#         in_channels=input_channels,
#         device=device
#     )
#
#     gaussian_diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule='linear')
#     # Initialize reconstruction
#     reconstruction = Reconstruction(diffusion_model, gaussian_diffusion, device)
#
#     # Load test dataset
#     print("[INFO] Loading test dataset...")
#     test_loader = load_mvtec_test_dataset(
#         dataset_root_dir=mvtec_data_dir,
#         category=category_name,
#         image_size=image_size,
#         batch_size=batch_size,
#         shuffle=False,
#     )
#
#     # Accumulators for evaluation
#     labels_all: List[int] = []
#     img_scores_all: List[float] = []
#     maps_all: List[torch.Tensor] = []
#     gts_all: List[torch.Tensor] = []
#
#     # Visualization accumulators
#     all_visualization_results = []
#
#     print("[INFO] Starting inference...")
#     for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
#         images = batch['image'].to(device)
#         masks = batch['mask'].to(device)  # [B,1,H,W]
#         labels = batch['label']
#
#         # VAE reconstruction
#         recon_vae, _, _ = vae_model(images)
#
#         # Diffusion reconstruction using guided reconstruction
#         images_reconstructed = reconstruction(
#             x=recon_vae,
#             y0=images,
#             w=w
#         )
#
#         # Convert list to tensor
#         images_reconstructed = to_batch_tensor(images_reconstructed, images)
#
#         # Compute anomaly map and scores
#         B = images.size(0)
#         anomaly_map = compute_anomaly_map(images, images_reconstructed)  # [B,H,W]
#
#         ############
#         # image_scores = anomaly_map.view(B, -1).max(dim=1).values
#         image_scores = anomaly_map.float().mean(dim=(1, 2))
#
#         # Accumulate evaluation results
#         labels_all.extend(to_label_list(labels))
#         img_scores_all.extend(image_scores.detach().cpu().tolist())
#         maps_all.extend(list(anomaly_map.detach().cpu()))
#         gts_all.extend(list(masks.detach().cpu().squeeze(1)))
#
#         # Prepare visualization data for this batch
#         if batch_idx < max_visualization_batches:
#             batch_results = {
#                 'input_images': [],
#                 'vae_reconstructions': [],
#                 'diffusion_reconstructions': [],
#                 'heatmaps': [],
#                 'overlays': [],
#                 'gt_masks': []
#             }
#
#             for i in range(min(B, max_images_per_batch)):
#                 # Convert tensors to numpy for visualization
#                 input_img = tensor_to_numpy(images[i])
#                 vae_img = tensor_to_numpy(recon_vae[i])
#                 diff_img = tensor_to_numpy(images_reconstructed[i])
#                 heatmap = create_heatmap(anomaly_map[i])
#                 overlay = overlay_heatmap_on_image(input_img, heatmap)
#                 gt_mask = tensor_to_numpy(masks[i]) if masks is not None else None
#
#                 batch_results['input_images'].append(input_img)
#                 batch_results['vae_reconstructions'].append(vae_img)
#                 batch_results['diffusion_reconstructions'].append(diff_img)
#                 batch_results['heatmaps'].append(heatmap)
#                 batch_results['overlays'].append(overlay)
#                 if gt_mask is not None:
#                     batch_results['gt_masks'].append(gt_mask)
#
#             all_visualization_results.append(batch_results)
#
#             # Save individual visualizations
#             if save_visualizations:
#                 batch_vis_dir = os.path.join(visualization_dir, f'batch_{batch_idx:03d}')
#                 save_visualization_results(batch_results, batch_vis_dir, category_name, batch_idx)
#
#     # Calculate evaluation metrics
#     print("\n[INFO] Calculating evaluation metrics...")
#     img_auroc = eval_auroc_image(labels_all, img_scores_all)
#     px_auroc = eval_auroc_pixel(maps_all, gts_all)
#
#     print(f"\n[RESULTS] Image AUROC: {img_auroc:.4f}")
#     print(f"[RESULTS] Pixel AUROC: {px_auroc:.4f}")
#
#     # Create comprehensive visualization grid
#     print("\n[INFO] Creating visualization grid...")
#     if all_visualization_results:
#         # Create a large grid with all batches
#         total_images = sum(len(batch['input_images']) for batch in all_visualization_results)
#         n_cols = 5  # input, vae, diffusion, heatmap, overlay
#         n_rows = (total_images + n_cols - 1) // n_cols
#
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
#         if n_rows == 1:
#             axes = axes.reshape(1, -1)
#
#         img_idx = 0
#         for batch_idx, batch_results in enumerate(all_visualization_results):
#             for sample_idx in range(len(batch_results['input_images'])):
#                 row = img_idx // n_cols
#                 col = img_idx % n_cols
#
#                 if col == 0:
#                     axes[row, col].imshow(batch_results['input_images'][sample_idx])
#                     axes[row, col].set_title(f'Input (Batch {batch_idx}, Sample {sample_idx})', fontsize=8)
#                 elif col == 1:
#                     axes[row, col].imshow(batch_results['vae_reconstructions'][sample_idx])
#                     axes[row, col].set_title('VAE Reconstruction', fontsize=8)
#                 elif col == 2:
#                     axes[row, col].imshow(batch_results['diffusion_reconstructions'][sample_idx])
#                     axes[row, col].set_title('Diffusion Reconstruction', fontsize=8)
#                 elif col == 3:
#                     axes[row, col].imshow(batch_results['heatmaps'][sample_idx])
#                     axes[row, col].set_title('Anomaly Heatmap', fontsize=8)
#                 elif col == 4:
#                     axes[row, col].imshow(batch_results['overlays'][sample_idx])
#                     axes[row, col].set_title('Overlay', fontsize=8)
#
#                 axes[row, col].axis('off')
#                 img_idx += 1
#
#         # Hide empty subplots
#         for i in range(img_idx, n_rows * n_cols):
#             row = i // n_cols
#             col = i % n_cols
#             axes[row, col].axis('off')
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_base_dir, f'{category_name}_comprehensive_visualization.png'),
#                    dpi=300, bbox_inches='tight')
#         plt.close()
#
#     # Save evaluation results
#     results_summary = {
#         'category': category_name,
#         'image_auroc': img_auroc,
#         'pixel_auroc': px_auroc,
#         'total_samples': len(labels_all),
#         'config': {
#             'image_size': image_size,
#             'batch_size': batch_size,
#             'num_timesteps': num_timesteps,
#             'beta_schedule': beta_schedule,
#             'w': w
#         }
#     }
#
#     # Save results to file
#     import json
#     with open(os.path.join(output_base_dir, 'evaluation_results.json'), 'w') as f:
#         json.dump(results_summary, f, indent=2)
#
#     print(f"\n[INFO] Results saved to: {output_base_dir}")
#     print(f"[INFO] Visualizations saved to: {visualization_dir}")
#
#     return results_summary
#
#
# if __name__ == "__main__":
#     results = run_inference_with_visualization()
#     print(f"\n[FINAL RESULTS] {results['category']}:")
#     print(f"  Image AUROC: {results['image_auroc']:.4f}")
#     print(f"  Pixel AUROC: {results['pixel_auroc']:.4f}")
#     print(f"  Total samples: {results['total_samples']}")
