from tqdm import tqdm
from typing import List
from config import load_config
from utils import load_mvtec_test_dataset, load_vae, load_diffusion_unet
from diffusion_gaussian import GaussianDiffusion
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC
from visualization import *
import json

config = load_config()

_seed = config.general.seed
_image_size = config.general.image_size
_batch_size = config.general.batch_size
_input_channels = config.general.input_channels
_output_channels = config.general.output_channels
_device = torch.device(f"cuda:{config.general.cuda}" if torch.cuda.is_available() else "cpu")

# Data settings
_category_name = config.data.category
_mvtec_data_dir = config.data.mvtec_data_dir

# VAE settings
_vae_name = config.vae_model.name
_z_dim = config.vae_model.z_dim
_dropout_p = config.vae_model.dropout_p
_backbone = config.vae_model.backbone

# Diffusion settings
_num_timesteps = config.diffusion_model.num_timesteps
_beta_schedule = config.diffusion_model.beta_schedule

# Image score type name
_image_score_type_name = config.diffusion_testing.image_score_type_name

# Testing paths
_diff_test_cfg = config.diffusion_testing
_diffusion_model_path = _diff_test_cfg.diffusion_model_path
_vae_model_path = _diff_test_cfg.diffusion_vae_model_path

# Visualization settings
save_visualizations = getattr(_diff_test_cfg, 'save_visualizations', True)
max_visualization_batches = getattr(_diff_test_cfg, 'max_visualization_batches', 5)
max_images_per_batch = getattr(_diff_test_cfg, 'max_images_per_batch', 8)


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def to_batch_tensor(x_like, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(x_like, torch.Tensor):
        t = x_like
    elif isinstance(x_like, (list, tuple)):
        if len(x_like) == 0:
            raise ValueError("images_reconstructed is empty.")
        first = x_like[0]
        if isinstance(first, torch.Tensor) and first.dim() == 4:
            # list các bước [K,B,C,H,W] -> average theo bước
            t = torch.stack(x_like, dim=0).mean(dim=0)
        elif isinstance(first, torch.Tensor) and first.dim() == 3:
            # list theo batch [B*[C,H,W]] -> [B,C,H,W]
            t = torch.stack(x_like, dim=0)
        else:
            raise TypeError(f"Unsupported element in list: dim={getattr(first,'dim',lambda:None)()}")
    else:
        raise TypeError(f"Unsupported type for images_reconstructed: {type(x_like)}")
    return t.to(device=ref.device, dtype=ref.dtype)


def compute_anomaly_map(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return (x - x_recon).abs().mean(dim=1) # Tính mean giữa 3 channels


def to_label_list(labels_field) -> List[int]:
    if isinstance(labels_field, list):
        return [int(x) for x in labels_field]
    if torch.is_tensor(labels_field):
        if labels_field.ndim == 0:
            return [int(labels_field.item())]
        return [int(x) for x in labels_field.tolist()]
    return [int(labels_field)]


def normalize_maps_global(anomaly_maps: List[torch.Tensor]) -> List[torch.Tensor]:
    stacked = torch.cat(anomaly_maps, dim=0)
    amin = stacked.min()
    amax = stacked.max()
    if (amax - amin) < 1e-8:
        return [torch.zeros_like(m) for m in anomaly_maps]
    return [(m - amin) / (amax - amin) for m in anomaly_maps]


def eval_auroc_image(labels: List[int], scores: List[float]) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def eval_auroc_pixel(anomaly_maps: List[torch.Tensor], gt_masks: List[torch.Tensor]) -> float:
    norm_maps = normalize_maps_global(anomaly_maps)
    pred = torch.cat([m.flatten() for m in norm_maps], dim=0).cpu()
    gt = torch.cat([g.flatten() for g in gt_masks], dim=0).cpu().bool()
    metric = AUROC(task="binary")
    return float(metric(pred, gt))


def calc_image_score(anomaly_map):
    if _image_score_type_name == 'max':
        return anomaly_map.amax(dim=(1, 2))
    elif _image_score_type_name == 'mean':
        return anomaly_map.float().mean(dim=(1, 2))
    elif _image_score_type_name == 'std':
        return anomaly_map.float().std(dim=(1, 2), unbiased=False)
    else:
        raise ValueError(f"Unsupported score type: {_image_score_type_name}")

@torch.no_grad()
def run_inference_with_visualization():
    set_seed(_seed)

    # Create output dir
    output_base_dir = os.path.join(_diff_test_cfg.diffusion_test_result_base_dir, _category_name)
    visualization_dir = os.path.join(output_base_dir, 'visualizations')
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"[INFO] Running inference, category: {_category_name}")
    print(f"[INFO] Device: {_device}")
    print(f"[INFO] Output dir: {output_base_dir}")

    vae_model = load_vae(
        checkpoint_path=_vae_model_path,
        vae_name=_vae_name,
        input_channels=_input_channels,
        output_channels=_output_channels,
        z_dim=_z_dim,
        backbone=_backbone,
        dropout_p=_dropout_p,
        image_size=_image_size,
        device=_device
    )

    diffusion_model = load_diffusion_unet(
        checkpoint_path=_diffusion_model_path,
        image_size=_image_size,
        in_channels=_input_channels,
        device=_device
    )

    gaussian_diffusion = GaussianDiffusion(num_timesteps=_num_timesteps, beta_schedule=_beta_schedule)

    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=_mvtec_data_dir,
        category=_category_name,
        image_size=_image_size,
        batch_size=_batch_size
    )

    labels_all: List[int] = []
    img_scores_all: List[float] = []
    maps_all: List[torch.Tensor] = []
    gts_all: List[torch.Tensor] = []
    all_visualization_results = []

    print("Starting inference...")
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
        images = batch['image'].to(_device)
        masks = batch['mask'].to(_device)  # [B,1,H,W]
        labels = batch['label']

        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Labels: {labels}")

        # vae reconstruction
        vae_reconstructed, _, _ = vae_model(images)

        B = images.size(0)
        t = torch.randint(300, _num_timesteps, (B,), device=_device)
        noise = torch.randn_like(vae_reconstructed).to(_device)
        # x_t: add noise
        x_t = gaussian_diffusion.q_sample(vae_reconstructed, t, noise)

        images_reconstructed = diffusion_model(x_t, t)
        anomaly_map = compute_anomaly_map(images, images_reconstructed)  # [B,H,W] or [B,1,H,W]

        # Handle different anomaly map shapes
        if anomaly_map.dim() == 4 and anomaly_map.shape[1] == 1:
            anomaly_map = anomaly_map.squeeze(1)  # [B,H,W]

        # Calculate image scores
        image_scores = calc_image_score(anomaly_map)

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
                vae_img = tensor_to_numpy(vae_reconstructed[i])
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
                save_visualization_results(batch_results, batch_vis_dir, _category_name, batch_idx)

    # Calculate evaluation metrics
    print("\nCalculating evaluation metrics...")
    img_auroc = eval_auroc_image(labels_all, img_scores_all)
    px_auroc = eval_auroc_pixel(maps_all, gts_all)

    print(f"\n[RESULTS] Image AUROC: {img_auroc:.4f}")
    print(f"[RESULTS] Pixel AUROC: {px_auroc:.4f}")

    # Create comprehensive visualization grid
    print("\nCreating visualization grid...")
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
        plt.savefig(os.path.join(output_base_dir, f'{_category_name}_comprehensive_visualization.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Save evaluation results
    results_summary = {
        'category': _category_name,
        'image_auroc': img_auroc,
        'pixel_auroc': px_auroc,
        'total_samples': len(labels_all),
        'config': {
            'image_size': _image_size,
            'batch_size': _batch_size,
            'num_timesteps': _num_timesteps,
            'beta_schedule': _beta_schedule
        }
    }

    # Save results to file
    with open(os.path.join(output_base_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults file saved to: {output_base_dir}")
    print(f"Visualizations saved to: {visualization_dir}")
    return results_summary

if __name__ == "__main__":
    results = run_inference_with_visualization()
    print(f"\n[FINAL RESULTS] {results['category']}:")
    print(f"  Image AUROC: {results['image_auroc']:.4f}")
    print(f"  Pixel AUROC: {results['pixel_auroc']:.4f}")
    print(f"  Total samples: {results['total_samples']}")



""" Inference version in train """
@torch.no_grad()
def evaluate_model(model, vae_model, gaussian_diffusion, device):
    print("\n[INFO] Running evaluation ...")
    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=_mvtec_data_dir,
        category=_category_name,
        image_size=_image_size,
        batch_size=_batch_size,
        shuffle=False,
    )
    model.eval()
    vae_model.eval()

    labels_all: List[int] = []
    img_scores_all: List[float] = []
    maps_all: List[torch.Tensor] = []
    gts_all: List[torch.Tensor] = []

    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)  # [B,1,H,W]
        labels = batch['label']          # list/int/tensor

        # vae reconstruction
        vae_reconstructed, _, _ = vae_model(images)

        B = images.size(0)
        t = torch.randint(300, _num_timesteps, (B,), device=device)
        noise = torch.randn_like(vae_reconstructed).to(device)
        # x_t: add noise
        x_t = gaussian_diffusion.q_sample(vae_reconstructed, t, noise)

        images_reconstructed = model(x_t, t)

        # Compute anomaly map and scores
        anomaly_map = compute_anomaly_map(images, images_reconstructed)  # [B,H,W] or [B,1,H,W]

        # Handle different anomaly map shapes
        if anomaly_map.dim() == 4 and anomaly_map.shape[1] == 1:
            anomaly_map = anomaly_map.squeeze(1)  # [B,H,W]

        image_scores = calc_image_score(anomaly_map)

        # Accumulate
        labels_all.extend(to_label_list(labels))
        img_scores_all.extend(image_scores.detach().cpu().tolist())
        maps_all.extend(list(anomaly_map.detach().cpu()))
        gts_all.extend(list(masks.detach().cpu().squeeze(1)))

    # Calculate metrics
    img_auroc = eval_auroc_image(labels_all, img_scores_all)
    px_auroc = eval_auroc_pixel(maps_all, gts_all)

    # print(f"[EVAL] Image AUROC: {img_auroc:.4f}, Pixel AUROC: {px_auroc:.4f}")
    return img_auroc, px_auroc