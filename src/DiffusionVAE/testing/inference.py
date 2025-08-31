import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from config import load_config
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC
from data.dataloader import load_mvtec_test_dataset
from vae.vae_resnet import VAEResNet
from diffusion.model.ddpm_model import DDPM
from testing.visualization import create_visualization_grid, save_visualization_results, tensor_to_numpy, create_heatmap, overlay_heatmap_on_image
import json
import matplotlib.pyplot as plt

config = load_config()

_seed = config.general.seed
_image_score_type_name = config.testing.image_score_type_name
_testing_category = config.testing.category
# Model path
_vae_model_path = config.testing.vae_model_path
_diffusion_model_path = config.testing.diffusion_model_path
# Save result dir
_testing_result_dir = config.testing.test_result_base_dir + _testing_category + '/'
# Visualization
_save_visualizations = config.testing.save_visualizations
_max_visualization_batches = config.testing.max_visualization_batches
_max_images_per_batch = config.testing.max_images_per_batch

# Device
_device = torch.device(f'cuda:{config.general.cuda}' if torch.cuda.is_available() else 'cpu')

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

def calc_image_score(anomaly_map):
    if _image_score_type_name == 'max':
        return anomaly_map.amax(dim=(1, 2))
    elif _image_score_type_name == 'mean':
        return anomaly_map.float().mean(dim=(1, 2))
    elif _image_score_type_name == 'std':
        return anomaly_map.float().std(dim=(1, 2), unbiased=False)
    else:
        raise ValueError(f"Unsupported score type: {_image_score_type_name}")

def load_vae_model():
    """Load VAE model from checkpoint"""
    print(f"Loading VAE model from: {_vae_model_path}")
    
    # Initialize VAE model
    vae_model = VAEResNet(
        image_size=config.general.image_size,
        in_channels=config.vae_model.in_channels,
        latent_dim=config.vae_model.z_dim,
        out_channels=config.vae_model.out_channels,
        resnet_name=config.vae_model.backbone,
        dropout_p=config.vae_model.dropout_p
    )
    
    # Load checkpoint
    checkpoint = torch.load(_vae_model_path, map_location=_device)
    vae_model.load_state_dict(checkpoint['model_state_dict'])
    vae_model.to(_device)
    vae_model.eval()
    
    print("VAE model loaded successfully")
    return vae_model

def load_diffusion_model():
    """Load Diffusion model from checkpoint"""
    print(f"Loading Diffusion model from: {_diffusion_model_path}")
    
    # Initialize diffusion model
    diffusion_model = DDPM(
        in_channel=config.diffusion_model.unet.in_channel,
        out_channel=config.diffusion_model.unet.out_channel,
        inner_channel=config.diffusion_model.unet.inner_channel,
        norm_groups=config.diffusion_model.unet.norm_groups,
        channel_mults=config.diffusion_model.unet.channel_mults,
        attn_res=config.diffusion_model.unet.attn_res,
        res_blocks=config.diffusion_model.unet.res_blocks,
        dropout_p=config.diffusion_model.unet.dropout,
        image_size=config.diffusion_model.diffusion.image_size,
        channels=config.diffusion_model.diffusion.channels,
        loss_type=config.diffusion_model.loss_type
    )
    
    # Load checkpoint
    checkpoint = torch.load(_diffusion_model_path, map_location=_device)
    diffusion_model.netG.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.to(_device)
    diffusion_model.eval()
    
    # Set validation noise schedule
    val_schedule = config.diffusion_model.beta_schedule.val
    diffusion_model.set_new_noise_schedule(
        schedule=val_schedule.schedule,
        n_timestep=val_schedule.n_timestep,
        linear_start=val_schedule.linear_start,
        linear_end=val_schedule.linear_end,
        schedule_phase='val'
    )
    
    print("Diffusion model loaded successfully")
    return diffusion_model

def run_inference():
    """Run inference pipeline: Original → VAE → VAE Reconstruction → Diffusion Restoration → Calculate metrics"""
    
    # Create result directory
    os.makedirs(_testing_result_dir, exist_ok=True)
    
    # Load models
    vae_model = load_vae_model()
    diffusion_model = load_diffusion_model()
    
    # Load test dataset
    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=config.data.mvtec_data_dir,
        category=_testing_category,
        image_size=config.general.image_size,
        batch_size=config.general.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"Testing on {_testing_category} category with {len(test_loader)} batches")
    
    # Initialize metrics storage
    all_labels = []
    all_image_scores = []
    all_anomaly_maps = []
    all_gt_masks = []
    
    # Visualization storage
    visualization_results = {
        'input_images': [],
        'vae_reconstructions': [],
        'diffusion_reconstructions': [],
        'anomaly_maps': [],
        'gt_masks': []
    }
    
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks, labels) in enumerate(tqdm(test_loader, desc="Running inference")):
            # Move to device
            images = images.to(_device)  # [B, C, H, W]
            masks = masks.to(_device)    # [B, 1, H, W]
            labels = labels.to(_device)  # [B]
            
            # Step 1: VAE Reconstruction
            vae_reconstructions = vae_model.reconstruct(images)  # [B, C, H, W]
            
            # Step 2: Diffusion Restoration (using VAE reconstructions as input)
            # For conditional diffusion, we need to use the super_resolution method
            if config.diffusion_model.diffusion.conditional:
                # Use super_resolution method which expects the low-quality input (VAE reconstruction)
                diffusion_reconstructions = diffusion_model.netG.super_resolution(
                    vae_reconstructions, 
                    continous=False
                )  # [B, C, H, W]
            else:
                # For non-conditional, use regular sample method
                diffusion_reconstructions = diffusion_model.netG.sample(
                    batch_size=images.size(0), 
                    continous=False
                )  # [B, C, H, W]
            
            # Step 3: Calculate anomaly maps (difference between original and diffusion output)
            anomaly_maps = compute_anomaly_map(images, diffusion_reconstructions)  # [B, H, W]
            
            # Step 4: Calculate image-level scores
            image_scores = calc_image_score(anomaly_maps)  # [B]
            
            # Store results
            all_labels.extend(labels.cpu().numpy().tolist())
            all_image_scores.extend(image_scores.cpu().numpy().tolist())
            all_anomaly_maps.extend([am.cpu() for am in anomaly_maps])
            all_gt_masks.extend([mask.squeeze(0).cpu() for mask in masks])  # Remove channel dimension
            
            # Store visualization data (limit to max batches)
            if batch_count < _max_visualization_batches:
                max_images = min(images.size(0), _max_images_per_batch)
                
                visualization_results['input_images'].append(images[:max_images])
                visualization_results['vae_reconstructions'].append(vae_reconstructions[:max_images])
                visualization_results['diffusion_reconstructions'].append(diffusion_reconstructions[:max_images])
                visualization_results['anomaly_maps'].append(anomaly_maps[:max_images])
                visualization_results['gt_masks'].append(masks[:max_images])
                
                batch_count += 1
    
    # Calculate metrics
    image_auroc = eval_auroc_image(all_labels, all_image_scores)
    pixel_auroc = eval_auroc_pixel(all_anomaly_maps, all_gt_masks)
    
    # Save evaluation results
    evaluation_results = {
        'category': _testing_category,
        'image_auroc': image_auroc,
        'pixel_auroc': pixel_auroc,
        'image_score_type': _image_score_type_name,
        'total_samples': len(all_labels),
        'anomaly_samples': sum(all_labels),
        'normal_samples': len(all_labels) - sum(all_labels)
    }
    
    results_path = os.path.join(_testing_result_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Evaluation Results:")
    print(f"  Category: {_testing_category}")
    print(f"  Image AUROC: {image_auroc:.4f}")
    print(f"  Pixel AUROC: {pixel_auroc:.4f}")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Anomaly samples: {sum(all_labels)}")
    print(f"  Normal samples: {len(all_labels) - sum(all_labels)}")
    print(f"Results saved to: {results_path}")
    
    # Save visualizations if enabled
    if _save_visualizations:
        print("Saving visualizations...")
        vis_dir = os.path.join(_testing_result_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        for batch_idx in range(len(visualization_results['input_images'])):
            # Create visualization grid
            fig = create_visualization_grid(
                input_images=visualization_results['input_images'][batch_idx],
                vae_reconstructions=visualization_results['vae_reconstructions'][batch_idx],
                diffusion_reconstructions=visualization_results['diffusion_reconstructions'][batch_idx],
                anomaly_maps=visualization_results['anomaly_maps'][batch_idx],
                gt_masks=visualization_results['gt_masks'][batch_idx],
                max_images=_max_images_per_batch
            )
            
            # Save grid
            grid_path = os.path.join(vis_dir, f'batch_{batch_idx:03d}_grid.png')
            fig.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Save individual components
            batch_vis_dir = os.path.join(vis_dir, f'batch_{batch_idx:03d}')
            save_visualization_results(
                results={
                    'input_images': [tensor_to_numpy(img) for img in visualization_results['input_images'][batch_idx]],
                    'vae_reconstructions': [tensor_to_numpy(img) for img in visualization_results['vae_reconstructions'][batch_idx]],
                    'diffusion_reconstructions': [tensor_to_numpy(img) for img in visualization_results['diffusion_reconstructions'][batch_idx]],
                    'heatmaps': [create_heatmap(am, am.shape[0], am.shape[1]) for am in visualization_results['anomaly_maps'][batch_idx]],
                    'overlays': [overlay_heatmap_on_image(
                        tensor_to_numpy(visualization_results['input_images'][batch_idx][i]),
                        create_heatmap(visualization_results['anomaly_maps'][batch_idx][i], 
                                     visualization_results['input_images'][batch_idx][i].shape[1], 
                                     visualization_results['input_images'][batch_idx][i].shape[2])
                    ) for i in range(len(visualization_results['input_images'][batch_idx]))],
                    'gt_masks': [tensor_to_numpy(mask) for mask in visualization_results['gt_masks'][batch_idx]]
                },
                save_dir=batch_vis_dir,
                category_name=_testing_category,
                batch_idx=batch_idx
            )
        
        print(f"Visualizations saved to: {vis_dir}")

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    set_seed(_seed)
    run_inference()
