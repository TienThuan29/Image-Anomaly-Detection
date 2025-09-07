import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from config import load_config
from data.dataloader import load_mvtec_test_dataset
from diffusion.model.ddpm_model import DDPM
from testing.visualization import (
    create_visualization_grid,
    save_visualization_results,
    tensor_to_numpy,
    create_heatmap,
    overlay_heatmap_on_image
)
import json
import matplotlib.pyplot as plt
from testing.metric import (
    calc_image_score,
    eval_auroc_image,
    eval_auroc_pixel,
    compute_anomaly_map
)
from vae.utils import load_vae_model
import torchvision.models as models

config = load_config()

_seed = config.general.seed
_image_score_type_name = config.testing.image_score_type_name
_testing_category = config.testing.category
_image_size = config.general.image_size
# Model path
_vae_model_path = config.testing.vae_model_path
_diffusion_model_path = config.testing.diffusion_model_path
# Save result dir
_testing_result_dir = config.testing.test_result_base_dir + _testing_category + '/'
# Visualization
_save_visualizations = config.testing.save_visualizations
_max_visualization_batches = config.testing.max_visualization_batches
_max_images_per_batch = config.testing.max_images_per_batch

# vae
_vae_name = config.vae_model.name
_backbone = config.vae_model.backbone
_input_channels = config.vae_model.in_channels
_output_channels = config.vae_model.out_channels
_z_dim = config.vae_model.z_dim
_dropout_p = config.vae_model.dropout_p

# Device
_device = torch.device(f'cuda:{config.general.cuda}' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor:
    """Feature extractor using pre-trained VGG16 for perceptual loss"""
    def __init__(self, device):
        self.device = device
        # Load pre-trained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Extract features from conv layers before maxpool
        self.features = torch.nn.Sequential(*list(vgg.features.children())[:16])  # Up to conv3_3
        self.features.eval()
        self.features.to(device)
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def extract_features(self, x):
        """Extract features from input tensor"""
        with torch.no_grad():
            features = self.features(x)
        return features

def compute_anomaly_map_improved(x, x_recon, feature_extractor=None, 
                                l1_weight=0.7, perceptual_weight=0.3, 
                                use_multi_scale=True, use_perceptual=True):
    """
    Improved anomaly map computation combining multiple approaches
    
    Args:
        x: Original images [B, C, H, W]
        x_recon: Reconstructed images [B, C, H, W]
        feature_extractor: FeatureExtractor instance for perceptual loss
        l1_weight: Weight for L1 pixel difference
        perceptual_weight: Weight for perceptual difference
        use_multi_scale: Whether to use multi-scale analysis
        use_perceptual: Whether to use perceptual features
    
    Returns:
        anomaly_map: Combined anomaly map [B, H, W]
    """
    batch_size = x.shape[0]
    
    # 1. L1 pixel difference (baseline)
    l1_map = (x - x_recon).abs().mean(dim=1)  # [B, H, W]
    
    if not use_perceptual or feature_extractor is None:
        return l1_map
    
    # 2. Perceptual difference using VGG features
    with torch.no_grad():
        feat_orig = feature_extractor.extract_features(x)  # [B, 256, H/8, W/8]
        feat_recon = feature_extractor.extract_features(x_recon)
        
        # Compute perceptual difference
        perceptual_diff = (feat_orig - feat_recon).abs().mean(dim=1)  # [B, H/8, W/8]
        
        # Upsample to original resolution
        perceptual_map = F.interpolate(
            perceptual_diff.unsqueeze(1), 
            size=(x.shape[2], x.shape[3]), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)  # [B, H, W]
    
    # 3. Multi-scale analysis (optional)
    if use_multi_scale:
        # Create multi-scale anomaly maps
        scales = [0.5, 0.75, 1.0]
        multi_scale_maps = []
        
        for scale in scales:
            if scale != 1.0:
                # Resize images
                h_scaled = int(x.shape[2] * scale)
                w_scaled = int(x.shape[3] * scale)
                x_scaled = F.interpolate(x, size=(h_scaled, w_scaled), mode='bilinear', align_corners=False)
                x_recon_scaled = F.interpolate(x_recon, size=(h_scaled, w_scaled), mode='bilinear', align_corners=False)
                
                # Compute difference at this scale
                scale_diff = (x_scaled - x_recon_scaled).abs().mean(dim=1)
                
                # Resize back to original
                scale_diff = F.interpolate(
                    scale_diff.unsqueeze(1), 
                    size=(x.shape[2], x.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            else:
                scale_diff = l1_map
            
            multi_scale_maps.append(scale_diff)
        
        # Combine multi-scale maps
        multi_scale_map = torch.stack(multi_scale_maps, dim=0).mean(dim=0)  # [B, H, W]
        
        # Combine all components
        anomaly_map = (l1_weight * l1_map + 
                      perceptual_weight * perceptual_map + 
                      0.2 * multi_scale_map)
    else:
        # Simple combination without multi-scale
        anomaly_map = l1_weight * l1_map + perceptual_weight * perceptual_map
    
    return anomaly_map

def compute_anomaly_map_advanced(x, x_recon, feature_extractor=None):
    results = {}
    # 1. L1 pixel difference
    results['l1'] = (x - x_recon).abs().mean(dim=1)
    # 2. L2 pixel difference
    results['l2'] = ((x - x_recon) ** 2).mean(dim=1).sqrt()
    # 3. SSIM-based difference (simplified)
    mu1 = x.mean(dim=1, keepdim=True)
    mu2 = x_recon.mean(dim=1, keepdim=True)
    sigma1 = x.var(dim=1, keepdim=True)
    sigma2 = x_recon.var(dim=1, keepdim=True)
    sigma12 = ((x - mu1) * (x_recon - mu2)).mean(dim=1, keepdim=True)
    
    c1, c2 = 0.01**2, 0.03**2
    ssim = (2*mu1*mu2 + c1) * (2*sigma12 + c2) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    results['ssim'] = 1 - ssim.squeeze(1)
    
    # 4. Perceptual difference
    if feature_extractor is not None:
        with torch.no_grad():
            feat_orig = feature_extractor.extract_features(x)
            feat_recon = feature_extractor.extract_features(x_recon)
            perceptual_diff = (feat_orig - feat_recon).abs().mean(dim=1)
            
            # Upsample to original resolution
            results['perceptual'] = F.interpolate(
                perceptual_diff.unsqueeze(1), 
                size=(x.shape[2], x.shape[3]), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
    
    # 5. Combined map
    combined = (0.4 * results['l1'] + 
               0.3 * results['l2'] + 
               0.2 * results['ssim'])
    
    if 'perceptual' in results:
        combined += 0.1 * results['perceptual']
    results['combined'] = combined
    return results

def load_diffusion_model():
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

    checkpoint = torch.load(_diffusion_model_path, map_location=_device)
    diffusion_model.netG.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.netG.to(_device)
    diffusion_model.netG.eval()

    val_schedule = config.diffusion_model.beta_schedule.val
    diffusion_model.set_new_noise_schedule(
        schedule=val_schedule['schedule'],
        n_timestep=val_schedule['n_timestep'],
        linear_start=val_schedule['linear_start'],
        linear_end=val_schedule['linear_end'],
        schedule_phase='val'
    )
    
    print("Diffusion model loaded success")
    return diffusion_model

def run_inference_improved(use_advanced=False, save_individual_maps=True):
    """
    Args:
        use_advanced: Whether to use advanced multi-strategy anomaly detection
        save_individual_maps: Whether to save individual anomaly map types
    """
    set_seed(_seed)
    os.makedirs(_testing_result_dir, exist_ok=True)

    # Load models
    vae_model = load_vae_model(
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
    diffusion_model = load_diffusion_model()
    diffusion_model.set_noise_schedule_for_val()

    # Initialize feature extractor for perceptual loss
    feature_extractor = FeatureExtractor(_device)
    print("Feature extractor (VGG16) loaded successfully")

    vae_model = vae_model.to(_device)
    diffusion_model.netG = diffusion_model.netG.to(_device)
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

    all_labels = []
    all_image_scores = []
    all_image_scores_max = []
    all_image_scores_mean = []
    all_image_scores_std = []
    all_anomaly_maps = []
    all_gt_masks = []

    # Store individual anomaly map types if using advanced method
    if use_advanced:
        individual_maps = {
            'l1': [], 'l2': [], 'ssim': [], 'perceptual': [], 'combined': []
        }

    visualization_results = {
        'input_images': [],
        'vae_reconstructions': [],
        'diffusion_reconstructions': [],
        'anomaly_maps': [],
        'gt_masks': []
    }
    
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running improved inference")):
            
            # Extract data from batch dictionary
            images = batch['image'].to(_device)  # [B, C, H, W]
            masks = batch['mask'].to(_device)    # [B, 1, H, W]
            labels = batch['label'].to(_device)  # [B]

            vae_reconstructions = vae_model.reconstruct(images)  # [B, C, H, W]

            if config.diffusion_model.diffusion.conditional:
                diffusion_reconstructions = diffusion_model.netG.super_resolution(
                    vae_reconstructions,
                    continous=False
                )  # [B, C, H, W]
            else:
                diffusion_reconstructions = diffusion_model.netG.sample(
                    batch_size=images.size(0),
                    continous=False
                )  # [B, C, H, W]

            # Compute anomaly maps using improved method
            if use_advanced:
                anomaly_maps_dict = compute_anomaly_map_advanced(images, diffusion_reconstructions, feature_extractor)
                anomaly_maps = anomaly_maps_dict['combined']  # Use combined map for evaluation
                
                # Store individual maps if requested
                if save_individual_maps:
                    for map_type, map_tensor in anomaly_maps_dict.items():
                        individual_maps[map_type].extend([am.cpu() for am in map_tensor])
            else:
                # Use improved single method
                anomaly_maps = compute_anomaly_map_improved(
                    images, diffusion_reconstructions, feature_extractor,
                    l1_weight=0.6, perceptual_weight=0.4, 
                    use_multi_scale=True, use_perceptual=True
                )
            
            image_scores = calc_image_score(anomaly_maps, _image_score_type_name)  # [B] or tuple([B],[B],[B])
            
            # Store results
            all_labels.extend(labels.cpu().numpy().tolist())
            if isinstance(image_scores, tuple):
                max_scores, mean_scores, std_scores = image_scores
                all_image_scores_max.extend(max_scores.cpu().numpy().tolist())
                all_image_scores_mean.extend(mean_scores.cpu().numpy().tolist())
                all_image_scores_std.extend(std_scores.cpu().numpy().tolist())
            else:
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
    if _image_score_type_name == 'all':
        image_auroc = {
            'max': eval_auroc_image(all_labels, all_image_scores_max),
            'mean': eval_auroc_image(all_labels, all_image_scores_mean),
            'std': eval_auroc_image(all_labels, all_image_scores_std)
        }
    else:
        image_auroc = eval_auroc_image(all_labels, all_image_scores)

    pixel_auroc = eval_auroc_pixel(all_anomaly_maps, all_gt_masks)
    
    # Calculate metrics for individual map types if using advanced method
    individual_metrics = {}
    if use_advanced and save_individual_maps:
        print("Calculating metrics for individual anomaly map types...")
        for map_type, maps in individual_maps.items():
            if len(maps) > 0:
                individual_metrics[map_type] = {
                    'image_auroc': eval_auroc_image(all_labels, 
                        [calc_image_score(torch.stack([maps[i]]), _image_score_type_name).item() 
                         for i in range(len(maps))]),
                    'pixel_auroc': eval_auroc_pixel(maps, all_gt_masks)
                }
    
    # Save evaluation results
    evaluation_results = {
        'category': _testing_category,
        'method': 'improved_advanced' if use_advanced else 'improved_single',
        'image_auroc': image_auroc,
        'pixel_auroc': pixel_auroc,
        'image_score_type': _image_score_type_name,
        'total_samples': len(all_labels),
        'anomaly_samples': sum(all_labels),
        'normal_samples': len(all_labels) - sum(all_labels),
        'feature_extractor': 'VGG16',
        'anomaly_map_weights': {
            'l1_weight': 0.6 if not use_advanced else 0.4,
            'perceptual_weight': 0.4 if not use_advanced else 0.1,
            'multi_scale': True
        }
    }
    
    # Add individual metrics if available
    if individual_metrics:
        evaluation_results['individual_metrics'] = individual_metrics
    
    results_path = os.path.join(_testing_result_dir, 'evaluation_results_improved.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Improved Evaluation Results:")
    print(f"  Category: {_testing_category}")
    print(f"  Method: {'Advanced Multi-Strategy' if use_advanced else 'Improved Single'}")
    if isinstance(image_auroc, dict):
        print(f"  Image AUROC (max): {image_auroc['max']:.4f}")
        print(f"  Image AUROC (mean): {image_auroc['mean']:.4f}")
        print(f"  Image AUROC (std): {image_auroc['std']:.4f}")
    else:
        print(f"  Image AUROC: {image_auroc:.4f}")
    print(f"  Pixel AUROC: {pixel_auroc:.4f}")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Anomaly samples: {sum(all_labels)}")
    print(f"  Normal samples: {len(all_labels) - sum(all_labels)}")
    
    # Print individual metrics if available
    if individual_metrics:
        print(f"\nIndividual Anomaly Map Metrics:")
        for map_type, metrics in individual_metrics.items():
            print(f"  {map_type.upper()}:")
            print(f"    Image AUROC: {metrics['image_auroc']:.4f}")
            print(f"    Pixel AUROC: {metrics['pixel_auroc']:.4f}")
    
    print(f"Results saved to: {results_path}")
    
    # Save visualizations if enabled
    if _save_visualizations:
        print("Saving visualizations...")
        vis_dir = os.path.join(_testing_result_dir, 'visualizations_improved')
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
            grid_path = os.path.join(vis_dir, f'batch_{batch_idx:03d}_grid_improved.png')
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

@torch.no_grad()
def run_inference_during_training_improved(vae_model, diffusion_model, result_dir=None, use_advanced=False):
    """Run improved inference during training with optional custom result directory."""
    
    # Define device
    device = torch.device(f'cuda:{config.general.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # Ensure models are on the correct device and in eval mode
    vae_model = vae_model.to(device)
    diffusion_model.netG = diffusion_model.netG.to(device)
    vae_model.eval()
    diffusion_model.netG.eval()
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(device)
    
    # Use provided result_dir or default to testing result dir
    save_dir = result_dir if result_dir is not None else _testing_result_dir
    os.makedirs(save_dir, exist_ok=True)

    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=config.data.mvtec_data_dir,
        category=_testing_category,
        image_size=config.general.image_size,
        batch_size=config.general.batch_size
    )
    print(f"Testing on {_testing_category} category with {len(test_loader)} batches")

    all_labels = []
    all_image_scores = []
    all_image_scores_max = []
    all_image_scores_mean = []
    all_image_scores_std = []
    all_anomaly_maps = []
    all_gt_masks = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running improved inference during training")):
            
            # Extract data from batch dictionary
            images = batch['image'].to(device)  # [B, C, H, W]
            masks = batch['mask'].to(device)  # [B, 1, H, W]
            labels = batch['label'].to(device)  # [B]

            vae_reconstructions, _, _ = vae_model(images)  # [B, C, H, W]

            # Diffusion Restoration using VAE reconstructions as input
            if config.diffusion_model.diffusion.conditional:
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

            # Calculate anomaly maps using improved method
            if use_advanced:
                anomaly_maps_dict = compute_anomaly_map_advanced(images, diffusion_reconstructions, feature_extractor)
                anomaly_maps = anomaly_maps_dict['combined']
            else:
                anomaly_maps = compute_anomaly_map_improved(
                    images, diffusion_reconstructions, feature_extractor,
                    l1_weight=0.6, perceptual_weight=0.4, 
                    use_multi_scale=True, use_perceptual=True
                )

            # Calculate image-level scores
            image_scores = calc_image_score(anomaly_maps, _image_score_type_name)  # [B] or tuple([B],[B],[B])

            # Store results
            all_labels.extend(labels.cpu().numpy().tolist())
            if isinstance(image_scores, tuple):
                max_scores, mean_scores, std_scores = image_scores
                all_image_scores_max.extend(max_scores.cpu().numpy().tolist())
                all_image_scores_mean.extend(mean_scores.cpu().numpy().tolist())
                all_image_scores_std.extend(std_scores.cpu().numpy().tolist())
            else:
                all_image_scores.extend(image_scores.cpu().numpy().tolist())
            all_anomaly_maps.extend([am.cpu() for am in anomaly_maps])
            all_gt_masks.extend([mask.squeeze(0).cpu() for mask in masks])  # Remove channel dimension

    # Calculate metrics
    if _image_score_type_name == 'all':
        image_auroc = {
            'max': eval_auroc_image(all_labels, all_image_scores_max),
            'mean': eval_auroc_image(all_labels, all_image_scores_mean),
            'std': eval_auroc_image(all_labels, all_image_scores_std)
        }
    else:
        image_auroc = eval_auroc_image(all_labels, all_image_scores)
    pixel_auroc = eval_auroc_pixel(all_anomaly_maps, all_gt_masks)

    # Save evaluation results
    evaluation_results = {
        'category': _testing_category,
        'method': 'improved_advanced' if use_advanced else 'improved_single',
        'image_auroc': image_auroc,
        'pixel_auroc': pixel_auroc,
        'image_score_type': _image_score_type_name,
        'total_samples': len(all_labels),
        'anomaly_samples': sum(all_labels),
        'normal_samples': len(all_labels) - sum(all_labels),
        'feature_extractor': 'VGG16'
    }

    results_path = os.path.join(save_dir, 'evaluation_results_improved.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"Improved Evaluation Results (During Training):")
    print(f"  Category: {_testing_category}")
    print(f"  Method: {'Advanced Multi-Strategy' if use_advanced else 'Improved Single'}")
    if isinstance(image_auroc, dict):
        print(f"  Image AUROC (max): {image_auroc['max']:.4f}")
        print(f"  Image AUROC (mean): {image_auroc['mean']:.4f}")
        print(f"  Image AUROC (std): {image_auroc['std']:.4f}")
    else:
        print(f"  Image AUROC: {image_auroc:.4f}")
    print(f"  Pixel AUROC: {pixel_auroc:.4f}")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Anomaly samples: {sum(all_labels)}")
    print(f"  Normal samples: {len(all_labels) - sum(all_labels)}")
    print(f"Results saved to: {results_path}")

    return image_auroc, pixel_auroc

if __name__ == "__main__":
    # Run improved inference
    print("Running improved inference with single strategy...")
    run_inference_improved(use_advanced=False, save_individual_maps=False)
    
    print("\nRunning improved inference with advanced multi-strategy...")
    run_inference_improved(use_advanced=True, save_individual_maps=True)
