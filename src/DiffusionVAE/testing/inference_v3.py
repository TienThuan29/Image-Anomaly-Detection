import os
import torch
from tqdm import tqdm
from config import load_config
from data.dataloader import load_mvtec_test_dataset
from testing.visualization import save_visualization
import json
from testing.metric import (
    calc_image_score,
    eval_auroc_image,
    eval_auroc_pixel,
    compute_anomaly_map,
    calc_ssim,
    compute_perceptual_anomaly_map,
    compute_multiscale_perceptual_anomaly,
    compute_combined_anomaly_map
)
import numpy as np
from vae.utils import load_vae_model
from diffusion.utils import load_diffusion_model

config = load_config()

_seed = config.general.seed
_testing_category = config.testing.category
_image_size = config.general.image_size
_mvtec_data_dir = config.data.mvtec_data_dir
# Model path
_vae_model_path = config.testing.vae_model_path
_diffusion_model_path = config.testing.diffusion_model_path

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

# test batch size
_test_batch_size = config.testing.batch_size
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    vae_model = load_vae_model(
        checkpoint_path=_vae_model_path,
        vae_name=_vae_name,
        input_channels=_input_channels,
        output_channels=_output_channels,
        z_dim=_z_dim,
        backbone=_backbone,
        dropout_p=_dropout_p,
        image_size=_image_size,
        device=device
    )
    diffusion_model = load_diffusion_model(
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
        loss_type=config.diffusion_model.loss_type,
        diffusion_model_path=_diffusion_model_path,
        device=device
    )
    diffusion_model.set_noise_schedule_for_val()
    return vae_model, diffusion_model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_anomaly_maps_v3(vae_model, diffusion_model, is_visualization=True, method='combined', 
                       pixel_weight=0.5, perceptual_weight=0.5, use_multiscale=True):
    """
    Get anomaly maps using V3 method (combined pixel + perceptual).
    
    Args:
        vae_model: Pre-trained VAE model
        diffusion_model: Pre-trained diffusion model
        is_visualization: Whether to store visualization data
        method: Method to use ('combined', 'perceptual', 'multiscale_perceptual', 'pixel_only')
        pixel_weight: Weight for pixel-level anomaly (for combined method)
        perceptual_weight: Weight for perceptual anomaly (for combined method)
        use_multiscale: Whether to use multi-scale perceptual features
    """
    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=_mvtec_data_dir,
        category=_testing_category,
        image_size=_image_size,
        batch_size=_test_batch_size
    )
    print(f"Test on {_testing_category} category using V3 {method.upper()} method")
    if method == 'combined':
        print(f"  Pixel weight: {pixel_weight}, Perceptual weight: {perceptual_weight}")
        print(f"  Use multiscale: {use_multiscale}")

    batch_count = 0
    all_labels = []
    all_anomaly_maps = []
    all_gt_masks = []
    visualization_results = {
        'input_images': [],
        'vae_reconstructions': [],
        'diffusion_reconstructions': [],
        'anomaly_maps': [],
        'pixel_anomaly_maps': [],
        'perceptual_anomaly_maps': [],
        'gt_masks': []
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="run inference v3")):
            images = batch['image'].to(device)  # [B, C, H, W]
            masks = batch['mask'].to(device)  # [B, 1, H, W]
            labels = batch['label'].to(device)  # [B]

            vae_reconstructions, _, _ = vae_model(images)  # [B, C, H, W]
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

            # Compute different types of anomaly maps
            pixel_anomaly_maps = None
            perceptual_anomaly_maps = None
            
            if method == 'pixel_only':
                anomaly_maps = compute_anomaly_map(
                    x=images,
                    x_recon=diffusion_reconstructions
                )
            elif method == 'perceptual':
                anomaly_maps = compute_perceptual_anomaly_map(
                    vae_model=vae_model,
                    x=images,
                    x_recon=diffusion_reconstructions
                )
            elif method == 'multiscale_perceptual':
                anomaly_maps = compute_multiscale_perceptual_anomaly(
                    vae_model=vae_model,
                    x=images,
                    x_recon=diffusion_reconstructions
                )
            elif method == 'combined':
                # Compute both pixel and perceptual for visualization
                pixel_anomaly_maps = compute_anomaly_map(
                    x=images,
                    x_recon=diffusion_reconstructions
                )
                if use_multiscale:
                    perceptual_anomaly_maps = compute_multiscale_perceptual_anomaly(
                        vae_model=vae_model,
                        x=images,
                        x_recon=diffusion_reconstructions
                    )
                else:
                    perceptual_anomaly_maps = compute_perceptual_anomaly_map(
                        vae_model=vae_model,
                        x=images,
                        x_recon=diffusion_reconstructions
                    )
                
                # Combine them
                anomaly_maps = compute_combined_anomaly_map(
                    vae_model=vae_model,
                    x=images,
                    x_recon=diffusion_reconstructions,
                    pixel_weight=pixel_weight,
                    perceptual_weight=perceptual_weight,
                    use_multiscale=use_multiscale
                )
            else:
                raise ValueError(f"Unsupported method: {method}. Use 'combined', 'perceptual', 'multiscale_perceptual', or 'pixel_only'.")

            all_labels.extend(labels.cpu().numpy().tolist())
            all_anomaly_maps.extend([am.cpu() for am in anomaly_maps])
            all_gt_masks.extend([mask.squeeze(0).cpu() for mask in masks])  # Remove channel dim

            # Store visualization data, limit max batches
            if is_visualization:
                if batch_count < _max_visualization_batches:
                    max_images = min(images.size(0), _max_images_per_batch)
                    visualization_results['input_images'].append(images[:max_images])
                    visualization_results['vae_reconstructions'].append(vae_reconstructions[:max_images])
                    visualization_results['diffusion_reconstructions'].append(diffusion_reconstructions[:max_images])
                    visualization_results['anomaly_maps'].append(anomaly_maps[:max_images])
                    visualization_results['gt_masks'].append(masks[:max_images])
                    
                    # Store individual components for combined method
                    if method == 'combined':
                        visualization_results['pixel_anomaly_maps'].append(pixel_anomaly_maps[:max_images])
                        visualization_results['perceptual_anomaly_maps'].append(perceptual_anomaly_maps[:max_images])
                    else:
                        # For other methods, use the same anomaly map
                        visualization_results['pixel_anomaly_maps'].append(anomaly_maps[:max_images])
                        visualization_results['perceptual_anomaly_maps'].append(anomaly_maps[:max_images])
                    
                    batch_count += 1

    return all_labels, all_anomaly_maps, all_gt_masks, visualization_results


def run_inference_v3(all_labels, all_anomaly_maps, all_gt_masks, method='combined'):
    results = []
    score_types = ['max', 'mean', 'std']

    pixel_auroc = eval_auroc_pixel(all_anomaly_maps, all_gt_masks)
    for score_type in score_types:
        print(f'-- Type: {score_type} --')
        image_scores = calc_image_score(all_anomaly_maps, score_type)
        image_auroc = eval_auroc_image(all_labels, image_scores.cpu().numpy().tolist())

        evaluation_result = {
            'category': _testing_category,
            'method': f'v3_{method}',
            'image_auroc': image_auroc,
            'pixel_auroc': pixel_auroc,
            'image_score_type': score_type,
            'total_samples': len(all_labels),
            'anomaly_samples': sum(all_labels),
            'normal_samples': len(all_labels) - sum(all_labels),
        }

        print(f"  Image AUROC: {image_auroc:.4f}")
        print(f"  Pixel AUROC: {pixel_auroc:.4f}")
        print(f"  Total samples: {len(all_labels)}")
        print(f"  Anomaly samples: {sum(all_labels)}")
        print(f"  Normal samples: {len(all_labels) - sum(all_labels)}")
        results.append(evaluation_result)

    return results


def run_all_inference_v3():
    """Run all V3 inference methods."""
    set_seed(_seed)
    all_results = []
    vae_model, diffusion_model = load_models()

    # Test different methods
    methods = [
        ('pixel_only', 1.0, 0.0, False),
        ('perceptual', 0.0, 1.0, False),
        ('multiscale_perceptual', 0.0, 1.0, True),
        ('combined', 0.5, 0.5, False),
        ('combined', 0.3, 0.7, False),
        ('combined', 0.7, 0.3, False),
        ('combined', 0.5, 0.5, True),
    ]

    for method, pixel_weight, perceptual_weight, use_multiscale in methods:
        print(f'\n========== V3 {method.upper()} ===========')
        if method == 'combined':
            print(f'Pixel weight: {pixel_weight}, Perceptual weight: {perceptual_weight}, Multiscale: {use_multiscale}')
        
        testing_result_dir = config.testing.test_result_base_dir + _testing_category + f'/v3_{method}/'
        if method == 'combined':
            testing_result_dir += f'p{pixel_weight}_per{perceptual_weight}_ms{use_multiscale}/'
        os.makedirs(testing_result_dir, exist_ok=True)
        
        all_labels, all_anomaly_maps, all_gt_masks, visualization_results = get_anomaly_maps_v3(
            vae_model=vae_model, 
            diffusion_model=diffusion_model, 
            method=method,
            pixel_weight=pixel_weight,
            perceptual_weight=perceptual_weight,
            use_multiscale=use_multiscale
        )
        
        results = run_inference_v3(all_labels, all_anomaly_maps, all_gt_masks, method=method)
        all_results.extend(results)

        # Save results and visualizations
        if _save_visualizations:
            save_visualization(
                testing_result_dir=testing_result_dir,
                visualization_results=visualization_results,
                max_images_per_batch=_max_images_per_batch,
                testing_category=_testing_category
            )
        
        results_path = os.path.join(testing_result_dir, f'inference_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

    # Save combined results
    combined_results_path = os.path.join(config.testing.test_result_base_dir + _testing_category, f'v3_combined_inference_results.json')
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll V3 results saved to: {combined_results_path}")
    
    return all_results


def run_single_method_v3(method='combined', pixel_weight=0.5, perceptual_weight=0.5, use_multiscale=True):
    """Run single V3 method."""
    set_seed(_seed)
    vae_model, diffusion_model = load_models()

    print(f'\n========== V3 {method.upper()} ===========')
    if method == 'combined':
        print(f'Pixel weight: {pixel_weight}, Perceptual weight: {perceptual_weight}, Multiscale: {use_multiscale}')
    
    testing_result_dir = config.testing.test_result_base_dir + _testing_category + f'/v3_{method}/'
    if method == 'combined':
        testing_result_dir += f'p{pixel_weight}_per{perceptual_weight}_ms{use_multiscale}/'
    os.makedirs(testing_result_dir, exist_ok=True)
    
    all_labels, all_anomaly_maps, all_gt_masks, visualization_results = get_anomaly_maps_v3(
        vae_model=vae_model, 
        diffusion_model=diffusion_model, 
        method=method,
        pixel_weight=pixel_weight,
        perceptual_weight=perceptual_weight,
        use_multiscale=use_multiscale
    )
    
    results = run_inference_v3(all_labels, all_anomaly_maps, all_gt_masks, method=method)

    # Save results and visualizations
    if _save_visualizations:
        save_visualization(
            testing_result_dir=testing_result_dir,
            visualization_results=visualization_results,
            max_images_per_batch=_max_images_per_batch,
            testing_category=_testing_category
        )
    
    results_path = os.path.join(testing_result_dir, f'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    return results


