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
    eval_auroc_pixel
)
import numpy as np
from vae.utils import load_vae_model
from utils.feature_extractor import VggFeatureExtractor
import torch.nn.functional as F
from diffusion.model.ddpm_model import DDPM

config = load_config()

_seed = config.general.seed
_image_score_type_name = config.testing.image_score_type_name
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
_device = torch.device(f'cuda:{config.general.cuda}' if torch.cuda.is_available() else 'cpu')

""" Load diffusion restoration model """
def load_diffusion_model(
    in_channel: int,
    inner_channel: int ,
    out_channel: int,
    norm_groups: int,
    channel_mults: list,
    attn_res: list,
    res_blocks: int,
    dropout_p: float,
    image_size: int,
    channels: int,
    loss_type: str,
    diffusion_model_path: str,
    device
):
    diffusion_model = DDPM(
        in_channel=in_channel,
        out_channel=out_channel,
        inner_channel=inner_channel,
        norm_groups=norm_groups,
        channel_mults=channel_mults,
        attn_res=attn_res,
        res_blocks=res_blocks,
        dropout_p=dropout_p,
        image_size=image_size,
        channels=channels,
        loss_type=loss_type
    )

    checkpoint = torch.load(diffusion_model_path, map_location=device)
    diffusion_model.netG.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.netG.to(device)
    diffusion_model.netG.eval()

    return diffusion_model

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
        device=_device
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
        device=_device
    )
    diffusion_model.set_noise_schedule_for_val()
    return vae_model, diffusion_model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_anomaly_map(
        x: torch.Tensor,
        x_recon: torch.Tensor,
        l1_weight,
        perceptual_weight,
        feature_extractor=None,
        use_perceptual=True
):
    # l1 pixel difference
    l1_map = (x - x_recon).abs().mean(dim=1)  # [B, H, W]
    if not use_perceptual:
        return l1_map

    # feature difference
    with torch.no_grad():
        feature_origin = feature_extractor.extract_features(x)  # [B, 256, H/8, W/8]
        feature_recon = feature_extractor.extract_features(x_recon)

        perceptual_diff = (feature_origin - feature_recon).abs().mean(dim=1) # [B, H/8, W/8]
        perceptual_map = F.interpolate(
            perceptual_diff.unsqueeze(1), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False
        ).squeeze(1)  # [B, H, W]

    anomaly_map = l1_weight * l1_map + perceptual_weight * perceptual_map
    return anomaly_map


def get_anomaly_maps(use_perceptual: bool):
    set_seed(_seed)
    os.makedirs(_testing_result_dir, exist_ok=True)

    vae_model, diffusion_model = load_models()
    print('Load vae, diffusion models success')
    feature_extractor = VggFeatureExtractor(_device) if use_perceptual else None
    if use_perceptual:
        print('Load Vgg16 success')

    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=_mvtec_data_dir,
        category=_testing_category,
        image_size=_image_size,
        batch_size=_test_batch_size
    )
    print(f"Test on {_testing_category} category")

    batch_count = 0
    all_labels = []
    all_anomaly_maps = []
    all_gt_masks = []
    visualization_results = {
        'input_images': [],
        'vae_reconstructions': [],
        'diffusion_reconstructions': [],
        'anomaly_maps': [],
        'gt_masks': []
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="run inference ")):
            images = batch['image'].to(_device)  # [B, C, H, W]
            masks = batch['mask'].to(_device)  # [B, 1, H, W]
            labels = batch['label'].to(_device)  # [B]

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

            # Compute anomaly maps
            anomaly_maps = compute_anomaly_map(
                x=images,
                x_recon=diffusion_reconstructions,
                feature_extractor=feature_extractor,
                l1_weight=0.6,
                perceptual_weight=0.4,
                use_perceptual=use_perceptual
            )

            all_labels.extend(labels.cpu().numpy().tolist())
            all_anomaly_maps.extend([am.cpu() for am in anomaly_maps])
            all_gt_masks.extend([mask.squeeze(0).cpu() for mask in masks])  # Remove channel dim

            # Store visualization data, limit max batches
            if batch_count < _max_visualization_batches:
                max_images = min(images.size(0), _max_images_per_batch)
                visualization_results['input_images'].append(images[:max_images])
                visualization_results['vae_reconstructions'].append(vae_reconstructions[:max_images])
                visualization_results['diffusion_reconstructions'].append(diffusion_reconstructions[:max_images])
                visualization_results['anomaly_maps'].append(anomaly_maps[:max_images])
                visualization_results['gt_masks'].append(masks[:max_images])
                batch_count += 1

    return all_labels, all_anomaly_maps, all_gt_masks, visualization_results


def run_inference(all_labels, all_anomaly_maps, all_gt_masks, use_perceptual: bool):
    results = []
    score_types = ['max', 'mean', 'std']

    pixel_auroc = eval_auroc_pixel(all_anomaly_maps, all_gt_masks)
    for score_type in score_types:
        print(f'-- Type: {score_type} --')
        image_scores = calc_image_score(all_anomaly_maps, score_type)
        image_auroc = eval_auroc_image(all_labels, image_scores.cpu().numpy().tolist())

        evaluation_result = {
            'category': _testing_category,
            'method': 'l1 + feature' if use_perceptual else 'l1',
            'image_auroc': image_auroc,
            'pixel_auroc': pixel_auroc,
            'image_score_type': score_type,
            'total_samples': len(all_labels),
            'anomaly_samples': sum(all_labels),
            'normal_samples': len(all_labels) - sum(all_labels),
            'feature_extractor': 'VGG16' if use_perceptual else None,
            'anomaly_map_weights': {
                'l1_weight': 0.6,
                'perceptual_weight': 0.4 if use_perceptual else 0.0
            }
        }

        print(f"  Image AUROC: {image_auroc:.4f}")
        print(f"  Pixel AUROC: {pixel_auroc:.4f}")
        print(f"  Total samples: {len(all_labels)}")
        print(f"  Anomaly samples: {sum(all_labels)}")
        print(f"  Normal samples: {len(all_labels) - sum(all_labels)}")
        results.append(evaluation_result)

    return results


def run_all_inference():
    all_results = []
    testing_result_dir = config.testing.test_result_base_dir + _testing_category + '/l1/'

    print('========== Only l1 diff ===========')
    all_labels, all_anomaly_maps_l1, all_gt_masks, visualization_results = get_anomaly_maps(use_perceptual=False)
    l1_results = run_inference(all_labels, all_anomaly_maps_l1, all_gt_masks, use_perceptual=False)
    all_results.extend(l1_results)

    # Save visualizations if enabled
    if _save_visualizations:
        save_visualization(
            testing_result_dir=testing_result_dir,
            visualization_results=visualization_results,
            max_images_per_batch=_max_images_per_batch,
            testing_category=_testing_category
        )
    results_path = os.path.join(testing_result_dir, f'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)


    testing_result_dir = config.testing.test_result_base_dir + _testing_category + '/l1_feature/'
    print('========= Use perceptual diff ==============')
    all_labels, all_anomaly_maps, all_gt_masks, visualization_results = get_anomaly_maps(use_perceptual=True)
    perceptual_results = run_inference(all_labels, all_anomaly_maps, all_gt_masks, use_perceptual=True)
    all_results.extend(perceptual_results)

    # Save visualizations if enabled
    if _save_visualizations:
        save_visualization(
            testing_result_dir=testing_result_dir,
            visualization_results=visualization_results,
            max_images_per_batch=_max_images_per_batch,
            testing_category=_testing_category
        )
    results_path = os.path.join(testing_result_dir, f'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {results_path}")
    return all_results

