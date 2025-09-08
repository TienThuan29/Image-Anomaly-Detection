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
    compute_anomaly_map
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
_device = torch.device(f'cuda:{config.general.cuda}' if torch.cuda.is_available() else 'cpu')


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


def get_anomaly_maps(vae_model, diffusion_model, is_visualization=True):
    print('Load vae, diffusion models success')

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
                x_recon=diffusion_reconstructions
            )

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
                    batch_count += 1

    return all_labels, all_anomaly_maps, all_gt_masks, visualization_results


def run_inference(all_labels, all_anomaly_maps, all_gt_masks):
    results = []
    score_types = ['max', 'mean', 'std']

    pixel_auroc = eval_auroc_pixel(all_anomaly_maps, all_gt_masks)
    for score_type in score_types:
        print(f'-- Type: {score_type} --')
        image_scores = calc_image_score(all_anomaly_maps, score_type)
        image_auroc = eval_auroc_image(all_labels, image_scores.cpu().numpy().tolist())

        evaluation_result = {
            'category': _testing_category,
            'method': 'l1',
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


def run_all_inference():
    set_seed(_seed)
    all_results = []
    testing_result_dir = config.testing.test_result_base_dir + _testing_category + '/l1/'

    os.makedirs(testing_result_dir, exist_ok=True)
    vae_model, diffusion_model = load_models()

    print('========== L1 diff ===========')
    all_labels, all_anomaly_maps_l1, all_gt_masks, visualization_results = get_anomaly_maps(vae_model=vae_model, diffusion_model=diffusion_model)
    l1_results = run_inference(all_labels, all_anomaly_maps_l1, all_gt_masks)
    all_results.extend(l1_results)

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


def run_inference_during_training(vae_model, diffusion_model, epoch):
    set_seed(_seed)
    all_results = []
    testing_result_dir = config.testing.test_result_base_dir + _testing_category + '/l1/'
    os.makedirs(testing_result_dir, exist_ok=True)
    print("======= Inference during the training =======")
    all_labels, all_anomaly_maps_l1, all_gt_masks, visualization_results = get_anomaly_maps(
        vae_model=vae_model,
        diffusion_model=diffusion_model,
        is_visualization=False
    )
    l1_results = run_inference(all_labels, all_anomaly_maps_l1, all_gt_masks)
    all_results.extend(l1_results)

    results_path = os.path.join(testing_result_dir, f'inference_at_epoch_{epoch}.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)


# import os
# import torch
# import numpy as np
# from tqdm import tqdm
# from config import load_config
# from data.dataloader import load_mvtec_test_dataset
# from testing.visualization import (
#     create_visualization_grid,
#     save_visualization_results,
#     tensor_to_numpy,
#     create_heatmap,
#     overlay_heatmap_on_image
# )
# import json
# import matplotlib.pyplot as plt
# from testing.metric import (
#     calc_image_score,
#     eval_auroc_image,
#     eval_auroc_pixel,
#     compute_anomaly_map
# )
# from vae.utils import load_vae_model
# from diffusion.utils import load_diffusion_model
#
# config = load_config()
#
# _seed = config.general.seed
# _image_score_type_name = config.testing.image_score_type_name
# _testing_category = config.testing.category
# _image_size = config.general.image_size
# _mvtec_data_dir = config.data.mvtec_data_dir
# # Model path
# _vae_model_path = config.testing.vae_model_path
# _diffusion_model_path = config.testing.diffusion_model_path
# # Save result dir
# _testing_result_dir = config.testing.test_result_base_dir + _testing_category + '/'
# # Visualization
# _save_visualizations = config.testing.save_visualizations
# _max_visualization_batches = config.testing.max_visualization_batches
# _max_images_per_batch = config.testing.max_images_per_batch
#
# # vae
# _vae_name = config.vae_model.name
# _backbone = config.vae_model.backbone
# _input_channels = config.vae_model.in_channels
# _output_channels = config.vae_model.out_channels
# _z_dim = config.vae_model.z_dim
# _dropout_p = config.vae_model.dropout_p
#
# # test batch size
# _test_batch_size = config.testing.batch_size
#
# # Device
# _device = torch.device(f'cuda:{config.general.cuda}' if torch.cuda.is_available() else 'cpu')
#
# def load_models():
#     vae_model = load_vae_model(
#         checkpoint_path=_vae_model_path,
#         vae_name=_vae_name,
#         input_channels=_input_channels,
#         output_channels=_output_channels,
#         z_dim=_z_dim,
#         backbone=_backbone,
#         dropout_p=_dropout_p,
#         image_size=_image_size,
#         device=_device
#     )
#     diffusion_model = load_diffusion_model(
#         in_channel=config.diffusion_model.unet.in_channel,
#         out_channel=config.diffusion_model.unet.out_channel,
#         inner_channel=config.diffusion_model.unet.inner_channel,
#         norm_groups=config.diffusion_model.unet.norm_groups,
#         channel_mults=config.diffusion_model.unet.channel_mults,
#         attn_res=config.diffusion_model.unet.attn_res,
#         res_blocks=config.diffusion_model.unet.res_blocks,
#         dropout_p=config.diffusion_model.unet.dropout,
#         image_size=config.diffusion_model.diffusion.image_size,
#         channels=config.diffusion_model.diffusion.channels,
#         loss_type=config.diffusion_model.loss_type,
#         diffusion_model_path=_diffusion_model_path,
#         device=_device
#     )
#     diffusion_model.set_noise_schedule_for_val()
#     return vae_model, diffusion_model
#
#
# def run_inference():
#     set_seed(_seed)
#     os.makedirs(_testing_result_dir, exist_ok=True)
#
#     vae_model, diffusion_model = load_models()
#
#     test_loader = load_mvtec_test_dataset(
#         dataset_root_dir=_mvtec_data_dir,
#         category=_testing_category,
#         image_size=_image_size,
#         batch_size=_test_batch_size
#     )
#     print(f"Testing on {_testing_category} category with {len(test_loader)} batches")
#
#     all_labels = []
#     all_image_scores = []
#     all_image_scores_max = []
#     all_image_scores_mean = []
#     all_image_scores_std = []
#     all_anomaly_maps = []
#     all_gt_masks = []
#
#     visualization_results = {
#         'input_images': [],
#         'vae_reconstructions': [],
#         'diffusion_reconstructions': [],
#         'anomaly_maps': [],
#         'gt_masks': []
#     }
#     batch_count = 0
#
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running inference")):
#
#             # Extract data from batch dictionary
#             images = batch['image'].to(_device)  # [B, C, H, W]
#             masks = batch['mask'].to(_device)    # [B, 1, H, W]
#             labels = batch['label'].to(_device)  # [B]
#
#             vae_reconstructions = vae_model.reconstruct(images)  # [B, C, H, W]
#
#             if config.diffusion_model.diffusion.conditional:
#                 diffusion_reconstructions = diffusion_model.netG.super_resolution(
#                     vae_reconstructions,
#                     continous=False
#                 )  # [B, C, H, W]
#             else:
#                 diffusion_reconstructions = diffusion_model.netG.sample(
#                     batch_size=images.size(0),
#                     continous=False
#                 )  # [B, C, H, W]
#
#             anomaly_maps = compute_anomaly_map(images, diffusion_reconstructions)  # [B, H, W]
#             image_scores = calc_image_score(anomaly_maps, _image_score_type_name)  # [B] or tuple([B],[B],[B])
#
#             # Store results
#             all_labels.extend(labels.cpu().numpy().tolist())
#             if isinstance(image_scores, tuple):
#                 max_scores, mean_scores, std_scores = image_scores
#                 all_image_scores_max.extend(max_scores.cpu().numpy().tolist())
#                 all_image_scores_mean.extend(mean_scores.cpu().numpy().tolist())
#                 all_image_scores_std.extend(std_scores.cpu().numpy().tolist())
#             else:
#                 all_image_scores.extend(image_scores.cpu().numpy().tolist())
#             all_anomaly_maps.extend([am.cpu() for am in anomaly_maps])
#             all_gt_masks.extend([mask.squeeze(0).cpu() for mask in masks])  # Remove channel dimension
#
#             # Store visualization data (limit to max batches)
#             if batch_count < _max_visualization_batches:
#                 max_images = min(images.size(0), _max_images_per_batch)
#
#                 visualization_results['input_images'].append(images[:max_images])
#                 visualization_results['vae_reconstructions'].append(vae_reconstructions[:max_images])
#                 visualization_results['diffusion_reconstructions'].append(diffusion_reconstructions[:max_images])
#                 visualization_results['anomaly_maps'].append(anomaly_maps[:max_images])
#                 visualization_results['gt_masks'].append(masks[:max_images])
#
#                 batch_count += 1
#
#     # Calculate metrics
#     if _image_score_type_name == 'all':
#         image_auroc = {
#             'max': eval_auroc_image(all_labels, all_image_scores_max),
#             'mean': eval_auroc_image(all_labels, all_image_scores_mean),
#             'std': eval_auroc_image(all_labels, all_image_scores_std)
#         }
#     else:
#         image_auroc = eval_auroc_image(all_labels, all_image_scores)
#     pixel_auroc = eval_auroc_pixel(all_anomaly_maps, all_gt_masks)
#
#     # Save evaluation results
#     evaluation_results = {
#         'category': _testing_category,
#         'image_auroc': image_auroc,
#         'pixel_auroc': pixel_auroc,
#         'image_score_type': _image_score_type_name,
#         'total_samples': len(all_labels),
#         'anomaly_samples': sum(all_labels),
#         'normal_samples': len(all_labels) - sum(all_labels)
#     }
#
#     results_path = os.path.join(_testing_result_dir, 'evaluation_results.json')
#     with open(results_path, 'w') as f:
#         json.dump(evaluation_results, f, indent=2)
#
#     print(f"Evaluation Results:")
#     print(f"  Category: {_testing_category}")
#     if isinstance(image_auroc, dict):
#         print(f"  Image AUROC (max): {image_auroc['max']:.4f}")
#         print(f"  Image AUROC (mean): {image_auroc['mean']:.4f}")
#         print(f"  Image AUROC (std): {image_auroc['std']:.4f}")
#     else:
#         print(f"  Image AUROC: {image_auroc:.4f}")
#     print(f"  Pixel AUROC: {pixel_auroc:.4f}")
#     print(f"  Total samples: {len(all_labels)}")
#     print(f"  Anomaly samples: {sum(all_labels)}")
#     print(f"  Normal samples: {len(all_labels) - sum(all_labels)}")
#     print(f"Results saved to: {results_path}")
#
#     # Save visualizations if enabled
#     if _save_visualizations:
#         print("Saving visualizations...")
#         vis_dir = os.path.join(_testing_result_dir, 'visualizations')
#         os.makedirs(vis_dir, exist_ok=True)
#
#         for batch_idx in range(len(visualization_results['input_images'])):
#             # Create visualization grid
#             fig = create_visualization_grid(
#                 input_images=visualization_results['input_images'][batch_idx],
#                 vae_reconstructions=visualization_results['vae_reconstructions'][batch_idx],
#                 diffusion_reconstructions=visualization_results['diffusion_reconstructions'][batch_idx],
#                 anomaly_maps=visualization_results['anomaly_maps'][batch_idx],
#                 gt_masks=visualization_results['gt_masks'][batch_idx],
#                 max_images=_max_images_per_batch
#             )
#
#             # Save grid
#             grid_path = os.path.join(vis_dir, f'batch_{batch_idx:03d}_grid.png')
#             fig.savefig(grid_path, dpi=150, bbox_inches='tight')
#             plt.close(fig)
#
#             # Save individual components
#             batch_vis_dir = os.path.join(vis_dir, f'batch_{batch_idx:03d}')
#             save_visualization_results(
#                 results={
#                     'input_images': [tensor_to_numpy(img) for img in visualization_results['input_images'][batch_idx]],
#                     'vae_reconstructions': [tensor_to_numpy(img) for img in visualization_results['vae_reconstructions'][batch_idx]],
#                     'diffusion_reconstructions': [tensor_to_numpy(img) for img in visualization_results['diffusion_reconstructions'][batch_idx]],
#                     'heatmaps': [create_heatmap(am, am.shape[0], am.shape[1]) for am in visualization_results['anomaly_maps'][batch_idx]],
#                     'overlays': [overlay_heatmap_on_image(
#                         tensor_to_numpy(visualization_results['input_images'][batch_idx][i]),
#                         create_heatmap(visualization_results['anomaly_maps'][batch_idx][i],
#                                      visualization_results['input_images'][batch_idx][i].shape[1],
#                                      visualization_results['input_images'][batch_idx][i].shape[2])
#                     ) for i in range(len(visualization_results['input_images'][batch_idx]))],
#                     'gt_masks': [tensor_to_numpy(mask) for mask in visualization_results['gt_masks'][batch_idx]]
#                 },
#                 save_dir=batch_vis_dir,
#                 category_name=_testing_category,
#                 batch_idx=batch_idx
#             )
#
#         print(f"Visualizations saved to: {vis_dir}")
#
#
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#
# @torch.no_grad()
# def run_inference_during_training(vae_model, diffusion_model, result_dir=None):
#     """Run inference during training with optional custom result directory."""
#
#     # Define device
#     device = torch.device(f'cuda:{config.general.cuda}' if torch.cuda.is_available() else 'cpu')
#
#     # Ensure models are on the correct device and in eval mode
#     vae_model = vae_model.to(device)
#     diffusion_model.netG = diffusion_model.netG.to(device)
#     vae_model.eval()
#     diffusion_model.netG.eval()
#
#     # Use provided result_dir or default to testing result dir
#     save_dir = result_dir if result_dir is not None else _testing_result_dir
#     os.makedirs(save_dir, exist_ok=True)
#
#     test_loader = load_mvtec_test_dataset(
#         dataset_root_dir=config.data.mvtec_data_dir,
#         category=_testing_category,
#         image_size=config.general.image_size,
#         batch_size=config.general.batch_size
#     )
#     print(f"Testing on {_testing_category} category with {len(test_loader)} batches")
#
#     all_labels = []
#     all_image_scores = []
#     all_image_scores_max = []
#     all_image_scores_mean = []
#     all_image_scores_std = []
#     all_anomaly_maps = []
#     all_gt_masks = []
#
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running inference")):
#
#             # Extract data from batch dictionary
#             images = batch['image'].to(device)  # [B, C, H, W]
#             masks = batch['mask'].to(device)  # [B, 1, H, W]
#             labels = batch['label'].to(device)  # [B]
#
#             vae_reconstructions, _, _ = vae_model(images)  # [B, C, H, W]
#
#             # Diffusion Restoration using VAE reconstructions as input
#             if config.diffusion_model.diffusion.conditional:
#                 diffusion_reconstructions = diffusion_model.netG.super_resolution(
#                     vae_reconstructions,
#                     continous=False
#                 )  # [B, C, H, W]
#             else:
#                 # For non-conditional, use regular sample method
#                 diffusion_reconstructions = diffusion_model.netG.sample(
#                     batch_size=images.size(0),
#                     continous=False
#                 )  # [B, C, H, W]
#
#             # Calculate anomaly maps (difference between original and diffusion output)
#             anomaly_maps = compute_anomaly_map(images, diffusion_reconstructions)  # [B, H, W]
#
#             # Calculate image-level scores
#             image_scores = calc_image_score(anomaly_maps, _image_score_type_name)  # [B] or tuple([B],[B],[B])
#
#             # Store results
#             all_labels.extend(labels.cpu().numpy().tolist())
#             if isinstance(image_scores, tuple):
#                 max_scores, mean_scores, std_scores = image_scores
#                 all_image_scores_max.extend(max_scores.cpu().numpy().tolist())
#                 all_image_scores_mean.extend(mean_scores.cpu().numpy().tolist())
#                 all_image_scores_std.extend(std_scores.cpu().numpy().tolist())
#             else:
#                 all_image_scores.extend(image_scores.cpu().numpy().tolist())
#             all_anomaly_maps.extend([am.cpu() for am in anomaly_maps])
#             all_gt_masks.extend([mask.squeeze(0).cpu() for mask in masks])  # Remove channel dimension
#
#     # Calculate metrics
#     if _image_score_type_name == 'all':
#         image_auroc = {
#             'max': eval_auroc_image(all_labels, all_image_scores_max),
#             'mean': eval_auroc_image(all_labels, all_image_scores_mean),
#             'std': eval_auroc_image(all_labels, all_image_scores_std)
#         }
#     else:
#         image_auroc = eval_auroc_image(all_labels, all_image_scores)
#     pixel_auroc = eval_auroc_pixel(all_anomaly_maps, all_gt_masks)
#
#     # Save evaluation results
#     evaluation_results = {
#         'category': _testing_category,
#         'image_auroc': image_auroc,
#         'pixel_auroc': pixel_auroc,
#         'image_score_type': _image_score_type_name,
#         'total_samples': len(all_labels),
#         'anomaly_samples': sum(all_labels),
#         'normal_samples': len(all_labels) - sum(all_labels)
#     }
#
#     results_path = os.path.join(save_dir, 'evaluation_results.json')
#     with open(results_path, 'w') as f:
#         json.dump(evaluation_results, f, indent=2)
#
#     print(f"Evaluation Results:")
#     print(f"  Category: {_testing_category}")
#     if isinstance(image_auroc, dict):
#         print(f"  Image AUROC (max): {image_auroc['max']:.4f}")
#         print(f"  Image AUROC (mean): {image_auroc['mean']:.4f}")
#         print(f"  Image AUROC (std): {image_auroc['std']:.4f}")
#     else:
#         print(f"  Image AUROC: {image_auroc:.4f}")
#     print(f"  Pixel AUROC: {pixel_auroc:.4f}")
#     print(f"  Total samples: {len(all_labels)}")
#     print(f"  Anomaly samples: {sum(all_labels)}")
#     print(f"  Normal samples: {len(all_labels) - sum(all_labels)}")
#     print(f"Results saved to: {results_path}")
#
#     return image_auroc, pixel_auroc
#