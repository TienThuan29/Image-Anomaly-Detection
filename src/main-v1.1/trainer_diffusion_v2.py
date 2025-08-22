import os
import time
import torch
import numpy as np
import random
from diffusion_gaussian import GaussianDiffusion
from utils import load_mvtec_train_dataset, load_mvtec_test_dataset
from diffusion_model import UNetModel
from tqdm import tqdm
from config import load_config
from utils import get_optimizer, load_vae
from typing import List
from reconstruction import Reconstruction
from inference_v2 import compute_anomaly_map, eval_auroc_image, eval_auroc_pixel, to_label_list, guided_reconstruction, to_batch_tensor

config = load_config("config.yml")

# general
_seed = config.general.seed
_cuda = config.general.cuda
_image_size = config.general.image_size
_batch_size = config.general.batch_size
_input_channels = config.general.input_channels # 3
_output_channels = config.general.output_channels # 3

# data
_category_name = config.data.category
_mvtec_data_dir = config.data.mvtec_data_dir
_mask = config.data.mask

# vae
_vae_name = config.vae_model.name
_backbone = config.vae_model.backbone

# diffusion
_diffusion_name = config.diffusion_model.name
_lr = float(config.diffusion_model.lr)
_weight_decay = float(config.diffusion_model.weight_decay)
_epochs = config.diffusion_model.epochs
_dropout_p = config.diffusion_model.dropout_p
_z_dim = config.diffusion_model.z_dim
_num_timesteps = config.diffusion_model.num_timesteps
_optimizer_name = config.diffusion_model.optimizer_name
_beta_schedule = config.diffusion_model.beta_schedule
_phase1_vae_pretrained_path = config.diffusion_model.phase1_vae_pretrained_path
_w = config.diffusion_model.w

# save train results
_train_result_dir = config.diffusion_model.train_result_base_dir + _category_name + "/"
_pretrained_save_dir = config.diffusion_model.pretrained_save_base_dir + _category_name + "/"

# eval
_eval_interval = config.diffusion_model.eval_interval


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def loss_function(pred_noise, noise):
    return (noise - pred_noise).square().sum(dim=(1, 2, 3)).mean(dim=0)


@torch.no_grad()
def evaluate_model(model, vae_model, gaussian_diffusion, device):
    """Run evaluation on test dataset using guided reconstruction (v2)."""
    print("\n[INFO] Running evaluation (guided v2)...")
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
        # VAE reconstruction
        recon_vae, _, _ = vae_model(images)
        # Guided reconstruction using diffusion UNet
        reconstruction = Reconstruction(model, device)
        images_reconstructed = reconstruction(
            x = recon_vae,
            y0 = images,
            w = _w
        )
        # Convert list to tensor
        images_reconstructed = to_batch_tensor(images_reconstructed, images)

        # Anomaly map and image-level score
        B = images.size(0)
        anomaly_map = compute_anomaly_map(images, images_reconstructed)  # output: [B,H,W]
        image_scores = anomaly_map.view(B, -1).max(dim=1).values  # max pooling

        # Accumulate
        labels_all.extend(to_label_list(labels))
        img_scores_all.extend(image_scores.detach().cpu().tolist())
        maps_all.extend(list(anomaly_map.detach().cpu()))
        gts_all.extend(list(masks.detach().cpu().squeeze(1)))

    # Calculate metrics
    img_auroc = eval_auroc_image(labels_all, img_scores_all)
    px_auroc = eval_auroc_pixel(maps_all, gts_all)

    print(f"[EVAL] Image AUROC: {img_auroc:.4f}, Pixel AUROC: {px_auroc:.4f}")
    return img_auroc, px_auroc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=*200")
    print(f"Using device: {device}")
    print("Category: ", _category_name)
    print("="*200)

    os.makedirs(_pretrained_save_dir, exist_ok=True)
    os.makedirs(_train_result_dir, exist_ok=True)

    # Load VAE model
    print("[INFO] Loading VAE model...")
    vae_model = load_vae(
        checkpoint_path=_phase1_vae_pretrained_path,
        vae_name=_vae_name,
        input_channels=_input_channels,
        output_channels=_output_channels,
        z_dim=_z_dim,
        backbone=_backbone,
        dropout_p=_dropout_p,
        image_size=_image_size,
        device=device
    )

    # Load dataset
    print("[INFO] Loading dataset...")
    train_dataset = load_mvtec_train_dataset(
        dataset_root_dir=_mvtec_data_dir,
        category=_category_name,
        image_size=_image_size,
        batch_size=_batch_size
    )

    # Initialize UNet model
    print("[INFO] Initializing UNet model...")
    model = UNetModel(
        img_size=_image_size,
        base_channels=32,
        n_heads=2,
        num_res_blocks=2,
        dropout=0.1,
        attention_resolutions="32,16,8",
        biggan_updown=True,
        in_channels=_input_channels,
    ).to(device)

    gaussian_diffusion = GaussianDiffusion(num_timesteps=_num_timesteps,beta_schedule=_beta_schedule)

    optimizer = get_optimizer(
        optimizer_name=_optimizer_name,
        params=model.parameters(),
        lr=_lr,
        weight_decay=_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_epochs)

    start_epoch = 0
    total_epochs = _epochs
    loss_history = []
    eval_history = {'img_auroc': [], 'px_auroc': [], 'epochs': []}

    # Init eval
    eval_history['img_auroc'] = []
    eval_history['px_auroc'] = []
    eval_history['epochs'].append(0)

    print(f"Starting training for {total_epochs} epochs...")
    # Main epoch progress bar
    epoch_bar = tqdm(range(start_epoch, total_epochs),desc="Training Progress",position=0,leave=True)

    for epoch in epoch_bar:
        model.train()
        vae_model.eval()
        epoch_start_time = time.time()
        batch_losses = []
        batch_bar = tqdm(train_dataset,desc=f"Epoch {epoch + 1}/{total_epochs}",position=1,leave=False,)

        for batch_idx, batch in enumerate(batch_bar):
            images = batch['image'].to(device)

            # Get reconstruct image from vae model phase 1
            with torch.no_grad():
                vae_reconstructed, _, _ = vae_model(images)

            B = images.size(0)
            t = torch.randint(0, _num_timesteps, (B,), device=device)
            noise = torch.randn_like(vae_reconstructed).to(device)
            # x_t: add noise
            x_t = gaussian_diffusion.q_sample(vae_reconstructed, t, noise)

            # Predict noise
            pred_noise = model(x_t, t)

            # Calculate loss
            total_loss = loss_function(pred_noise=pred_noise, noise=noise)
            batch_losses.append(total_loss.item())

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_bar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # Calculate epoch metrics
        epoch_loss = np.mean(batch_losses)
        loss_history.append(epoch_loss)
        epoch_time = time.time() - epoch_start_time

        scheduler.step()

        eval_info = ""
        if (epoch + 1) % _eval_interval == 0 and epoch != 1 and epoch != 0:
            img_auroc, px_auroc = evaluate_model(model, vae_model, gaussian_diffusion, device)
            eval_history['img_auroc'].append(img_auroc)
            eval_history['px_auroc'].append(px_auroc)
            eval_history['epochs'].append(epoch + 1)
            eval_info = f" | Img AUROC: {img_auroc:.4f} | Px AUROC: {px_auroc:.4f}"

        # Update epoch progress bar with comprehensive info
        epoch_bar.set_postfix({
            'Loss': f'{epoch_loss:.6f}',
            'Time': f'{epoch_time:.1f}s',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'Best Img AUROC': f'{max(eval_history["img_auroc"]):.4f}' if eval_history['img_auroc'] else 'N/A'
        })

        # Log epoch results
        if (epoch + 1) % 10 == 0 and epoch != 1 and epoch != 0:
            print(f"\n[INFO] Epoch {epoch + 1}/{total_epochs} completed:")
            print(f"        Loss: {epoch_loss:.6f}")
            print(f"        Time: {epoch_time:.1f}s")
            print(f"        Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            if eval_info:
                print(f"        {eval_info}")
                eval_checkpoint = {
                    'epoch': total_epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'config': {
                        'image_size': _image_size,
                        'input_channels': _input_channels,
                        'num_timesteps': _num_timesteps,
                        'beta_schedule': _beta_schedule
                    }
                }
                torch.save(eval_checkpoint, os.path.join(_pretrained_save_dir, f'diffusion_model_{epoch+1}_.pth'))

        # Save checkpoint periodically
        if (epoch + 1) % 50 == 0:
            os.makedirs(_pretrained_save_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_history': loss_history,
                'config': {
                    'image_size': _image_size,
                    'input_channels': _input_channels,
                    'num_timesteps': _num_timesteps,
                    'beta_schedule': _beta_schedule
                }
            }
            torch.save(checkpoint, os.path.join(_pretrained_save_dir, f'diffusion_model_epoch_{epoch+1}.pth'))
            print(f"[INFO] Checkpoint saved at epoch {epoch+1}")


    # Final evaluation
    print("\n[INFO] Running final evaluation...")
    final_img_auroc, final_px_auroc = evaluate_model(model, vae_model, gaussian_diffusion, device)
    eval_history['img_auroc'].append(final_img_auroc)
    eval_history['px_auroc'].append(final_px_auroc)
    eval_history['epochs'].append(total_epochs)

    # Training completed
    print(f"\n[INFO] Training completed!")
    print(f"        Final loss: {loss_history[-1]:.6f}")
    print(f"        Best loss: {min(loss_history):.6f}")
    print(f"        Final Image AUROC: {final_img_auroc:.4f}")
    print(f"        Final Pixel AUROC: {final_px_auroc:.4f}")
    print(f"        Best Image AUROC: {max(eval_history['img_auroc']):.4f}")
    print(f"        Best Pixel AUROC: {max(eval_history['px_auroc']):.4f}")

    # Save final model
    os.makedirs(_pretrained_save_dir, exist_ok=True)
    final_checkpoint = {
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
        'config': {
            'image_size': _image_size,
            'input_channels': _input_channels,
            'num_timesteps': _num_timesteps,
            'beta_schedule': _beta_schedule
        }
    }
    torch.save(final_checkpoint, os.path.join(_pretrained_save_dir, 'diffusion_model_final.pth'))
    print(f"[INFO] Final model saved to {_pretrained_save_dir}")


if __name__ == "__main__":
    set_seed(_seed)
    main()




