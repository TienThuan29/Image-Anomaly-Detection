import os
import time
import torch
import numpy as np
import random
import torch.nn.functional as F
from vae_resnet_model import VAEResNet
from vae_unet_model import VAEUnet
from diffusion_gaussian import GaussianDiffusion
from utils import load_mvtec_train_dataset, load_mvtec_test_dataset
from diffusion_model import UNetModel
from tqdm import tqdm
from config import load_config
from utils import get_optimizer
from typing import List
from inference import compute_anomaly_map, normalize_maps_global, eval_auroc_image, eval_auroc_pixel, to_label_list


config = load_config("config.yml")

# general
seed = config.general.seed
cuda = config.general.cuda
image_size = config.general.image_size
batch_size = config.general.batch_size
input_channels = config.general.input_channels # 3
output_channels = config.general.output_channels # 3

# data
category_name = config.data.category
mvtec_data_dir = config.data.mvtec_data_dir
mask = config.data.mask

# vae
vae_name = config.vae_model.name
backbone = config.vae_model.backbone

# diffusion
diffusion_name = config.diffusion_model.name
lr = float(config.diffusion_model.lr)
weight_decay = float(config.diffusion_model.weight_decay)
epochs = config.diffusion_model.epochs
dropout_p = config.diffusion_model.dropout_p
z_dim = config.diffusion_model.z_dim
num_timesteps = config.diffusion_model.num_timesteps
optimizer_name = config.diffusion_model.optimizer_name
beta_schedule = config.diffusion_model.beta_schedule
phase1_vae_pretrained_path = config.diffusion_model.phase1_vae_pretrained_path

# save train results
train_result_dir = config.diffusion_model.train_result_base_dir + category_name + "/"
pretrained_save_dir = config.diffusion_model.pretrained_save_base_dir + category_name + "/"

# eval
eval_interval = config.diffusion_model.eval_interval


def load_vae(checkpoint_path: str, device: torch.device):
    if vae_name == 'vae_resnet':
        print("VAE model name: vae_resnet")
        model = VAEResNet(
            image_size=image_size,
            in_channels=input_channels,
            out_channels=output_channels,
            latent_dim=z_dim,
            resnet_name=backbone,
            dropout_p=dropout_p
        ).to(device)
    elif vae_name == 'vae_unet':
        print("VAE model name: vae_unet")
        model = VAEUnet(
            in_channels=input_channels,
            latent_dim=z_dim,
            out_channels=output_channels
        ).to(device)
    else:
        raise ValueError(f"Unknown vae model: {vae_name}")
    try:
        import numpy as np
        safe_globals = [
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np.float32, np.float64, np.int32, np.int64,
            np.bool_, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64
        ]
        torch.serialization.add_safe_globals(safe_globals)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print(f"[INFO] Successfully loaded {checkpoint_path} with weights_only=True")

    except Exception as e:
        # If weights_only=True still fails, use the fallback but warn about security
        print(f"[WARN] Secure loading failed, using fallback (potential security risk)")
        print(f"[WARN] Error details: {str(e)[:200]}...")  # Truncate long error messages
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both direct state_dict and nested checkpoint formats
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt  # Assume it's a direct state_dict

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing}, Unexpected keys: {unexpected}")

    model.eval()
    return model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def loss_function(z_predict, z_origin):
    return F.mse_loss(z_predict, z_origin, reduction='mean')


@torch.no_grad()
def evaluate_model(model, vae_model, gaussian_diffusion, device):
    """Run evaluation on test dataset."""
    print("\n[INFO] Running evaluation...")

    # Load test dataset
    test_loader = load_mvtec_test_dataset(
        dataset_root_dir=mvtec_data_dir,
        category=category_name,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # Set models to eval mode
    model.eval()
    vae_model.eval()

    # Accumulators
    labels_all: List[int] = []
    img_scores_all: List[float] = []
    maps_all: List[torch.Tensor] = []
    gts_all: List[torch.Tensor] = []

    # Fixed t for inference (can be tuned). Use moderately noisy step.
    infer_t = max(1, int(0.4 * num_timesteps))

    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)  # [B,1,H,W]
        labels = batch['label']  # list/int/tensor

        # VAE reconstruction
        recon_vae, _, _ = vae_model(images)

        # Add noise at step t and denoise via UNet to predict x0
        B = images.size(0)
        t = torch.full((B,), infer_t, dtype=torch.long, device=device)
        noise = torch.randn_like(recon_vae)
        x_t = gaussian_diffusion.q_sample(recon_vae, t, noise)
        pred_noise = model(x_t, t)
        recon = gaussian_diffusion.predict_start_from_noise(x_t, t, pred_noise).clamp(0, 1)

        # Anomaly map and image-level score
        anomaly_map = compute_anomaly_map(images, recon)  # [B,H,W]
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


def main(batch_size=batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=*200")
    print(f"Using device: {device}")
    print("Category: ", category_name)
    print("=*200")

    os.makedirs(pretrained_save_dir, exist_ok=True)
    os.makedirs(train_result_dir, exist_ok=True)

    # Load VAE model
    print("[INFO] Loading VAE model...")
    vae_model = load_vae(checkpoint_path=phase1_vae_pretrained_path,device=device)

    # Load dataset
    print("[INFO] Loading dataset...")
    train_dataset = load_mvtec_train_dataset(
        dataset_root_dir=mvtec_data_dir,
        category=category_name,
        image_size=image_size,
        batch_size=batch_size
    )

    # Initialize UNet model
    print("[INFO] Initializing UNet model...")
    model = UNetModel(
        img_size=image_size,
        base_channels=32,
        n_heads=2,
        # n_head_channels=64,
        # channel_mults=(1, 1, 2, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.1,
        attention_resolutions="32,16,8",
        biggan_updown=True,
        in_channels=input_channels,
    ).to(device)

    gaussian_diffusion = GaussianDiffusion(num_timesteps=num_timesteps,beta_schedule=beta_schedule)

    optimizer = get_optimizer(
        optimizer_name=optimizer_name,
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    total_epochs = epochs
    loss_history = []
    eval_history = {'img_auroc': [], 'px_auroc': [], 'epochs': []}

    # Init eval
    # img_auroc, px_auroc = evaluate_model(model, vae_model, gaussian_diffusion, device)
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
            t = torch.randint(0, num_timesteps, (B,), device=device)
            noise = torch.randn_like(vae_reconstructed).to(device)
            # x_t: add noise
            x_t = gaussian_diffusion.q_sample(vae_reconstructed, t, noise)

            # Predict noise
            pred_noise = model(x_t, t)

            x_pred = gaussian_diffusion.predict_start_from_noise(x_t, t, pred_noise)

            # Calculate loss
            total_loss = F.mse_loss(x_pred, images)
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
        if (epoch + 1) % eval_interval == 0 and epoch != 1 and epoch != 0:
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
                        'image_size': image_size,
                        'input_channels': input_channels,
                        'num_timesteps': num_timesteps,
                        'beta_schedule': beta_schedule
                    }
                }
                torch.save(eval_checkpoint, os.path.join(pretrained_save_dir, f'diffusion_model_{epoch}_.pth'))

        # Save checkpoint periodically
        if (epoch + 1) % 50 == 0:
            os.makedirs(pretrained_save_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_history': loss_history,
                'config': {
                    'image_size': image_size,
                    'input_channels': input_channels,
                    'num_timesteps': num_timesteps,
                    'beta_schedule': beta_schedule
                }
            }
            torch.save(checkpoint, os.path.join(pretrained_save_dir, f'diffusion_model_epoch_{epoch+1}.pth'))
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
    os.makedirs(pretrained_save_dir, exist_ok=True)
    final_checkpoint = {
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
        'config': {
            'image_size': image_size,
            'input_channels': input_channels,
            'num_timesteps': num_timesteps,
            'beta_schedule': beta_schedule
        }
    }
    torch.save(final_checkpoint, os.path.join(pretrained_save_dir, 'diffusion_model_final.pth'))
    print(f"[INFO] Final model saved to {pretrained_save_dir}")


if __name__ == "__main__":
    set_seed(seed)
    main()




