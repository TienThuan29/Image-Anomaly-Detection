import os
import time
import torch
import numpy as np
import random
import torch.nn.functional as F
from vae_resnet_model import VAEResNet
from vae_unet_model import VAEUnet
from diffusion_gaussian import GaussianDiffusion
from utils import load_mvtec_train_dataset
from diffusion_model import UNetModel
from tqdm import tqdm
from config import load_config
from utils import get_optimizer

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
vae_name = config.vae.name
backbone = config.vae.backbone

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


def load_vae(
      checkpoint_path: str, 
      in_channels: int, 
      latent_dim: int, 
      out_channels: int, 
      device: torch.device
):
    if vae_name == 'vae_resnet':
        model = VAEResNet(
            in_channels=input_channels,
            out_channels=output_channels,
            latent_dim=z_dim,
            resnet_name=backbone,
            dropout_p=dropout_p
        ).to(device)
    elif vae_name == 'vae_unet':
        model = VAEUnet(
            in_channels=input_channels,
            latent_dim=z_dim,
            out_channels=output_channels
        ).to(device)
    else:
        raise ValueError(f"Unknown vae model: {vae_name}")
    # Handle .pth checkpoint loading with proper security
    try:
        # Add safe globals for numpy objects commonly found in PyTorch checkpoints
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


def main(batch_size=batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load VAE model
    print("[INFO] Loading VAE model...")
    vae_model = load_vae(
        checkpoint_path=phase1_vae_pretrained_path,
        in_channels=input_channels,
        latent_dim=z_dim,
        out_channels=output_channels,
        device=device
    )

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
        base_channels=128,
        conv_resample=True,
        n_heads=1,
        n_head_channels=64,
        channel_mults=(1, 1, 2, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0,
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

    print(f"Starting training for {total_epochs} epochs...")
    # Main epoch progress bar
    epoch_bar = tqdm(range(start_epoch, total_epochs),desc="Training Progress",position=0,leave=True)

    for epoch in epoch_bar:
        model.train()
        epoch_start_time = time.time()
        batch_losses = []
        batch_bar = tqdm(train_dataset,desc=f"Epoch {epoch + 1}/{total_epochs}",position=1,leave=False,)

        for batch_idx, batch in enumerate(batch_bar):
            images = batch['image'].to(device)

            # Get reconstruct image from vae model phase 1
            with torch.no_grad():
                vae_reconstructed = vae_model(images)

            B = images.size(0)
            t = torch.randint(0, num_timesteps, (B,), device=device)
            noise = torch.randn_like(vae_reconstructed).to(device)
            # x_t: add noise
            x_t = gaussian_diffusion.q_sample(vae_reconstructed, t, noise)

            # x_t = x_t.view(-1, 16, 4, 4)
            # print("[INFO] Z_view shape:", z.shape)

            # Predict noise
            pred_noise = model(x_t, t)

            #print(f"pred_noise mean={pred_noise.mean().item()}, std={pred_noise.std().item()}")
            #print(f"true_noise mean={noise.mean().item()}, std={noise.std().item()}")

            # x_predict = gaussian_diffusion.predict_start_from_noise(x_t, t, pred_noise)
            # reconstructed = x_predict.reshape(batch_size, -1)
            # print("[INFO] z_predict shape:", z_predict.shape)

            # Calculate loss
            # total_loss = loss_function(reconstructed, images)
            total_loss = F.mse_loss(pred_noise, noise)
            batch_losses.append(total_loss.item())

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update batch progress bar with current loss
            # current_avg_loss = np.mean(batch_losses) # use 'sum'
            # current_avg_loss = batch_losses
            batch_bar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # Calculate epoch metrics
        epoch_loss = np.mean(batch_losses)
        loss_history.append(epoch_loss)
        epoch_time = time.time() - epoch_start_time

        scheduler.step()
        # Update epoch progress bar with comprehensive info
        epoch_bar.set_postfix({
            'Epoch Loss': f'{epoch_loss:.6f}',
            'Time': f'{epoch_time:.1f}s',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

        # Log epoch results
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\n[INFO] Epoch {epoch + 1}/{total_epochs} completed:")
            print(f"        Loss: {epoch_loss:.6f}")
            print(f"        Time: {epoch_time:.1f}s")
            print(f"        Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

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

    # Training completed
    print(f"\n[INFO] Training completed!")
    print(f"        Final loss: {loss_history[-1]:.6f}")
    print(f"        Best loss: {min(loss_history):.6f}")

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


