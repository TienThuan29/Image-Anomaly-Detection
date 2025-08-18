import os
import time
import torch
import numpy as np
import random
from torch import optim
from torch.functional import F
from utils import ConfigLoader
from vae_model import VAEResNet
from gaussian_diffusion import GaussianDiffusion
from utils import load_mvtec_train_dataset
from diffusion_model import UNetModel
from tqdm import tqdm

config_loader = ConfigLoader("config.yml")
config = config_loader.load_config()
data_config = config_loader.get_section("data")
# vae_config = config_loader.get_section("vae_model")
diffusion_config = config_loader.get_section("diffusion_model")
early_stopping_config = config_loader.get_section("early_stopping")

category_name = data_config.get('category')
# train_result_dir = vae_config.get('train_result_base_dir') + category_name
# pretrained_save_dir = vae_config.get('pretrained_save_base_dir') + category_name


def load_vae(checkpoint_path: str, in_channels: int, latent_dim: int, out_channels: int, device: torch.device):
    model = VAEResNet(in_channels=in_channels, latent_dim=latent_dim, out_channels=out_channels).to(device)

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

        # Add all safe globals at once
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load VAE model
    print("[INFO] Loading VAE model...")
    vae_model = load_vae(
        checkpoint_path=diffusion_config.get("phase1_model_path"),
        in_channels=diffusion_config.get("input_channels"),
        latent_dim=diffusion_config.get("z_dim"),
        out_channels=diffusion_config.get("output_channels"),
        device=device
    )

    # Load dataset
    print("[INFO] Loading dataset...")
    train_dataset = load_mvtec_train_dataset(
        dataset_root_dir=data_config.get('mvtec_data_dir'),
        category=data_config.get('category'),
        image_size=data_config.get('image_size'),
        batch_size=data_config.get('batch_size')
    )

    # Initialize UNet model
    print("[INFO] Initializing UNet model...")
    unet_model = UNetModel(
        img_size=4,  # spatial size = 4x4
        in_channels=16, # because z_2d = (B, 16, 4, 4)
        base_channels=128,
        channel_mults=(1, 2),
        num_res_blocks=2,
        attention_resolutions="64,32",
        n_head_channels=32,
        dropout=0.1,
        conv_resample=True
    ).to(device)

    # Initialize diffusion
    num_timesteps = 1000
    gaussian_diffusion = GaussianDiffusion(
        num_timesteps=num_timesteps,
        beta_schedule=diffusion_config.get('beta_schedule'),
    )

    # Initialize optimizer and scheduler
    # optimizer = torch.optim.Adam(
    #     unet_model.parameters(),
    #     lr=float(diffusion_config.get('lr')),
    #     weight_decay=float(diffusion_config.get('weight_decay'))
    # )
    optimizer = torch.optim.AdamW(
        unet_model.parameters(),
        lr=float(diffusion_config.get('lr')),
        weight_decay=float(diffusion_config.get('weight_decay'))
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=diffusion_config.get('epochs')
    # )

    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=12,
        min_lr=1e-6,
        verbose=True
    )

    #warmup_steps = 1000
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup_steps, 1.0))

    # Training setup
    start_epoch = 0
    total_epochs = diffusion_config.get('epochs')  # Fixed: use diffusion_config instead of vae_config
    loss_history = []

    print(f"[INFO] Starting training for {total_epochs} epochs...")

    # Main epoch progress bar
    epoch_bar = tqdm(
        range(start_epoch, total_epochs),
        desc="Training Progress",
        position=0,
        leave=True,
        colour='blue'
    )

    for epoch in epoch_bar:
        unet_model.train()
        epoch_start_time = time.time()

        # Track losses for this epoch
        batch_losses = []

        # Batch progress bar
        batch_bar = tqdm(
            train_dataset,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            position=1,
            leave=False,
            colour='green'
        )

        for batch_idx, batch in enumerate(batch_bar):
            images = batch['image'].to(device)

            # Get Z latent from VAE encoder
            with torch.no_grad():
                z = vae_model.get_z(images)  # [B, 256]
                # print("[INFO] Z shape:", z.shape)

            # Forward diffusion
            batch_size = z.shape[0]
            t = torch.randint(0, num_timesteps, (batch_size,), device=device)
            noise = torch.randn_like(z).to(device)
            # z_t: add noise
            z_t = gaussian_diffusion.q_sample(z, t, noise)

            z_t = z_t.view(-1, 16, 4, 4)
            # print("[INFO] Z_view shape:", z.shape)

            # Predict noise
            pred_noise = unet_model(z_t, t)

            #print(f"pred_noise mean={pred_noise.mean().item()}, std={pred_noise.std().item()}")
            #print(f"true_noise mean={noise.mean().item()}, std={noise.std().item()}")

            z_predict = gaussian_diffusion.predict_start_from_noise(z_t, t, pred_noise)
            z_predict = z_predict.reshape(batch_size, -1)
            # print("[INFO] z_predict shape:", z_predict.shape)

            # Calculate loss
            total_loss = loss_function(z_predict, z)
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

        # Update learning rate
        # scheduler.step()
        plateau.step(epoch_loss)
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

        # Optional: Save checkpoint periodically
        # if (epoch + 1) % 50 == 0:
        #     checkpoint_path = os.path.join(pretrained_save_dir, f'diffusion_checkpoint_epoch_{epoch + 1}.pth')
        #     os.makedirs(pretrained_save_dir, exist_ok=True)
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': unet_model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'loss_history': loss_history,
        #         'loss': epoch_loss
        #     }, checkpoint_path)
        #     tqdm.write(f"[INFO] Checkpoint saved: {checkpoint_path}")

    # Training completed
    print(f"\n[INFO] Training completed!")
    print(f"        Final loss: {loss_history[-1]:.6f}")
    print(f"        Best loss: {min(loss_history):.6f}")

    # Save final model
    # final_model_path = os.path.join(pretrained_save_dir, 'diffusion_final_model.pth')
    # os.makedirs(pretrained_save_dir, exist_ok=True)
    # torch.save({
    #     'model_state_dict': unet_model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'scheduler_state_dict': scheduler.state_dict(),
    #     'loss_history': loss_history,
    #     'final_loss': loss_history[-1],
    #     'config': {
    #         'diffusion_config': diffusion_config,
    #         'data_config': data_config
    #     }
    # }, final_model_path)
    # print(f"[INFO] Final model saved: {final_model_path}")


if __name__ == "__main__":
    set_seed(diffusion_config.get('seed'))
    main()



# import os
# import time
# import torch
# import numpy as np
# import random
# from torch import optim
# from torch.functional import F
# from utils import ConfigLoader
# from vae_model import VAEResNet
# from gaussian_diffusion import GaussianDiffusion
# from utils import load_mvtec_train_dataset
# from diffusion_model import UNetModel
# import tqdm
#
# config_loader = ConfigLoader("config.yml")
# config = config_loader.load_config()
# data_config = config_loader.get_section("data")
# vae_config = config_loader.get_section("vae_model")
# diffusion_config = config_loader.get_section("diffusion_model")
# early_stopping_config = config_loader.get_section("early_stopping")
#
# category_name = data_config.get('category')
# train_result_dir = vae_config.get('train_result_base_dir') + category_name
# pretrained_save_dir = vae_config.get('pretrained_save_base_dir') + category_name
#
#
# def load_vae(checkpoint_path: str, in_channels: int, latent_dim: int, out_channels: int, device: torch.device):
#     model = VAEResNet(in_channels=in_channels, latent_dim=latent_dim, out_channels=out_channels).to(device)
#
#     # Handle .pth checkpoint loading with proper security
#     try:
#         # Add safe globals for numpy objects commonly found in PyTorch checkpoints
#         import numpy as np
#         safe_globals = [
#             np.core.multiarray.scalar,
#             np.ndarray,
#             np.dtype,
#             np.float32, np.float64, np.int32, np.int64,
#             np.bool_, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64
#         ]
#
#         # Add all safe globals at once
#         torch.serialization.add_safe_globals(safe_globals)
#
#         ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
#         print(f"[INFO] Successfully loaded {checkpoint_path} with weights_only=True")
#
#     except Exception as e:
#         # If weights_only=True still fails, use the fallback but warn about security
#         print(f"[WARN] Secure loading failed, using fallback (potential security risk)")
#         print(f"[WARN] Error details: {str(e)[:200]}...")  # Truncate long error messages
#         ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
#
#     # Handle both direct state_dict and nested checkpoint formats
#     if 'model_state_dict' in ckpt:
#         state = ckpt['model_state_dict']
#     elif 'state_dict' in ckpt:
#         state = ckpt['state_dict']
#     else:
#         state = ckpt  # Assume it's a direct state_dict
#
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"[WARN] Missing keys: {missing}, Unexpected keys: {unexpected}")
#
#     model.eval()
#     return model
#
#
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#
# def loss_function(z_predict, z_origin):
#     return F.mse_loss(z_predict, z_origin, reduction='sum')
#
#
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # load vae model
#     vae_model = load_vae(
#         checkpoint_path=diffusion_config.get("phase1_model_path"),
#         in_channels=diffusion_config.get("input_channels"),
#         latent_dim=diffusion_config.get("z_dim"),
#         out_channels=diffusion_config.get("output_channels"),
#         device=device
#     )
#
#     train_dataset = load_mvtec_train_dataset(
#         dataset_root_dir=data_config.get('mvtec_data_dir'),
#         category=data_config.get('category'),
#         image_size=data_config.get('image_size'),
#         batch_size=data_config.get('batch_size')
#     )
#
#     unet_model = UNetModel(
#         img_size=8,  # latent Z: 8x8 spatial
#         base_channels=256,
#         channel_mults=(1, 2, 2),
#         num_res_blocks=2,
#         attention_resolutions="8",  # attention resolution 8x8
#         in_channels=512,  # input là latent z_2d với shape (B, 512, 8, 8)
#         conv_resample=True
#     )
#
#     num_timesteps = 1000
#     gaussian_diffusion = GaussianDiffusion(
#         num_timesteps = num_timesteps, beta_schedule='linear'
#     )
#
#     optimizer = torch.optim.Adam(unet_model.parameters(), lr=float(diffusion_config.get('lr')),
#                                  weight_decay=float(diffusion_config.get('weight_decay')))
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=diffusion_config.get('epochs'))
#
#     # Khởi tạo các biến training
#     start_epoch = 0
#     loss_epoch = []
#
#     epoch_bar = tqdm(range(start_epoch, vae_config.get('epochs')), desc="Training Progress", position=0)
#     for epoch in epoch_bar:
#         unet_model.train()
#         t1 = time.time()
#
#         loss_batch = []
#         loss_epoch = []
#         num_batches = 0
#
#         batch_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{vae_config.get('epochs')}", leave=False, position=1)
#         for batch_idx, batch in enumerate(batch_bar):
#             images = batch['image'].to(device)
#
#             # Get Z latent from vae encoder
#             with torch.no_grad():
#                 z = vae_model.get_z(images)  # (B, 512*8*8)
#             z = z.view(-1, 512, 8, 8)
#
#             # Forward diffusion
#             batch_size = z.shape[0]
#             t = torch.randint(0, num_timesteps, (batch_size,), device=device)
#             noise = torch.randn_like(z)
#             # z_t: add noise
#             z_t = gaussian_diffusion.q_sample(z, t, noise)
#
#             pred_noise = unet_model(z_t, t)
#             z_predict = gaussian_diffusion.predict_start_from_noise(z_t, t, pred_noise)
#             total_loss = loss_function(z_predict, z)
#
#             loss_batch.append(total_loss.item())
#
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#
#         epoch_loss = np.sum(loss_batch) / num_batches
#         loss_epoch.append(epoch_loss)
#
#
