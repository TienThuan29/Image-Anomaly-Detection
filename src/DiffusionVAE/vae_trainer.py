import os
import torch
import time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
from torch import Tensor
from config import load_config
from vae.utils import save_checkpoint, load_checkpoint
from data.dataloader import load_mvtec_train_dataset
from vae.vae_resnet import VAEResNet
from utils.optimizers import get_optimizer
from vae.early_stopping import LossEarlyStopping

config = load_config()

# general
_image_size = config.general.image_size
_batch_size = config.general.batch_size

# data
_mvtec_data_dir = config.data.mvtec_data_dir
_mask = config.data.mask

# Training category
_category_name = config.training.category

# vae
_vae_name = config.vae_model.name
_epochs = config.vae_model.epochs
_z_dim = config.vae_model.z_dim
_lr = float(config.vae_model.lr)
_dropout_p = config.vae_model.dropout_p
_weight_decay = float(config.vae_model.weight_decay)
_input_channels = config.vae_model.in_channels
_output_channels = config.vae_model.out_channels
_save_freq = config.vae_model.save_freq
_sample_freq = config.vae_model.sample_freq
_backbone = config.vae_model.backbone
_optimizer_name = config.vae_model.optimizer_name
_resume_checkpoint_path = config.vae_model.resume_checkpoint

# train result dir
_sub_path = f"{_backbone}/"
_train_result_dir = f"{config.vae_model.train_result_base_dir}{_vae_name}/{_sub_path}{_category_name}/"
_pretrained_save_dir = f"{config.vae_model.pretrained_save_base_dir}{_vae_name}/{_sub_path}{_category_name}/"

# early stopping
_patience = config.vae_model.early_stopping.patience
_min_delta = config.vae_model.early_stopping.min_delta
_smoothing_window = config.vae_model.early_stopping.smoothing_window

if not os.path.exists(_train_result_dir):
    os.makedirs(_train_result_dir)
if not os.path.exists(_pretrained_save_dir):
    os.makedirs(_pretrained_save_dir)


def vae_loss_function(
        x_hat: Tensor,
        x: Tensor,
        mu: Tensor,
        log_var: Tensor
):
    # Reconstruction loss (MSE)
    MSE = F.mse_loss(x_hat, x, reduction='sum')
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    loss = MSE + KLD
    return loss, MSE, KLD


def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training VAE on class: {_category_name}')
    print(f"Device: {device}")
    print(f"Batch size: {_batch_size}")
    print(f"VAE name: {_vae_name}")
    if _vae_name == "vae_resnet":
        print(f"Backbone: {_backbone}")
    
    # Variable to store the best model path
    best_model_path = None

    train_dataset = load_mvtec_train_dataset(
        dataset_root_dir=_mvtec_data_dir, category=_category_name,
        image_size=_image_size, batch_size=_batch_size
    )

    model = VAEResNet(
        image_size=_image_size,
        in_channels=_input_channels,
        out_channels=_output_channels,
        latent_dim=_z_dim,
        resnet_name=_backbone,
        dropout_p=_dropout_p
    ).to(device)

    optimizer = get_optimizer(optimizer_name=_optimizer_name,params=model.parameters(),lr=_lr, weight_decay=_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_epochs)
    early_stopping = LossEarlyStopping(patience=_patience,min_delta=_min_delta,smoothing_window=_smoothing_window,verbose=True)

    start_epoch = 0
    loss_epoch = []
    best_loss = float('inf')

    # Resume train
    if _resume_checkpoint_path and os.path.exists(_resume_checkpoint_path):
        start_epoch, loss_epoch, best_loss = load_checkpoint(
            _resume_checkpoint_path, model, optimizer, scheduler, device
        )
        model.to(device)
        early_stopping.best_loss = best_loss

    final_epoch = _epochs
    epoch_bar = tqdm(range(start_epoch, _epochs), desc="Training Progress", position=0)

    for epoch in epoch_bar:
        model.train()
        t1 = time.time()

        loss_batch = []
        num_batches = 0

        batch_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{_epochs}", leave=False, position=1)
        for batch_idx, batch in enumerate(batch_bar):
            images = batch['image'].to(device)

            # Forward
            reconstructed, mu, log_var = model(images)
            total_loss, recon_loss, kld_loss = vae_loss_function(reconstructed, images, mu, log_var)
            loss_batch.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Calculate RMSE
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(reconstructed, images)).item()

            num_batches += 1

            batch_bar.set_postfix({
                "Total": f"{total_loss:.4f}",
                "Recon": f"{recon_loss:.4f}",
                "KLD": f"{kld_loss:.4f}",
                'RMSE': f'{rmse:.4f}'
            })

        epoch_loss = np.sum(loss_batch) / num_batches
        loss_epoch.append(epoch_loss)

        # save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = save_checkpoint(model, optimizer, scheduler, epoch, loss_epoch, best_loss,
                            _category_name, _pretrained_save_dir, "best")

        if (epoch + 1) % 30 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, loss_epoch, best_loss,
                            _category_name, _pretrained_save_dir, "latest")

        # Check early stopping
        if early_stopping(epoch_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            final_epoch = epoch + 1
            # Save early stopping checkpoint
            early_stop_path = save_checkpoint(
                model, optimizer, scheduler, epoch, loss_epoch, best_loss,
                _category_name, _pretrained_save_dir, "early_stop"
            )
            epoch_bar.close()
            break

        scheduler.step()

        # Save sample reconstructions
        if (epoch + 1) % _sample_freq == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_dataset))
                sample_images = sample_batch['image'][:8].to(device)
                sample_recon, _, _ = model(sample_images)
                comparison = torch.cat([sample_images, sample_recon], dim=0)
                save_image(comparison,
                           f'{_train_result_dir}/{_category_name}_epoch_{epoch + 1:03d}_.png',
                           nrow=8, normalize=True)

        # Calculate timing
        t2 = time.time()
        epoch_time = t2 - t1
        remaining_time = (_epochs - epoch - 1) * epoch_time

        # Update main progress bar
        postfix_dict = {'Train': f'{epoch_loss:.4f}',
                        'Best': f'{best_loss:.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                        'ETA': f'{int(remaining_time // 3600)}h{int((remaining_time % 3600) // 60)}m'}
        epoch_bar.set_postfix(postfix_dict)

    epoch_bar.close()
    print(f"\nTraining completed!")
    print(f"Total epochs: {final_epoch}")
    print(f"Final loss: {loss_epoch[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f}")

    # save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, final_epoch - 1, loss_epoch, best_loss,
        _category_name, _pretrained_save_dir, "final"
    )

    summary_path = f'{_pretrained_save_dir}/{_category_name}_training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary for {_category_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total epochs trained: {final_epoch}\n")
        f.write(f"Best loss achieved: {best_loss:.6f}\n")
        f.write(f"Final loss: {loss_epoch[-1]:.6f}\n")
        f.write(f"Early stopping triggered: {'Yes' if final_epoch < _epochs else 'No'}\n")
        f.write(f"Available checkpoints:\n")
        f.write(f"  - Best model: {_category_name}_vae_best.pth\n")
        f.write(f"  - Latest model: {_category_name}_vae_latest.pth\n")
        f.write(f"  - Final model: {_category_name}_vae_final.pth\n")
        if final_epoch < _epochs:
            f.write(f"  - Early stop model: {_category_name}_vae_early_stop_epoch_{final_epoch - 1}.pth\n")
    
    return best_model_path

