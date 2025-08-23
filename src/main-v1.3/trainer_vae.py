import time
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from tqdm import tqdm
from loss_functions import vae_loss_function
from vae_unet_model import VAEUnet
from vae_resnet_model import VAEResNet
from config import load_config
from utils import  load_mvtec_train_dataset, LossEarlyStopping, get_optimizer

config = load_config()
# general
_seed = config.general.seed
_cuda = config.general.cuda
_image_size = config.general.image_size
_batch_size = config.general.batch_size
input_channels = config.general.input_channels
output_channels = config.general.output_channels

# data
category_name = config.data.category
mvtec_data_dir = config.data.mvtec_data_dir
mask = config.data.mask

# vae
vae_name = config.vae_model.name
epochs = config.vae_model.epochs
z_dim = config.vae_model.z_dim
lr = float(config.vae_model.lr)
dropout_p = config.vae_model.dropout_p
weight_decay = float(config.vae_model.weight_decay)
save_freq = config.vae_model.save_freq
sample_freq = config.vae_model.sample_freq
backbone = config.vae_model.backbone # backbone for resnet
optimizer_name = config.vae_model.optimizer_name
resume_checkpoint_path = config.vae_model.resume_checkpoint

# save train result dir
sub_path = f"{backbone}/" if vae_name == "vae_resnet" else ""
train_result_dir = f"{config.vae_model.train_result_base_dir}{vae_name}/{sub_path}{category_name}/"
pretrained_save_dir = f"{config.vae_model.pretrained_save_base_dir}{vae_name}/{sub_path}{category_name}/"


# early stopping
patience = config.early_stopping.patience
min_delta = config.early_stopping.min_delta
smoothing_window = config.early_stopping.smoothing_window


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, loss_epoch, best_loss,
                    category_name, save_dir, checkpoint_type="best"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': loss_epoch,
        'best_loss': best_loss,
        'class_name': category_name,
        'checkpoint_type': checkpoint_type,
        'timestamp': time.time()
    }

    if checkpoint_type == "best":
        filename = f'{category_name}_vae_best.pth'
    elif checkpoint_type == "latest":
        filename = f'{category_name}_vae_latest.pth'
    elif checkpoint_type == "early_stop":
        filename = f'{category_name}_vae_early_stop_epoch_{epoch}.pth'
    else:
        filename = f'{category_name}_vae_{checkpoint_type}.pth'

    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
    return filepath


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint.get('train_losses', [])
    best_loss = checkpoint.get('best_loss', float('inf'))

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Checkpoint loaded: epoch {checkpoint['epoch']}, best_loss: {best_loss:.6f}")
    return start_epoch, train_losses, best_loss


def main(resume_from_checkpoint=None):
    device = torch.device(f"cuda:{_cuda}" if _cuda >= 0 and torch.cuda.is_available() else "cpu")
    print(f'Training VAE on class: {category_name}')
    print(f"Device: {device}")
    print(f"Batch size: {_batch_size}")
    print(f"VAE name: {vae_name}")
    if vae_name == "vae_resnet":
        print(f"Backbone: {backbone}")

    train_dataset = load_mvtec_train_dataset(
        dataset_root_dir=mvtec_data_dir,
        category=category_name,
        image_size=_image_size,
        batch_size=_batch_size
    )

    if vae_name == 'vae_resnet':
        model = VAEResNet(
            image_size=_image_size,
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
            out_channels=output_channels,
            dropout_p=dropout_p
        ).to(device)
    else:
        raise ValueError(f"Unknown vae model: {vae_name}")

    optimizer = get_optimizer(
        optimizer_name=optimizer_name,
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    early_stopping = LossEarlyStopping(
        patience=patience,
        min_delta=min_delta,
        smoothing_window=smoothing_window,
        verbose=True
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    if not os.path.exists(train_result_dir):
        os.makedirs(train_result_dir)
    if not os.path.exists(pretrained_save_dir):
        os.makedirs(pretrained_save_dir)

    # Khởi tạo các biến training
    start_epoch = 0
    loss_epoch = []
    best_loss = float('inf')

    # Resume train
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch, loss_epoch, best_loss = load_checkpoint(
            resume_from_checkpoint, model, optimizer, scheduler, device
        )
        model.to(device)
        early_stopping.best_loss = best_loss

    # Log file
    log_file_path = f'{train_result_dir}/{category_name}_training_evaluation_log.txt'
    log_mode = 'a' if resume_from_checkpoint else 'w'
    with open(log_file_path, log_mode) as log_file:
        if not resume_from_checkpoint:
            log_file.write(f"Training Evaluation Log for {category_name}\n")
            log_file.write("=" * 50 + "\n\n")
        else:
            log_file.write(f"\nResuming training from epoch {start_epoch}\n")
            log_file.write("=" * 50 + "\n\n")

    final_epoch = epochs
    epoch_bar = tqdm(range(start_epoch, epochs), desc="Training Progress", position=0)

    for epoch in epoch_bar:
        model.train()
        t1 = time.time()

        loss_batch = []
        num_batches = 0

        batch_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, position=1)
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
            save_checkpoint(model, optimizer, scheduler, epoch, loss_epoch, best_loss,
                            category_name, pretrained_save_dir, "best")

        # Lưu latest checkpoint mỗi vài epochs
        if (epoch + 1) % 30 == 0:  # Lưu mỗi 5 epochs
            save_checkpoint(model, optimizer, scheduler, epoch, loss_epoch, best_loss,
                            category_name, pretrained_save_dir, "latest")

        # Check early stopping
        if early_stopping(epoch_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            final_epoch = epoch + 1

            # Lưu early stopping checkpoint
            early_stop_path = save_checkpoint(model, optimizer, scheduler, epoch, loss_epoch, best_loss,
                                              category_name, pretrained_save_dir, "early_stop")

            # Log early stopping event
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"\nEarly stopping triggered at epoch {epoch + 1}\n")
                log_file.write(f"Best loss achieved: {early_stopping.best_loss:.6f}\n")
                log_file.write(f"Early stop checkpoint saved: {early_stop_path}\n")
                log_file.write("=" * 50 + "\n")
            break

        scheduler.step()

        # Save sample reconstructions
        if (epoch + 1) % sample_freq == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_dataset))
                sample_images = sample_batch['image'][:8].to(device)
                sample_recon, _, _ = model(sample_images)
                comparison = torch.cat([sample_images, sample_recon], dim=0)
                save_image(comparison,
                           f'{train_result_dir}/{category_name}_epoch_{epoch + 1:03d}_.png',
                           nrow=8, normalize=True)

        # Calculate timing
        t2 = time.time()
        epoch_time = t2 - t1
        remaining_time = (epochs - epoch - 1) * epoch_time

        # Update main progress bar
        postfix_dict = {'Train': f'{epoch_loss:.4f}',
                        'Best': f'{best_loss:.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                        'ETA': f'{int(remaining_time // 3600)}h{int((remaining_time % 3600) // 60)}m'}
        epoch_bar.set_postfix(postfix_dict)

    # Close progress bars
    epoch_bar.close()
    print(f"\nTraining completed!")
    print(f"Total epochs: {final_epoch}")
    print(f"Final loss: {loss_epoch[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f}")

    # save final checkpoint
    final_checkpoint_path = save_checkpoint(model, optimizer, scheduler, final_epoch - 1, loss_epoch, best_loss,
                                            category_name, pretrained_save_dir, "final")

    summary_path = f'{pretrained_save_dir}/{category_name}_training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary for {category_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total epochs trained: {final_epoch}\n")
        f.write(f"Best loss achieved: {best_loss:.6f}\n")
        f.write(f"Final loss: {loss_epoch[-1]:.6f}\n")
        f.write(f"Early stopping triggered: {'Yes' if final_epoch < epochs else 'No'}\n")
        f.write(f"Available checkpoints:\n")
        f.write(f"  - Best model: {category_name}_vae_best.pth\n")
        f.write(f"  - Latest model: {category_name}_vae_latest.pth\n")
        f.write(f"  - Final model: {category_name}_vae_final.pth\n")
        if final_epoch < epochs:
            f.write(f"  - Early stop model: {category_name}_vae_early_stop_epoch_{final_epoch - 1}.pth\n")

    return loss_epoch


if __name__ == '__main__':
    set_seed(_seed)
    resume_checkpoint = resume_checkpoint_path
    loss_epoch = main(resume_from_checkpoint=resume_checkpoint)
    print(f"Training completed!")