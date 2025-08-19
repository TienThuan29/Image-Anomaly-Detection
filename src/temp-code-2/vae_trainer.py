import time
from torchvision.utils import save_image
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import random
import os
from tqdm import tqdm
from vae_model import VAEResNet
from utils import ConfigLoader, load_mvtec_train_dataset, load_mvtec_only_good_test_dataset, LossEarlyStopping

config_loader = ConfigLoader("config.yml")
config = config_loader.load_config()
data_config = config_loader.get_section("data")
vae_config = config_loader.get_section("vae_model")
early_stopping_config = config_loader.get_section("early_stopping")

category_name = data_config.get('category')
train_result_dir = vae_config.get('train_result_base_dir') + category_name
pretrained_save_dir = vae_config.get('pretrained_save_base_dir') + category_name


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


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
    device = torch.device(
        f"cuda:{vae_config.get('cuda')}" if vae_config.get('cuda') >= 0 and torch.cuda.is_available() else "cpu")
    print(f'Training VAE on class: {category_name}')
    print(f"Device: {device}")

    train_dataset = load_mvtec_train_dataset(
        dataset_root_dir=data_config.get('mvtec_data_dir'),
        category=data_config.get('category'),
        image_size=data_config.get('image_size'),
        batch_size=data_config.get('batch_size')
    )

    test_dataset = load_mvtec_only_good_test_dataset(
        dataset_root_dir=data_config.get('mvtec_data_dir'),
        category=data_config.get('category'),
        image_size=data_config.get('image_size'),
        batch_size=data_config.get('batch_size')
    )

    model = VAEResNet(
        in_channels=vae_config.get('input_channels'),
        latent_dim=vae_config.get('z_dim')
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(vae_config.get('lr')),
                                 weight_decay=float(vae_config.get('weight_decay')))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=vae_config.get('epochs'))

    early_stopping = LossEarlyStopping(
        patience=early_stopping_config.get('patience'),
        min_delta=early_stopping_config.get('min_delta'),
        smoothing_window=early_stopping_config.get('smoothing_window'),
        verbose=True
    )

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

    final_epoch = vae_config.get('epochs')
    epoch_bar = tqdm(range(start_epoch, vae_config.get('epochs')), desc="Training Progress", position=0)

    for epoch in epoch_bar:
        model.train()
        t1 = time.time()

        loss_batch = []
        num_batches = 0

        batch_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{vae_config.get('epochs')}", leave=False, position=1)
        for batch_idx, batch in enumerate(batch_bar):
            images = batch['image'].to(device)
            batch_size = images.size(0)

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
        if (epoch + 1) % vae_config.get('sample_freq') == 0:
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
        remaining_time = (vae_config.get('epochs') - epoch - 1) * epoch_time

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
        f.write(f"Early stopping triggered: {'Yes' if final_epoch < vae_config.get('epochs') else 'No'}\n")
        f.write(f"Available checkpoints:\n")
        f.write(f"  - Best model: {category_name}_vae_best.pth\n")
        f.write(f"  - Latest model: {category_name}_vae_latest.pth\n")
        f.write(f"  - Final model: {category_name}_vae_final.pth\n")
        if final_epoch < vae_config.get('epochs'):
            f.write(f"  - Early stop model: {category_name}_vae_early_stop_epoch_{final_epoch - 1}.pth\n")

    return loss_epoch


if __name__ == '__main__':
    set_seed(vae_config.get('seed'))
    resume_checkpoint = None
    loss_epoch = main(resume_from_checkpoint=resume_checkpoint)
    print(f"Training completed!")