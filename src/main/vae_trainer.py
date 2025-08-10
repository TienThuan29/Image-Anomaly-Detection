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
    loss =  MSE + KLD
    return loss, MSE, KLD


def main():
    device = torch.device(f"cuda:{vae_config.get('cuda')}" if vae_config.get('cuda') >= 0 and torch.cuda.is_available() else "cpu")
    print(f'Training VAE on class: {category_name}')
    print(f"Device: {device}")

    # print(vae_config.get('mvtec_data_dir'))
    # print(data_config.get('category'))

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

    optimizer = torch.optim.Adam(model.parameters(), lr=float(vae_config.get('lr')), weight_decay=float(vae_config.get('weight_decay')))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=vae_config.get('epochs'))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=20, min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.95)

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

    #
    log_file_path = f'{train_result_dir}/{category_name}_training_evaluation_log.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Training Evaluation Log for {category_name}\n")
        log_file.write("=" * 50 + "\n\n")

    loss_epoch = []
    final_epoch = vae_config.get('epochs')
    best_loss = float('inf')

    # best_image_auroc = 0.0
    # best_pixel_auroc = 0.0
    best_image_auroc_epoch = 0
    best_pixel_auroc_epoch = 0

    test_mse_scores = []
    best_test_mse = float('inf')
    best_test_mse_epoch = 0


    epoch_bar = tqdm(range(vae_config.get('epochs')), desc="Training Progress", position=0)

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
                "Recon" : f"{recon_loss:.4f}",
                "KLD" : f"{kld_loss:.4f}",
                'RMSE': f'{rmse:.4f}'
            })

        # loss_epoch.append(np.sum(loss_batch) / num_batches)
        # Calculate epoch loss
        epoch_loss = np.sum(loss_batch) / num_batches
        loss_epoch.append(epoch_loss)

        # Check early stopping
        if early_stopping(epoch_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            final_epoch = epoch + 1
            # Log early stopping event
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"\nEarly stopping triggered at epoch {epoch + 1}\n")
                log_file.write(f"Best loss achieved: {early_stopping.best_loss:.6f}\n")
                log_file.write("=" * 50 + "\n")
            break

        scheduler.step()

        # Eval each 10 epoch
        if (epoch + 1) % vae_config.get('testing_freq') == 0:
            print(f"\nRunning evaluation at epoch {epoch + 1}...")
            model.eval()
            test_mse_losses = []
            with torch.no_grad():
                for test_batch in test_dataset:
                    test_images = test_batch['image'].to(device)
                    test_reconstructed, _, _ = model(test_images)
                    # Cal MSE for good test dataset
                    batch_mse = F.mse_loss(test_reconstructed, test_images, reduction='none')
                    # Average over spatial dimensions (C, H, W) to get MSE per image
                    batch_mse = batch_mse.view(batch_mse.size(0), -1).mean(dim=1)
                    test_mse_losses.extend(batch_mse.cpu().numpy())
            # Calculate average MSE across all test images
            avg_test_mse = np.mean(test_mse_losses)
            test_mse_scores.append(avg_test_mse)

            # Update best test MSE
            if avg_test_mse < best_test_mse:
                best_test_mse = avg_test_mse
                best_test_mse_epoch = epoch + 1
                # Save best model based on test MSE
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': loss_epoch,
                    'test_mse_scores': test_mse_scores,
                    'best_test_mse': best_test_mse,
                    'test_mse': avg_test_mse,
                    'class_name': category_name
                }, f'{pretrained_save_dir}/{category_name}_vae_best_test_mse.pth')

            # Log evaluation results
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Epoch {epoch + 1}:\n")
                log_file.write(f"  Train Loss: {loss_epoch[-1]:.6f}\n")
                log_file.write(f"  Test MSE: {avg_test_mse:.6f}\n")
                log_file.write(f"  Best Test MSE: {best_test_mse:.6f} (Epoch {best_test_mse_epoch})\n")
                log_file.write(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n")
                log_file.write("-" * 30 + "\n")

            print(f"Test MSE: {avg_test_mse:.6f} | Best Test MSE: {best_test_mse:.6f} (Epoch {best_test_mse_epoch})")

        # Save sample reconstructions
        if (epoch + 1) % vae_config.get('sample_freq') == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_dataset))
                sample_images = sample_batch['image'][:8].to(device)  # Take first 5 images
                sample_recon, _, _ = model(sample_images)
                # Concatenate original and reconstructed
                comparison = torch.cat([sample_images, sample_recon], dim=0)
                save_image(comparison,
                           f'{train_result_dir}/{category_name}_epoch_{epoch + 1:03d}_.png',
                           nrow=8, normalize=True)

        # Calculate timing
        t2 = time.time()
        epoch_time = t2 - t1
        remaining_time = (vae_config.get('epochs') - epoch - 1) * epoch_time

        # Update main progress bar
        postfix_dict = {'Total': f'{total_loss:.4f}', 'Recon': f'{recon_loss:.4f}', 'KLD': f'{kld_loss:.4f}',
                        'RMSE': f'{rmse:.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                        'ETA': f'{int(remaining_time // 3600)}h{int((remaining_time % 3600) // 60)}m'}
        epoch_bar.set_postfix(postfix_dict)

    # Close progress bars
    epoch_bar.close()
    print(f"\nTraining completed!")
    print(f"Completed all {vae_config.get('epochs')} epochs")
    print(f"Final loss: {loss_epoch[-1]:.4f}")

    # print(f"\nSUMMARY:")
    # print(f"Best image_auroc: {best_image_auroc:.4f} at epoch {best_image_auroc_epoch}")
    # print(f"Best pixel_auroc: {best_pixel_auroc:.4f} at epoch {best_pixel_auroc_epoch}")

    # Save final model
    torch.save({
        'epoch': final_epoch - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_epoch,
        'best_loss': best_loss,
        'final_epoch': final_epoch,
        'class_name': category_name
    }, f'{pretrained_save_dir}/{category_name}_vae_final.pth')

    return loss_epoch


if __name__ == '__main__':
    set_seed(vae_config.get('seed'))
    loss_epoch = main()
    print(f"Completed!")

