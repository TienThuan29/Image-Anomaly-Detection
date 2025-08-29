import time
import torch
import os

def save_checkpoint(
        model, optimizer, scheduler, epoch, loss_epoch, best_loss,
        category_name, save_dir, checkpoint_type="best"
):
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
