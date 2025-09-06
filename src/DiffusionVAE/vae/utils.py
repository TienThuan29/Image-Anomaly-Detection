import time
import os
import torch
import inspect
from vae.vae_resnet import VAEResNet

def _add_numpy_allowlist_for_safe_unpickler():
    try:
        import numpy as np
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np.float32, np.float64, np.int32, np.int64,
            np.bool_, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64
        ])
    except Exception:
        pass

def _torch_load_compat(path: str, map_location, allow_fallback_to_untrusted: bool = True):
    sig = inspect.signature(torch.load)
    supports_weights_only = 'weights_only' in sig.parameters

    try:
        if supports_weights_only:
            return torch.load(path, map_location=map_location, weights_only=True)
        else:
            return torch.load(path, map_location=map_location)
    except Exception as e1:
        _add_numpy_allowlist_for_safe_unpickler()
        try:
            if supports_weights_only:
                return torch.load(path, map_location=map_location, weights_only=True)
            else:
                return torch.load(path, map_location=map_location)
        except Exception as e2:
            if supports_weights_only and allow_fallback_to_untrusted:
                print("[WARN] Secure loading failed twice. Falling back to weights_only=False ""(ONLY do this for trusted checkpoints).")
                try:
                    return torch.load(path, map_location=map_location, weights_only=False)
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to load checkpoint safely and fallback also failed.\n"
                        f"Safe error #1: {e1}\nSafe error #2 (after allowlist): {e2}\n"
                        f"Fallback error: {e3}"
                    )
            raise RuntimeError(
                f"Failed to load checkpoint with safe mode and fallback disabled.\n"
                f"Safe error #1: {e1}\nSafe error #2 (after allowlist): {e2}"
            )

def _extract_state_dict(ckpt: dict):
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            sd = ckpt['model_state_dict']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            sd = ckpt['state_dict']
        else:
            sd = ckpt
    else:
        sd = ckpt

    if isinstance(sd, dict):
        sd = {
            (k.replace('module.', '', 1) if isinstance(k, str) and k.startswith('module.') else k): v
            for k, v in sd.items()
        }
    return sd

""" Load pre-trained vae model """
@torch.no_grad()
def load_vae_model(
        checkpoint_path: str,
        vae_name: str,
        input_channels: int,
        output_channels: int,
        z_dim: int,
        backbone: str,
        dropout_p: float,
        image_size: int,
        device: torch.device
):
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
    else:
        raise ValueError(f"Unknown vae model: {vae_name}")

    ckpt = _torch_load_compat(checkpoint_path, map_location=device, allow_fallback_to_untrusted=True)
    state = _extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] VAE missing keys: {missing}, Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


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
