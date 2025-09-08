import diffusion.model
from vae.utils import load_vae_model
import torch
import os
import time
import json
import numpy as np
from config import load_config
from tqdm import tqdm
from data.dataloader import load_mvtec_train_dataset
from testing.inference import run_inference_during_training

config = load_config()

# general
_cuda = config.general.cuda
_image_size = config.general.image_size
_batch_size = config.general.batch_size

# data
_mvtec_data_dir = config.data.mvtec_data_dir
_mask = config.data.mask

# Training category
_category_name = config.training.category

# vae
_vae_name = config.vae_model.name
_backbone = config.vae_model.backbone
_input_channels = config.vae_model.in_channels
_output_channels = config.vae_model.out_channels
_z_dim = config.vae_model.z_dim
_dropout_p = config.vae_model.dropout_p
# vae pre-trained path
_vae_pretrained_path = config.diffusion_model.vae_pretrained_path

# diffusion train
_epochs = config.diffusion_model.epochs
_resume_checkpoint = config.diffusion_model.resume_checkpoint

# eval
_begin_eval_at_epoch = config.diffusion_model.begin_eval_at_epoch
_eval_interval = config.diffusion_model.eval_interval
_save_model_each = config.diffusion_model.save_model_each

# save train results
_train_result_dir = config.diffusion_model.train_result_base_dir + _category_name + "/"
_log_result_dir = config.diffusion_model.train_result_base_dir + _category_name + "/" + "log_results/"
_pretrained_save_dir = config.diffusion_model.pretrained_save_base_dir + _category_name + "/"


def load_checkpoint(checkpoint_path, diffusion_model, log_dir):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        diffusion_model.netG.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully")

    if 'optimizer_state_dict' in checkpoint:
        diffusion_model.optG.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded successfully")

    if 'selfiter' in checkpoint:
        diffusion_model.iter = checkpoint['selfiter']
        print(f"Loaded iteration counter: {diffusion_model.iter}")

    loss_history = checkpoint.get('loss_history', [])
    eval_history = checkpoint.get('eval_history', {'img_auroc': [], 'px_auroc': [], 'epochs': []})
    best_loss = checkpoint.get('best_loss', float('inf'))
    best_epoch = checkpoint.get('best_epoch', 0)
    start_epoch = checkpoint.get('epoch', 0)
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous best loss: {best_loss:.6f} at epoch {best_epoch}")
    
    log_file = os.path.join(log_dir, 'training_log.json')
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            print(f"Loaded {len(logs)} previous training log entries")
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
    else:
        logs = []
    
    return start_epoch, loss_history, eval_history, best_loss, best_epoch, logs

def train_diffusion():
    device = torch.device("cuda" if _cuda else "cpu")
    print(f'Training diffusion for image restoration on class: {_category_name}')
    print(f"Batch size: {_batch_size}")
    print(f"VAE name: {_vae_name}")
    if _vae_name == "vae_resnet":
        print(f"Backbone: {_backbone}")

    os.makedirs(_pretrained_save_dir, exist_ok=True)
    os.makedirs(_train_result_dir, exist_ok=True)
    os.makedirs(_log_result_dir, exist_ok=True)

    print("Loading dataset...")
    train_dataset = load_mvtec_train_dataset(
        dataset_root_dir=_mvtec_data_dir,
        category=_category_name,
        image_size=_image_size,
        batch_size=_batch_size
    )

    # Load VAE model
    print("Loading VAE model...")
    vae_model = load_vae_model(
        checkpoint_path=_vae_pretrained_path,
        vae_name=_vae_name,
        input_channels=_input_channels,
        output_channels=_output_channels,
        z_dim=_z_dim,
        backbone=_backbone,
        dropout_p=_dropout_p,
        image_size=_image_size,
        device=device
    )

    start_epoch = 0
    total_epochs = _epochs
    loss_history = []
    eval_history = {'img_auroc': [], 'px_auroc': [], 'epochs': []}
    best_loss = float('inf')
    best_epoch = 0
    logs = []

    diffusion_model = diffusion.model.create_model()
    diffusion_model.set_noise_schedule_for_training()

    # Load checkpoint if resuming training
    if _resume_checkpoint and os.path.exists(_resume_checkpoint):
        start_epoch, loss_history, eval_history, best_loss, best_epoch, logs = load_checkpoint(
            _resume_checkpoint, diffusion_model, _log_result_dir
        )
        print(f"Training resumed from epoch {start_epoch}")
    else:
        logs = []
        if _resume_checkpoint:
            print(f"Warning: Checkpoint path {_resume_checkpoint} not found. Starting from scratch.")

    print(f"Starting training for {total_epochs} epochs...")
    print(f"Starting from epoch: {start_epoch + 1}")
    print(f"Evaluation interval: {_eval_interval} epochs")
    epoch_bar = tqdm(range(start_epoch, total_epochs), desc="Training Progress", position=0, leave=True)
    
    for epoch in epoch_bar:
        diffusion_model.netG.train()
        vae_model.eval()
        epoch_start_time = time.time()
        batch_losses = []
        
        batch_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{total_epochs}", position=1, leave=False)

        for batch_idx, batch in enumerate(batch_bar):
            images = batch['image'].to(device)

            with torch.no_grad():
                vae_reconstructed, _, _ = vae_model(images)

            diffusion_model.feed_data({'HR': images,'SR': vae_reconstructed})

            diffusion_model.optimize_parameters()
            
            # Get loss from diffusion model
            current_loss = diffusion_model.log_dict['l_pix']
            batch_losses.append(current_loss)
            
            # Update batch progress bar
            batch_bar.set_postfix({'Loss': f'{current_loss:.6f}','Avg Loss': f'{np.mean(batch_losses):.6f}'})

        # Calculate epoch statistics
        epoch_loss = np.mean(batch_losses)
        epoch_time = time.time() - epoch_start_time
        loss_history.append(epoch_loss)
        
        # Update epoch progress bar
        epoch_bar.set_postfix({
            'Loss': f'{epoch_loss:.6f}',
            'Time': f'{epoch_time:.2f}s',
            'Best': f'{best_loss:.6f}'
        })
        
        # Log training progress
        log_entry = {
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'time': epoch_time,
            'iter': diffusion_model.iter
        }
        
        # Append to existing logs
        logs.append(log_entry)
        
        # Save training log
        log_file = os.path.join(_log_result_dir, 'training_log.json')
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

        # Save best model if current loss is better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            diffusion_model.save_network(epoch + 1, diffusion_model.iter, "best", loss_history, eval_history, best_loss, best_epoch)
            print(f"\nNew best model saved! Loss: {best_loss:.6f} at epoch {best_epoch}")


        # Save final model at the end
        if epoch == total_epochs - 1:
            diffusion_model.save_network(epoch + 1, diffusion_model.iter, "final", loss_history, eval_history, best_loss, best_epoch)
    
    # Save training summary
    training_summary = {
        'category': _category_name,
        'total_epochs': total_epochs,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'final_loss': loss_history[-1] if loss_history else None,
        'loss_history': loss_history,
        'evaluation_history': eval_history,
        'training_completed': True,
        'timestamp': time.time(),
        'resumed_from_checkpoint': _resume_checkpoint if _resume_checkpoint and os.path.exists(_resume_checkpoint) else None
    }
    
    summary_file = os.path.join(_train_result_dir, 'training_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Results saved to: {_train_result_dir}")
    print(f"Model checkpoints saved to: {_pretrained_save_dir}")
    print(f"Evaluation logs saved to: {_log_result_dir}")


