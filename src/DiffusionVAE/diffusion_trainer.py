import diffusion.model
from vae.utils import load_vae
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
_eval_interval = config.diffusion_model.eval_interval

# save train results
_train_result_dir = config.diffusion_model.train_result_base_dir + _category_name + "/"
_log_result_dir = config.diffusion_model.train_result_base_dir + _category_name + "/" + "log_results/"
_pretrained_save_dir = config.diffusion_model.pretrained_save_base_dir + _category_name + "/"

def save_evaluation_log(epoch, image_auroc, pixel_auroc, log_dir):
    """Save evaluation results to a .txt log file."""
    log_file = os.path.join(log_dir, 'evaluation_log.txt')
    
    # Create log entry
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Epoch {epoch}: Image AUROC: {image_auroc:.4f}, Pixel AUROC: {pixel_auroc:.4f}\n"
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(log_entry)
    
    print(f"Evaluation at epoch {epoch}: Image AUROC: {image_auroc:.4f}, Pixel AUROC: {pixel_auroc:.4f}")

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
    vae_model = load_vae(
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

    diffusion_model = diffusion.model.create_model()
    diffusion_model.set_noise_schedule_for_training()

    print(f"Starting training for {total_epochs} epochs...")
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

            diffusion_model.feed_data({
                'HR': images,
                'SR': vae_reconstructed
            })

            diffusion_model.optimize_parameters()
            
            # Get loss from diffusion model
            current_loss = diffusion_model.log_dict['l_pix']
            batch_losses.append(current_loss)
            
            # Update batch progress bar
            batch_bar.set_postfix({
                'Loss': f'{current_loss:.6f}',
                'Avg Loss': f'{np.mean(batch_losses):.6f}'
            })

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
        
        # Save training log
        log_file = os.path.join(_log_result_dir, 'training_log.json')
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

        # Save best model if current loss is better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            diffusion_model.save_network(epoch + 1, diffusion_model.iter, "best")
            print(f"\nNew best model saved! Loss: {best_loss:.6f} at epoch {best_epoch}")

        # Run evaluation every eval_interval epochs
        if (epoch + 1) % _eval_interval == 0:
            print(f"\nRunning evaluation at epoch {epoch + 1}...")
            diffusion_model.netG.eval()
            # đặt noise schedule về num_step 200 cho testing
            diffusion_model.set_noise_schedule_for_val()
            # Run inference during training
            image_auroc, pixel_auroc = run_inference_during_training(vae_model, diffusion_model, _train_result_dir)
            
            # Save evaluation log
            save_evaluation_log(epoch + 1, image_auroc, pixel_auroc, _log_result_dir)
            
            # Store evaluation history
            eval_history['epochs'].append(epoch + 1)
            eval_history['img_auroc'].append(image_auroc)
            eval_history['px_auroc'].append(pixel_auroc)
            
            # Save evaluation history
            eval_history_file = os.path.join(_log_result_dir, 'evaluation_history.json')
            with open(eval_history_file, 'w') as f:
                json.dump(eval_history, f, indent=2)
            
            diffusion_model.netG.train()
            # trả noise schedule về trạng thái training
            diffusion_model.set_noise_schedule_for_training()
            # Save checkpoint
            diffusion_model.save_network(epoch + 1, diffusion_model.iter, "latest")

        # Save final model at the end
        if epoch == total_epochs - 1:
            diffusion_model.save_network(epoch + 1, diffusion_model.iter, "final")
    
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
        'timestamp': time.time()
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


