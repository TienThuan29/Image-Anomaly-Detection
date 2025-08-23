import os
import time
import numpy as np
import torch
import random
from diffusion_gaussian import GaussianDiffusion
from utils import load_mvtec_train_dataset
from diffusion_model import UNetModel
from tqdm import tqdm
from config import load_config
from utils import get_optimizer, load_vae
from inference import evaluate_model
from torch.functional import F
from visualization_report import log_evaluation_results, create_realtime_visualization, \
    create_evaluation_visualizations, create_evaluation_report, save_training_summary

config = load_config()

# general
_seed = config.general.seed
_cuda = config.general.cuda
_image_size = config.general.image_size
_batch_size = config.general.batch_size
_input_channels = config.general.input_channels  # 3
_output_channels = config.general.output_channels  # 3

# data
_category_name = config.data.category
_mvtec_data_dir = config.data.mvtec_data_dir
_mask = config.data.mask

# vae
_vae_name = config.vae_model.name
_backbone = config.vae_model.backbone

# diffusion
_diffusion_name = config.diffusion_model.name
_lr = float(config.diffusion_model.lr)
_weight_decay = float(config.diffusion_model.weight_decay)
_epochs = config.diffusion_model.epochs
_dropout_p = config.diffusion_model.dropout_p
_z_dim = config.diffusion_model.z_dim
_num_timesteps = config.diffusion_model.num_timesteps
_optimizer_name = config.diffusion_model.optimizer_name
_beta_schedule = config.diffusion_model.beta_schedule
_phase1_vae_pretrained_path = config.diffusion_model.phase1_vae_pretrained_path

# save train results
_train_result_dir = config.diffusion_model.train_result_base_dir + _category_name + "/"
_log_result_dir = config.diffusion_model.train_result_base_dir + _category_name + "/" + "log_results/"
_pretrained_save_dir = config.diffusion_model.pretrained_save_base_dir + _category_name + "/"

# eval
_eval_interval = config.diffusion_model.eval_interval

# Resume training settings - Add these to your config file or set directly
_resume_training = config.diffusion_model.resume_training
_resume_checkpoint_path = config.diffusion_model.resume_checkpoint_path


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def diffusion_loss_function(pred_image, target_image):
    return F.mse_loss(pred_image, target_image)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[INFO] Model state loaded successfully")

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("[INFO] Optimizer state loaded successfully")

    # Load scheduler state
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("[INFO] Scheduler state loaded successfully")

    # Get training history
    start_epoch = checkpoint.get('epoch', 0)
    loss_history = checkpoint.get('loss_history', [])
    eval_history = checkpoint.get('eval_history', {'img_auroc': [], 'px_auroc': [], 'epochs': []})

    # Ensure eval_history has the correct structure
    if not isinstance(eval_history, dict):
        eval_history = {'img_auroc': [], 'px_auroc': [], 'epochs': []}
    else:
        # Ensure all required keys exist
        if 'img_auroc' not in eval_history:
            eval_history['img_auroc'] = []
        if 'px_auroc' not in eval_history:
            eval_history['px_auroc'] = []
        if 'epochs' not in eval_history:
            eval_history['epochs'] = []

    print(f"[INFO] Resuming from epoch {start_epoch}")
    print(f"[INFO] Loaded {len(loss_history)} loss history entries")
    print(f"[INFO] Loaded {len(eval_history['img_auroc'])} evaluation history entries")

    # Validate checkpoint config compatibility (optional but recommended)
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        current_config = {
            'image_size': _image_size,
            'input_channels': _input_channels,
            'num_timesteps': _num_timesteps,
            'beta_schedule': _beta_schedule
        }

        for key, value in current_config.items():
            if key in checkpoint_config and checkpoint_config[key] != value:
                print(f"[WARNING] Config mismatch for {key}: checkpoint={checkpoint_config[key]}, current={value}")

    return start_epoch, loss_history, eval_history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 200)
    print(f"Using device: {device}")
    print("Category: ", _category_name)
    print("=" * 200)

    os.makedirs(_pretrained_save_dir, exist_ok=True)
    os.makedirs(_train_result_dir, exist_ok=True)
    os.makedirs(_log_result_dir, exist_ok=True)

    # Load VAE model
    print("[INFO] Loading VAE model...")
    vae_model = load_vae(
        checkpoint_path=_phase1_vae_pretrained_path,
        vae_name=_vae_name,
        input_channels=_input_channels,
        output_channels=_output_channels,
        z_dim=_z_dim,
        backbone=_backbone,
        dropout_p=_dropout_p,
        image_size=_image_size,
        device=device
    )

    # Load dataset
    print("[INFO] Loading dataset...")
    train_dataset = load_mvtec_train_dataset(
        dataset_root_dir=_mvtec_data_dir,
        category=_category_name,
        image_size=_image_size,
        batch_size=_batch_size
    )

    # Initialize UNet model
    print("[INFO] Initializing UNet model...")
    model = UNetModel(
        img_size=_image_size,
        base_channels=128,
        in_channels=_input_channels,
        output_activation="sigmoid",
        use_input_residual=True,
        output_scale=1.0,
        num_res_blocks=2,
        attention_resolutions="32,16,8",
        dropout=0.1
    ).to(device)

    gaussian_diffusion = GaussianDiffusion(num_timesteps=_num_timesteps, beta_schedule=_beta_schedule)

    optimizer = get_optimizer(
        optimizer_name=_optimizer_name,
        params=model.parameters(),
        lr=_lr,
        weight_decay=_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_epochs)

    # Initialize training state
    start_epoch = 0
    total_epochs = _epochs
    loss_history = []
    eval_history = {'img_auroc': [], 'px_auroc': [], 'epochs': []}

    # Check if resuming from checkpoint
    if _resume_training and _resume_checkpoint_path:
        try:
            start_epoch, loss_history, eval_history = load_checkpoint(
                _resume_checkpoint_path, model, optimizer, scheduler, device
            )
            print(f"[INFO] Successfully resumed training from epoch {start_epoch}")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {str(e)}")
            print("[INFO] Starting training from scratch...")
            start_epoch = 0
            loss_history = []
            eval_history = {'img_auroc': [], 'px_auroc': [], 'epochs': []}
    else:
        print("[INFO] Starting training from scratch...")

    print(f"Starting training from epoch {start_epoch + 1} to {total_epochs}...")

    # Main epoch progress bar
    epoch_bar = tqdm(range(start_epoch, total_epochs), desc="Training Progress", position=0, leave=True)

    for epoch in epoch_bar:
        model.train()
        vae_model.eval()
        epoch_start_time = time.time()
        batch_losses = []
        batch_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{total_epochs}", position=1, leave=False)

        for batch_idx, batch in enumerate(batch_bar):
            images = batch['image'].to(device)

            # Get reconstruct image from vae model phase 1
            with torch.no_grad():
                vae_reconstructed, _, _ = vae_model(images)

            B = images.size(0)
            t = torch.randint(0, _num_timesteps, (B,), device=device)
            noise = torch.randn_like(vae_reconstructed).to(device)
            # x_t: add noise
            x_t = gaussian_diffusion.q_sample(vae_reconstructed, t, noise)

            # Predict target image
            pred_images = model(x_t, t)

            # Calculate loss
            total_loss = diffusion_loss_function(pred_image=pred_images, target_image=images)
            batch_losses.append(total_loss.item())

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_bar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # Calculate epoch metrics
        epoch_loss = np.mean(batch_losses)
        loss_history.append(epoch_loss)
        epoch_time = time.time() - epoch_start_time

        scheduler.step()

        eval_info = ""
        if (epoch + 1) % _eval_interval == 0 and epoch != 1 and epoch != 0:
            print(f"\n[DEBUG] Running evaluation at epoch {epoch + 1}")
            img_auroc, px_auroc = evaluate_model(model, vae_model, gaussian_diffusion, device)
            eval_history['img_auroc'].append(img_auroc)
            eval_history['px_auroc'].append(px_auroc)
            eval_history['epochs'].append(epoch + 1)
            eval_info = f"Img AUROC: {img_auroc:.4f}, Px AUROC: {px_auroc:.4f}"

            # Debug: Print current evaluation history lengths
            print(
                f"[DEBUG] Evaluation history lengths - epochs: {len(eval_history['epochs'])}, img_auroc: {len(eval_history['img_auroc'])}, px_auroc: {len(eval_history['px_auroc'])}")

            # Log evaluation results
            log_evaluation_results(
                epoch=epoch + 1,
                img_auroc=img_auroc,
                px_auroc=px_auroc,
                loss=epoch_loss,
                eval_history=eval_history,
                log_dir=_train_result_dir
            )
            # Update real-time visualization
            create_realtime_visualization(
                eval_history=eval_history,
                loss_history=loss_history,
                save_dir=_train_result_dir,
                category_name=_category_name,
                current_epoch=epoch + 1
            )

        # Update epoch progress bar with comprehensive info
        epoch_bar.set_postfix({
            'Loss': f'{epoch_loss:.6f}',
            'Time': f'{epoch_time:.1f}s',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'Best Img AUROC': f'{max(eval_history["img_auroc"]):.4f}' if eval_history['img_auroc'] else 'N/A'
        })

        # Log epoch results
        if (epoch + 1) % 10 == 0 and (epoch + 1) > 99:
            print(f"\n[INFO] Epoch {epoch + 1}/{total_epochs} completed:")
            print(f"        Loss: {epoch_loss:.6f}")
            print(f"        Time: {epoch_time:.1f}s")
            print(f"        Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            if eval_info:
                print(f"        {eval_info}")
                eval_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'eval_history': eval_history,
                    'config': {
                        'image_size': _image_size,
                        'input_channels': _input_channels,
                        'num_timesteps': _num_timesteps,
                        'beta_schedule': _beta_schedule
                    }
                }
                torch.save(eval_checkpoint, os.path.join(_pretrained_save_dir, f'diffusion_model_{epoch + 1}_.pth'))

    # Final evaluation
    print("\n[INFO] Running final evaluation...")
    final_img_auroc, final_px_auroc = evaluate_model(model, vae_model, gaussian_diffusion, device)
    eval_history['img_auroc'].append(final_img_auroc)
    eval_history['px_auroc'].append(final_px_auroc)
    eval_history['epochs'].append(total_epochs)

    # Training completed
    print(f"\n[INFO] Training completed!")
    print(f"        Final loss: {loss_history[-1]:.6f}")
    print(f"        Best loss: {min(loss_history):.6f}")
    print(f"        Final Image AUROC: {final_img_auroc:.4f}")
    print(f"        Final Pixel AUROC: {final_px_auroc:.4f}")
    print(f"        Best Image AUROC: {max(eval_history['img_auroc']):.4f}")
    print(f"        Best Pixel AUROC: {max(eval_history['px_auroc']):.4f}")

    # Log final evaluation results
    log_evaluation_results(
        epoch=total_epochs,
        img_auroc=final_img_auroc,
        px_auroc=final_px_auroc,
        loss=loss_history[-1],
        eval_history=eval_history,
        log_dir=_train_result_dir
    )

    # Create and save visualizations
    print("\n[INFO] Creating evaluation visualizations...")
    create_evaluation_visualizations(
        eval_history=eval_history,
        loss_history=loss_history,
        save_dir=_log_result_dir,
        category_name=_category_name
    )

    # Save comprehensive training summary
    config_info = {
        'image_size': _image_size,
        'input_channels': _input_channels,
        'output_channels': _output_channels,
        'num_timesteps': _num_timesteps,
        'beta_schedule': _beta_schedule,
        'learning_rate': _lr,
        'weight_decay': _weight_decay,
        'epochs': _epochs,
        'batch_size': _batch_size,
        'z_dim': _z_dim,
        'dropout_p': _dropout_p,
        'optimizer_name': _optimizer_name,
        'vae_name': _vae_name,
        'backbone': _backbone
    }

    save_training_summary(
        eval_history=eval_history,
        loss_history=loss_history,
        save_dir=_train_result_dir,
        category_name=_category_name,
        config_info=config_info
    )

    # Create detailed evaluation report
    print("\n[INFO] Creating detailed evaluation report...")
    create_evaluation_report(
        eval_history=eval_history,
        loss_history=loss_history,
        save_dir=_train_result_dir,
        category_name=_category_name
    )

    # Save final model
    os.makedirs(_pretrained_save_dir, exist_ok=True)
    final_checkpoint = {
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
        'eval_history': eval_history,  # Add evaluation history to checkpoint
        'config': {
            'image_size': _image_size,
            'input_channels': _input_channels,
            'num_timesteps': _num_timesteps,
            'beta_schedule': _beta_schedule
        }
    }
    torch.save(final_checkpoint, os.path.join(_pretrained_save_dir, 'diffusion_model_final.pth'))
    print(f"[INFO] Final model saved to {_pretrained_save_dir}")


if __name__ == "__main__":
    set_seed(_seed)
    main()



# import os
# import time
# import numpy as np
# import torch
# import random
# from diffusion_gaussian import GaussianDiffusion
# from utils import load_mvtec_train_dataset
# from diffusion_model import UNetModel
# from tqdm import tqdm
# from config import load_config
# from utils import get_optimizer, load_vae
# from inference import evaluate_model
# from torch.functional import F
# from visualization_report import log_evaluation_results, create_realtime_visualization, create_evaluation_visualizations, create_evaluation_report, save_training_summary
#
# config = load_config()
#
# # general
# _seed = config.general.seed
# _cuda = config.general.cuda
# _image_size = config.general.image_size
# _batch_size = config.general.batch_size
# _input_channels = config.general.input_channels # 3
# _output_channels = config.general.output_channels # 3
#
# # data
# _category_name = config.data.category
# _mvtec_data_dir = config.data.mvtec_data_dir
# _mask = config.data.mask
#
# # vae
# _vae_name = config.vae_model.name
# _backbone = config.vae_model.backbone
#
# # diffusion
# _diffusion_name = config.diffusion_model.name
# _lr = float(config.diffusion_model.lr)
# _weight_decay = float(config.diffusion_model.weight_decay)
# _epochs = config.diffusion_model.epochs
# _dropout_p = config.diffusion_model.dropout_p
# _z_dim = config.diffusion_model.z_dim
# _num_timesteps = config.diffusion_model.num_timesteps
# _optimizer_name = config.diffusion_model.optimizer_name
# _beta_schedule = config.diffusion_model.beta_schedule
# _phase1_vae_pretrained_path = config.diffusion_model.phase1_vae_pretrained_path
#
# # save train results
# _train_result_dir = config.diffusion_model.train_result_base_dir + _category_name + "/"
# _log_result_dir = config.diffusion_model.train_result_base_dir + _category_name + "/" + "log_results/"
# _pretrained_save_dir = config.diffusion_model.pretrained_save_base_dir + _category_name + "/"
#
# # eval
# _eval_interval = config.diffusion_model.eval_interval
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
# def diffusion_loss_function(pred_image, target_image):
#     return F.mse_loss(pred_image, target_image)
#
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("=*200")
#     print(f"Using device: {device}")
#     print("Category: ", _category_name)
#     print("="*200)
#
#     os.makedirs(_pretrained_save_dir, exist_ok=True)
#     os.makedirs(_train_result_dir, exist_ok=True)
#     os.makedirs(_log_result_dir, exist_ok=True)
#
#     # Load VAE model
#     print("[INFO] Loading VAE model...")
#     vae_model = load_vae(
#         checkpoint_path=_phase1_vae_pretrained_path,
#         vae_name=_vae_name,
#         input_channels=_input_channels,
#         output_channels=_output_channels,
#         z_dim=_z_dim,
#         backbone=_backbone,
#         dropout_p=_dropout_p,
#         image_size=_image_size,
#         device=device
#     )
#
#     # Load dataset
#     print("[INFO] Loading dataset...")
#     train_dataset = load_mvtec_train_dataset(
#         dataset_root_dir=_mvtec_data_dir,
#         category=_category_name,
#         image_size=_image_size,
#         batch_size=_batch_size
#     )
#
#     # Initialize UNet model
#     print("[INFO] Initializing UNet model...")
#     model = UNetModel(
#         img_size=_image_size,
#         base_channels=128,
#         in_channels=_input_channels,
#         output_activation="sigmoid",
#         use_input_residual=True,
#         output_scale=1.0,
#         num_res_blocks=2,
#         attention_resolutions="32,16,8",
#         dropout=0.1
#     ).to(device)
#
#     gaussian_diffusion = GaussianDiffusion(num_timesteps=_num_timesteps,beta_schedule=_beta_schedule)
#
#     optimizer = get_optimizer(
#         optimizer_name=_optimizer_name,
#         params=model.parameters(),
#         lr=_lr,
#         weight_decay=_weight_decay
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_epochs)
#
#     start_epoch = 0
#     total_epochs = _epochs
#     loss_history = []
#     eval_history = {'img_auroc': [], 'px_auroc': [], 'epochs': []}
#
#     # Initialize evaluation history (don't add epoch 0 since we don't have evaluation results for it)
#     eval_history['img_auroc'] = []
#     eval_history['px_auroc'] = []
#     eval_history['epochs'] = []
#
#     print(f"Starting training for {total_epochs} epochs...")
#     # Main epoch progress bar
#     epoch_bar = tqdm(range(start_epoch, total_epochs),desc="Training Progress",position=0,leave=True)
#
#     for epoch in epoch_bar:
#         model.train()
#         vae_model.eval()
#         epoch_start_time = time.time()
#         batch_losses = []
#         batch_bar = tqdm(train_dataset,desc=f"Epoch {epoch + 1}/{total_epochs}",position=1,leave=False,)
#
#         for batch_idx, batch in enumerate(batch_bar):
#             images = batch['image'].to(device)
#
#             # Get reconstruct image from vae model phase 1
#             with torch.no_grad():
#                 vae_reconstructed, _, _ = vae_model(images)
#
#             B = images.size(0)
#             t = torch.randint(0, _num_timesteps, (B,), device=device)
#             noise = torch.randn_like(vae_reconstructed).to(device)
#             # x_t: add noise
#             x_t = gaussian_diffusion.q_sample(vae_reconstructed, t, noise)
#
#             # Predict target image
#             pred_images = model(x_t, t)
#
#             # Calculate loss
#             total_loss = diffusion_loss_function(pred_image=pred_images, target_image=images)
#             batch_losses.append(total_loss.item())
#
#             # Backward pass
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#
#             batch_bar.set_postfix({
#                 'Loss': f'{total_loss.item():.6f}',
#                 'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
#             })
#
#         # Calculate epoch metrics
#         epoch_loss = np.mean(batch_losses)
#         loss_history.append(epoch_loss)
#         epoch_time = time.time() - epoch_start_time
#
#         scheduler.step()
#
#         eval_info = ""
#         if (epoch + 1) % _eval_interval == 0 and epoch != 1 and epoch != 0:
#             print(f"\n[DEBUG] Running evaluation at epoch {epoch + 1}")
#             img_auroc, px_auroc = evaluate_model(model, vae_model, gaussian_diffusion, device)
#             eval_history['img_auroc'].append(img_auroc)
#             eval_history['px_auroc'].append(px_auroc)
#             eval_history['epochs'].append(epoch + 1)
#             eval_info = f"Img AUROC: {img_auroc:.4f}, Px AUROC: {px_auroc:.4f}"
#
#             # Debug: Print current evaluation history lengths
#             print(f"[DEBUG] Evaluation history lengths - epochs: {len(eval_history['epochs'])}, img_auroc: {len(eval_history['img_auroc'])}, px_auroc: {len(eval_history['px_auroc'])}")
#
#             # Log evaluation results
#             log_evaluation_results(
#                 epoch=epoch + 1,
#                 img_auroc=img_auroc,
#                 px_auroc=px_auroc,
#                 loss=epoch_loss,
#                 eval_history=eval_history,
#                 log_dir=_train_result_dir
#             )
#             # Update real-time visualization
#             create_realtime_visualization(
#                 eval_history=eval_history,
#                 loss_history=loss_history,
#                 save_dir=_train_result_dir,
#                 category_name=_category_name,
#                 current_epoch=epoch + 1
#             )
#
#         # Update epoch progress bar with comprehensive info
#         epoch_bar.set_postfix({
#             'Loss': f'{epoch_loss:.6f}',
#             'Time': f'{epoch_time:.1f}s',
#             'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
#             'Best Img AUROC': f'{max(eval_history["img_auroc"]):.4f}' if eval_history['img_auroc'] else 'N/A'
#         })
#
#         # Log epoch results
#         if (epoch + 1) % 10 == 0 and epoch != 1 and epoch != 0:
#             print(f"\n[INFO] Epoch {epoch + 1}/{total_epochs} completed:")
#             print(f"        Loss: {epoch_loss:.6f}")
#             print(f"        Time: {epoch_time:.1f}s")
#             print(f"        Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
#             if eval_info:
#                 print(f"        {eval_info}")
#                 eval_checkpoint = {
#                     'epoch': total_epochs,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'scheduler_state_dict': scheduler.state_dict(),
#                     'loss_history': loss_history,
#                     'config': {
#                         'image_size': _image_size,
#                         'input_channels': _input_channels,
#                         'num_timesteps': _num_timesteps,
#                         'beta_schedule': _beta_schedule
#                     }
#                 }
#                 torch.save(eval_checkpoint, os.path.join(_pretrained_save_dir, f'diffusion_model_{epoch+1}_.pth'))
#
#         # Save checkpoint periodically
#         if (epoch + 1) % 50 == 0:
#             os.makedirs(_pretrained_save_dir, exist_ok=True)
#             checkpoint = {
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'loss_history': loss_history,
#                 'config': {
#                     'image_size': _image_size,
#                     'input_channels': _input_channels,
#                     'num_timesteps': _num_timesteps,
#                     'beta_schedule': _beta_schedule
#                 }
#             }
#             torch.save(checkpoint, os.path.join(_pretrained_save_dir, f'diffusion_model_epoch_{epoch+1}.pth'))
#             print(f"[INFO] Checkpoint saved at epoch {epoch+1}")
#
#
#     # Final evaluation
#     print("\n[INFO] Running final evaluation...")
#     final_img_auroc, final_px_auroc = evaluate_model(model, vae_model, gaussian_diffusion, device)
#     eval_history['img_auroc'].append(final_img_auroc)
#     eval_history['px_auroc'].append(final_px_auroc)
#     eval_history['epochs'].append(total_epochs)
#
#     # Training completed
#     print(f"\n[INFO] Training completed!")
#     print(f"        Final loss: {loss_history[-1]:.6f}")
#     print(f"        Best loss: {min(loss_history):.6f}")
#     print(f"        Final Image AUROC: {final_img_auroc:.4f}")
#     print(f"        Final Pixel AUROC: {final_px_auroc:.4f}")
#     print(f"        Best Image AUROC: {max(eval_history['img_auroc']):.4f}")
#     print(f"        Best Pixel AUROC: {max(eval_history['px_auroc']):.4f}")
#
#     # Log final evaluation results
#     log_evaluation_results(
#         epoch=total_epochs,
#         img_auroc=final_img_auroc,
#         px_auroc=final_px_auroc,
#         loss=loss_history[-1],
#         eval_history=eval_history,
#         log_dir=_train_result_dir
#     )
#
#     # Create and save visualizations
#     print("\n[INFO] Creating evaluation visualizations...")
#     create_evaluation_visualizations(
#         eval_history=eval_history,
#         loss_history=loss_history,
#         save_dir=_log_result_dir,
#         category_name=_category_name
#     )
#
#     # Save comprehensive training summary
#     config_info = {
#         'image_size': _image_size,
#         'input_channels': _input_channels,
#         'output_channels': _output_channels,
#         'num_timesteps': _num_timesteps,
#         'beta_schedule': _beta_schedule,
#         'learning_rate': _lr,
#         'weight_decay': _weight_decay,
#         'epochs': _epochs,
#         'batch_size': _batch_size,
#         'z_dim': _z_dim,
#         'dropout_p': _dropout_p,
#         'optimizer_name': _optimizer_name,
#         'vae_name': _vae_name,
#         'backbone': _backbone
#     }
#
#     save_training_summary(
#         eval_history=eval_history,
#         loss_history=loss_history,
#         save_dir=_train_result_dir,
#         category_name=_category_name,
#         config_info=config_info
#     )
#
#     # Create detailed evaluation report
#     print("\n[INFO] Creating detailed evaluation report...")
#     create_evaluation_report(
#         eval_history=eval_history,
#         loss_history=loss_history,
#         save_dir=_train_result_dir,
#         category_name=_category_name
#     )
#
#     # Save final model
#     os.makedirs(_pretrained_save_dir, exist_ok=True)
#     final_checkpoint = {
#         'epoch': total_epochs,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'loss_history': loss_history,
#         'eval_history': eval_history,  # Add evaluation history to checkpoint
#         'config': {
#             'image_size': _image_size,
#             'input_channels': _input_channels,
#             'num_timesteps': _num_timesteps,
#             'beta_schedule': _beta_schedule
#         }
#     }
#     torch.save(final_checkpoint, os.path.join(_pretrained_save_dir, 'diffusion_model_final.pth'))
#     print(f"[INFO] Final model saved to {_pretrained_save_dir}")
#
#
# if __name__ == "__main__":
#     set_seed(_seed)
#     main()
