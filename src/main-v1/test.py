# import time
# from torchvision.utils import save_image
# import torch
# from torch import Tensor
# import torch.nn.functional as F
# import numpy as np
# import random
# import os
# from tqdm import tqdm
# from vae_model import VAE
# from utils import ConfigLoader, load_mvtec_train_dataset, load_mvtec_only_good_test_dataset
#
# config_loader = ConfigLoader("config.yml")
# config = config_loader.load_config()
# data_config = config_loader.get_section("data")
# vae_config = config_loader.get_section("vae_model")
# early_stopping_config = config_loader.get_section("early_stopping")
# testing_config = config_loader.get_section("testing")
#
# category_name = data_config.get('category')
# test_result_dir = testing_config.get('vae_test_result_base_dir') + category_name
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
# def run_testing():
#     # Initialize device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Initialize and load the model
#     model = VAE(
#         in_channels=vae_config.get('input_channels'),
#         latent_dim=vae_config.get('z_dim')
#     ).to(device)
#
#     # load model
#     if os.path.exists(testing_config.get('vae_model_path')):
#         checkpoint = torch.load(testing_config.get('vae_model_path'), map_location=device)
#         if 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             model.load_state_dict(checkpoint)
#         print(f"Loaded model from {testing_config.get('vae_model_path')}")
#     else:
#         raise FileNotFoundError(f"Model checkpoint not found at {testing_config.get('vae_model_path')}")
#
#     # Load test dataset
#     test_dataset = load_mvtec_only_good_test_dataset(
#         dataset_root_dir=data_config.get('mvtec_data_dir'),
#         category=data_config.get('category'),
#         image_size=data_config.get('image_size'),
#         batch_size=data_config.get('batch_size')
#     )
#
#     model.eval()
#     test_mse_losses = []
#
#     print("Running inference on test dataset...")
#     with torch.no_grad():
#         for test_batch in tqdm(test_dataset, desc="Testing"):
#             test_images = test_batch['image'].to(device)
#             test_reconstructed, _, _ = model(test_images)
#
#             # Calculate MSE for good test dataset
#             batch_mse = F.mse_loss(test_reconstructed, test_images, reduction='none')
#             # Average over spatial dimensions (C, H, W) to get MSE per image
#             batch_mse = batch_mse.view(batch_mse.size(0), -1).mean(dim=1)
#             test_mse_losses.extend(batch_mse.cpu().numpy())
#
#     # Calculate average MSE across all test images
#     avg_test_mse = np.mean(test_mse_losses)
#     print(f"Average test MSE: {avg_test_mse:.6f}")
#
#     # Save results
#     results_file = os.path.join(test_result_dir, 'test_results.txt')
#     os.makedirs(test_result_dir, exist_ok=True)
#     with open(results_file, 'w') as f:
#         f.write(f"Average test MSE: {avg_test_mse:.6f}\n")
#         f.write(f"Number of test samples: {len(test_mse_losses)}\n")
#
#     print(f"Results saved to {results_file}")
#     return avg_test_mse, test_mse_losses
#
#
# if __name__ == '__main__':
#     set_seed(vae_config.get('seed'))
#     avg_mse, mse_losses = run_testing()


import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from vae_model import VAEResNet
from utils import ConfigLoader, load_mvtec_test_dataset, load_mvtec_only_good_test_dataset
import warnings

warnings.filterwarnings('ignore')

config_loader = ConfigLoader("config.yml")
config = config_loader.load_config()
data_config = config_loader.get_section("data")
test_config = config_loader.get_section("testing")
vae_config = config_loader.get_section("vae_model")
early_stopping_config = config_loader.get_section("early_stopping")

category_name = data_config.get('category')
train_result_dir = vae_config.get('train_result_base_dir') + category_name
pretrained_save_dir = vae_config.get('pretrained_save_base_dir') + category_name


class VAESSIMEvaluator:
    def __init__(self, config_path="config.yml"):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        self.data_config = self.config_loader.get_section("data")
        self.vae_config = self.config_loader.get_section("vae_model")

        self.category_name = self.data_config.get('category')
        self.device = torch.device(f"cuda:{self.vae_config.get('cuda')}" if self.vae_config.get(
            'cuda') >= 0 and torch.cuda.is_available() else "cpu")

        # Initialize SSIM module
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)
        self.ms_ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=3)

        print(f"Evaluating VAE on category: {self.category_name}")
        print(f"Device: {self.device}")

    def load_model(self, model_path):
        """Load pretrained VAE model"""
        model = VAEResNet(
            in_channels=self.vae_config.get('input_channels'),
            latent_dim=self.vae_config.get('z_dim')
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Loaded model from: {model_path}")
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch'] + 1} epochs")

        return model

    def calculate_reconstruction_metrics(self, original, reconstructed):
        """Calculate MSE, RMSE, SSIM, and MS-SSIM metrics"""
        with torch.no_grad():
            # MSE and RMSE
            mse = F.mse_loss(reconstructed, original, reduction='none')
            mse_per_image = mse.view(mse.size(0), -1).mean(dim=1)
            rmse_per_image = torch.sqrt(mse_per_image)

            # SSIM (per image)
            ssim_scores = []
            ms_ssim_scores = []

            for i in range(original.size(0)):
                # SSIM for single image pair
                ssim_val = ssim(
                    original[i:i + 1], reconstructed[i:i + 1],
                    data_range=1.0, size_average=True
                )
                ssim_scores.append(ssim_val.item())

                # MS-SSIM for single image pair
                ms_ssim_val = ms_ssim(
                    original[i:i + 1], reconstructed[i:i + 1],
                    data_range=1.0, size_average=True
                )
                ms_ssim_scores.append(ms_ssim_val.item())

            return {
                'mse': mse_per_image.cpu().numpy(),
                'rmse': rmse_per_image.cpu().numpy(),
                'ssim': np.array(ssim_scores),
                'ms_ssim': np.array(ms_ssim_scores)
            }

    def evaluate_on_dataset(self, model, dataset, dataset_name="Dataset"):
        """Evaluate model on a dataset and return detailed metrics"""
        all_metrics = {
            'mse': [], 'rmse': [], 'ssim': [], 'ms_ssim': [],
            'labels': [], 'image_paths': []
        }

        print(f"\nEvaluating on {dataset_name}...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
                images = batch['image'].to(self.device)
                labels = batch.get('label', torch.zeros(images.size(0)))

                # Get reconstructions
                reconstructed, _, _ = model(images)

                # Calculate metrics
                metrics = self.calculate_reconstruction_metrics(images, reconstructed)

                # Store results
                all_metrics['mse'].extend(metrics['mse'])
                all_metrics['rmse'].extend(metrics['rmse'])
                all_metrics['ssim'].extend(metrics['ssim'])
                all_metrics['ms_ssim'].extend(metrics['ms_ssim'])
                all_metrics['labels'].extend(labels.numpy())

        # Convert to numpy arrays
        for key in ['mse', 'rmse', 'ssim', 'ms_ssim', 'labels']:
            all_metrics[key] = np.array(all_metrics[key])

        return all_metrics

    def calculate_anomaly_scores(self, metrics):
        """Calculate anomaly scores based on different metrics"""
        anomaly_scores = {
            'mse_score': metrics['mse'],  # Higher MSE = more anomalous
            'rmse_score': metrics['rmse'],  # Higher RMSE = more anomalous
            'ssim_score': 1 - metrics['ssim'],  # Lower SSIM = more anomalous
            'ms_ssim_score': 1 - metrics['ms_ssim'],  # Lower MS-SSIM = more anomalous
            # Combined score (you can adjust weights)
            'combined_score': 0.3 * metrics['mse'] + 0.3 * metrics['rmse'] +
                              0.2 * (1 - metrics['ssim']) + 0.2 * (1 - metrics['ms_ssim'])
        }
        return anomaly_scores

    def calculate_auroc(self, anomaly_scores, labels):
        """Calculate AUROC for different anomaly scores"""
        auroc_results = {}

        for score_name, scores in anomaly_scores.items():
            if len(np.unique(labels)) > 1:  # Only calculate if we have both normal and anomalous samples
                auroc = roc_auc_score(labels, scores)
                auroc_results[score_name] = auroc
            else:
                auroc_results[score_name] = None

        return auroc_results

    def print_statistics(self, metrics, dataset_name):
        """Print detailed statistics for good test images only"""
        print(f"\n=== {dataset_name} Statistics ===")
        print(f"Number of images: {len(metrics['mse'])}")

        metrics_to_show = ['mse', 'rmse', 'ssim', 'ms_ssim']

        for metric_name in metrics_to_show:
            values = metrics[metric_name]
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean:   {values.mean():.6f}")
            print(f"  Std:    {values.std():.6f}")
            print(f"  Min:    {values.min():.6f}")
            print(f"  Max:    {values.max():.6f}")
            print(f"  Median: {np.median(values):.6f}")

    def save_sample_reconstructions(self, model, dataset, save_dir, num_samples=8):
        """Save sample reconstructions for visual inspection"""
        os.makedirs(save_dir, exist_ok=True)

        # Create subdirectory for paired images
        pairs_dir = os.path.join(save_dir, 'reconstruction_pairs')
        os.makedirs(pairs_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataset))
            sample_images = sample_batch['image'][:num_samples].to(self.device)
            sample_recon, _, _ = model(sample_images)
            sample_labels = sample_batch.get('label', torch.zeros(num_samples))[:num_samples]

            # Calculate metrics for samples
            metrics = self.calculate_reconstruction_metrics(sample_images, sample_recon)

            # Save overall comparison image (original layout)
            comparison = torch.cat([sample_images, sample_recon], dim=0)
            overall_save_path = os.path.join(save_dir, f'{self.category_name}_reconstruction_overview.png')
            save_image(comparison, overall_save_path, nrow=num_samples, normalize=True)

            # Save individual paired images
            for i in range(num_samples):
                # Create pair: original on left, reconstruction on right
                pair = torch.cat([sample_images[i:i + 1], sample_recon[i:i + 1]], dim=0)
                pair_path = os.path.join(pairs_dir,
                                         f'{self.category_name}_pair_{i + 1:02d}_ssim_{metrics["ssim"][i]:.4f}.png')
                save_image(pair, pair_path, nrow=2, normalize=True, padding=2)

            # Save individual original and reconstruction images separately
            originals_dir = os.path.join(save_dir, 'originals')
            reconstructions_dir = os.path.join(save_dir, 'reconstructions')
            os.makedirs(originals_dir, exist_ok=True)
            os.makedirs(reconstructions_dir, exist_ok=True)

            for i in range(num_samples):
                # Original image
                orig_path = os.path.join(originals_dir, f'{self.category_name}_original_{i + 1:02d}.png')
                save_image(sample_images[i:i + 1], orig_path, normalize=True)

                # Reconstruction image
                recon_path = os.path.join(reconstructions_dir, f'{self.category_name}_reconstruction_{i + 1:02d}.png')
                save_image(sample_recon[i:i + 1], recon_path, normalize=True)

            # Save individual metrics
            metrics_path = os.path.join(save_dir, f'{self.category_name}_good_test_sample_metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Sample Reconstruction Metrics for {self.category_name}\n")
                f.write("(Good Test Images Only)\n")
                f.write("=" * 50 + "\n\n")

                for i in range(num_samples):
                    f.write(f"Sample {i + 1:02d} (Label: {sample_labels[i].item()}):\n")
                    f.write(f"  MSE: {metrics['mse'][i]:.6f}\n")
                    f.write(f"  RMSE: {metrics['rmse'][i]:.6f}\n")
                    f.write(f"  SSIM: {metrics['ssim'][i]:.6f}\n")
                    f.write(f"  MS-SSIM: {metrics['ms_ssim'][i]:.6f}\n")
                    f.write(f"  Files:\n")
                    f.write(
                        f"    - Pair: reconstruction_pairs/{self.category_name}_pair_{i + 1:02d}_ssim_{metrics['ssim'][i]:.4f}.png\n")
                    f.write(f"    - Original: originals/{self.category_name}_original_{i + 1:02d}.png\n")
                    f.write(
                        f"    - Reconstruction: reconstructions/{self.category_name}_reconstruction_{i + 1:02d}.png\n")
                    f.write("-" * 50 + "\n")

            print(f"Overview image saved to: {overall_save_path}")
            print(f"Paired images saved to: {pairs_dir}")
            print(f"Original images saved to: {originals_dir}")
            print(f"Reconstruction images saved to: {reconstructions_dir}")

    def save_all_reconstruction_pairs(self, model, dataset, save_dir, max_images=None):
        """Save all reconstruction pairs from dataset"""
        pairs_dir = os.path.join(save_dir, 'all_reconstruction_pairs')
        os.makedirs(pairs_dir, exist_ok=True)

        model.eval()
        image_count = 0

        print(f"\nSaving all reconstruction pairs...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataset, desc="Saving pairs")):
                images = batch['image'].to(self.device)
                reconstructed, _, _ = model(images)

                # Calculate metrics for this batch
                metrics = self.calculate_reconstruction_metrics(images, reconstructed)

                # Save each pair in batch
                for i in range(images.size(0)):
                    if max_images and image_count >= max_images:
                        break

                    # Create pair: original on left, reconstruction on right
                    pair = torch.cat([images[i:i + 1], reconstructed[i:i + 1]], dim=0)

                    # Include SSIM score in filename
                    ssim_score = metrics['ssim'][i]
                    pair_path = os.path.join(pairs_dir,
                                             f'{self.category_name}_pair_{image_count + 1:04d}_ssim_{ssim_score:.4f}.png')

                    save_image(pair, pair_path, nrow=2, normalize=True, padding=2)
                    image_count += 1

                if max_images and image_count >= max_images:
                    break

        print(f"Saved {image_count} reconstruction pairs to: {pairs_dir}")
        return image_count

    def run_evaluation(self, model_path, save_results_dir=None):
        """Run evaluation pipeline - only on good test images"""
        # Load model
        model = self.load_model(model_path)

        # Setup save directory
        if save_results_dir is None:
            save_results_dir = f"evaluation_results/{self.category_name}"
        os.makedirs(save_results_dir, exist_ok=True)

        # Load only good test dataset (normal images)
        print("Loading good test dataset...")
        good_test_dataset = load_mvtec_only_good_test_dataset(
            dataset_root_dir=self.data_config.get('mvtec_data_dir'),
            category=self.data_config.get('category'),
            image_size=self.data_config.get('image_size'),
            batch_size=self.data_config.get('batch_size')
        )

        # Evaluate only on good test dataset
        good_test_metrics = self.evaluate_on_dataset(model, good_test_dataset, "Good Test Set")

        # Print statistics
        self.print_statistics(good_test_metrics, "Good Test Set")

        # Save sample reconstructions (from good test set)
        self.save_sample_reconstructions(model, good_test_dataset, save_results_dir)

        # Optionally save all reconstruction pairs (uncomment if needed)
        # print("\nDo you want to save all reconstruction pairs? (y/n): ", end="")
        # save_all = input().lower().strip() == 'y'
        # if save_all:
        #     max_images = int(input("Maximum number of pairs to save (or 0 for all): "))
        #     if max_images == 0:
        #         max_images = None
        #     self.save_all_reconstruction_pairs(model, good_test_dataset, save_results_dir, max_images)

        # Or automatically save a limited number of pairs
        print("\nSaving additional reconstruction pairs...")
        self.save_all_reconstruction_pairs(model, good_test_dataset, save_results_dir, max_images=50)

        # Save detailed results
        results_file = os.path.join(save_results_dir, f'{self.category_name}_good_test_evaluation.txt')
        with open(results_file, 'w') as f:
            f.write(f"VAE SSIM Evaluation Results for {self.category_name}\n")
            f.write("(Evaluated only on Good Test Images)\n")
            f.write("=" * 60 + "\n\n")

            # Good test set results
            f.write("GOOD TEST SET RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of images: {len(good_test_metrics['mse'])}\n\n")

            for metric_name in ['mse', 'rmse', 'ssim', 'ms_ssim']:
                values = good_test_metrics[metric_name]
                f.write(f"{metric_name.upper()}:\n")
                f.write(f"  Mean: {values.mean():.6f}\n")
                f.write(f"  Std:  {values.std():.6f}\n")
                f.write(f"  Min:  {values.min():.6f}\n")
                f.write(f"  Max:  {values.max():.6f}\n")
                f.write(f"  Median: {np.median(values):.6f}\n\n")

        print(f"\nDetailed results saved to: {results_file}")

        return {
            'good_test_metrics': good_test_metrics
        }


def main():
    """Main evaluation function"""
    # Initialize evaluator
    evaluator = VAESSIMEvaluator("config.yml")

    # Model path - adjust this to your model path
    category_name = evaluator.category_name
    pretrained_save_dir = evaluator.vae_config.get('pretrained_save_base_dir') + category_name
    # model_path = f'{pretrained_save_dir}/{category_name}_vae_best_test_mse.pth'  # or _vae_final.pth
    model_path = test_config.get("vae_model_path")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please check the model path and try again.")
        return

    # Run evaluation
    results = evaluator.run_evaluation(model_path)

    print("\n=== Evaluation Completed Successfully! ===")


if __name__ == '__main__':
    main()

