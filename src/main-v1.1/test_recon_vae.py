import torch
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from vae_unet_model import VAEUnet
from vae_resnet_model import VAEResNet
from utils import load_mvtec_only_good_test_dataset
from config import load_config

config = load_config()
# general
seed = config.general.seed
cuda = config.general.cuda
image_size = config.general.image_size
batch_size = config.general.batch_size
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

# vae testing
vae_model_path = config.vae_testing.vae_model_path
vae_test_result_dir = config.vae_testing.vae_test_result_base_dir + config.vae_testing.name + category_name +"/"


class TestReconstruction:
    def __init__(self):
        self.category_name = category_name
        self.device = torch.device(f"cuda:{cuda}" if cuda >= 0 and torch.cuda.is_available() else "cpu")
        print(f"Evaluating VAE on category: {self.category_name}")
        print(f"Device: {self.device}")

    def load_model(self, model_path):

        """Load pretrained VAE model"""
        if vae_name == 'vae_resnet':
            model = VAEResNet(
                in_channels=input_channels,
                out_channels=output_channels,
                latent_dim=z_dim,
                resnet_name=backbone,
                dropout_p=dropout_p
            ).to(self.device)
        elif vae_name == 'vae_unet':
            model = VAEUnet(
                in_channels=input_channels,
                latent_dim=z_dim,
                out_channels=output_channels
            ).to(self.device)
        else:
            raise ValueError(f"Unknown vae model: {vae_name}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Loaded model from: {model_path}")
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch'] + 1} epochs")

        return model


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

            # Save overall comparison image (original layout)
            comparison = torch.cat([sample_images, sample_recon], dim=0)
            overall_save_path = os.path.join(save_dir, f'{self.category_name}_reconstruction_overview.png')
            save_image(comparison, overall_save_path, nrow=num_samples, normalize=True)

            # Save individual paired images
            for i in range(num_samples):
                # Create pair: original on left, reconstruction on right
                pair = torch.cat([sample_images[i:i + 1], sample_recon[i:i + 1]], dim=0)
                pair_path = os.path.join(pairs_dir,
                                         f'{self.category_name}_pair_{i + 1:02d}_.png')
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

                # Save each pair in batch
                for i in range(images.size(0)):
                    if max_images and image_count >= max_images:
                        break

                    # Create pair: original on left, reconstruction on right
                    pair = torch.cat([images[i:i + 1], reconstructed[i:i + 1]], dim=0)
                    pair_path = os.path.join(pairs_dir,
                                             f'{self.category_name}_pair_{image_count + 1:04d}_.png')

                    save_image(pair, pair_path, nrow=2, normalize=True, padding=2)
                    image_count += 1

                if max_images and image_count >= max_images:
                    break

        print(f"Saved {image_count} reconstruction pairs to: {pairs_dir}")
        return image_count

    def run_evaluation(self, model_path, save_results_dir=None):
        model = self.load_model(model_path)

        # Setup save directory
        if save_results_dir is None:
            save_results_dir = vae_test_result_dir
        os.makedirs(save_results_dir, exist_ok=True)

        # Load only good test dataset (normal images)
        print("Loading good test dataset...")
        good_test_dataset = load_mvtec_only_good_test_dataset(
            dataset_root_dir=mvtec_data_dir,
            category=category_name,
            image_size=image_size,
            batch_size=batch_size
        )
        # Save sample reconstructions (from good test set)
        self.save_sample_reconstructions(model, good_test_dataset, save_results_dir)

        # Or automatically save a limited number of pairs
        print("\nSaving additional reconstruction pairs...")
        self.save_all_reconstruction_pairs(model, good_test_dataset, save_results_dir, max_images=50)


def main():
    evaluator = TestReconstruction("config.yml")
    model_path = vae_model_path

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please check the model path and try again.")
        return

    # Run evaluation
    evaluator.run_evaluation(model_path)

    print("\n=== Evaluation Completed Successfully! ===")


if __name__ == '__main__':
    main()

