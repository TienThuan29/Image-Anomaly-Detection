import torch
import os
from tqdm import tqdm
from torchvision.utils import save_image
from utils import load_vae
# from vae_resnet_skip_model import VAEResNetWithSkip
from utils import load_mvtec_test_dataset
from config import load_config

config = load_config()
# general
_seed = config.general.seed
_cuda = config.general.cuda
_image_size = config.general.image_size
_batch_size = config.general.batch_size
_input_channels = config.general.input_channels
_output_channels = config.general.output_channels

# data
category_name = config.data.category
mvtec_data_dir = config.data.mvtec_data_dir
mask = config.data.mask

# vae
_vae_name = config.vae_model.name
_epochs = config.vae_model.epochs
_z_dim = config.vae_model.z_dim
_lr = float(config.vae_model.lr)
_dropout_p = config.vae_model.dropout_p
_weight_decay = float(config.vae_model.weight_decay)
_save_freq = config.vae_model.save_freq
_sample_freq = config.vae_model.sample_freq
_backbone = config.vae_model.backbone # backbone for resnet
_optimizer_name = config.vae_model.optimizer_name
_resume_checkpoint_path = config.vae_model.resume_checkpoint

# vae testing
_vae_model_path = config.vae_testing.vae_model_path
_vae_test_result_dir = config.vae_testing.vae_test_result_base_dir + config.vae_testing.name + category_name +"/"


class TestReconstruction:
    def __init__(self):
        self.category_name = category_name
        self.device = torch.device(f"cuda:{_cuda}" if _cuda >= 0 and torch.cuda.is_available() else "cpu")
        print(f"Evaluating VAE on category: {self.category_name}")
        print(f"Device: {self.device}")

    def save_sample_reconstructions(self, model, dataset, save_dir, num_samples=8):
        os.makedirs(save_dir, exist_ok=True)
        pairs_dir = os.path.join(save_dir, 'reconstruction_pairs')
        os.makedirs(pairs_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataset))
            sample_images = sample_batch['image'][:num_samples].to(self.device)
            sample_recon, _, _ = model(sample_images)

            comparison = torch.cat([sample_images, sample_recon], dim=0)
            overall_save_path = os.path.join(save_dir, f'{self.category_name}_reconstruction_overview.png')
            save_image(comparison, overall_save_path, nrow=num_samples, normalize=True)

            # Save individual paired images
            for i in range(num_samples):
                # Create pair
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

    def run_evaluation(self, save_results_dir=None):
        model = load_vae(
            checkpoint_path=_vae_model_path,
            vae_name=_vae_name,
            input_channels=_input_channels,
            output_channels=_output_channels,
            z_dim=_z_dim,
            backbone=_backbone,
            dropout_p=_dropout_p,
            image_size=_image_size,
            device=self.device
        )

        if save_results_dir is None: save_results_dir = _vae_test_result_dir
        os.makedirs(save_results_dir, exist_ok=True)

        print("Loading good test dataset...")
        good_test_dataset = load_mvtec_test_dataset(
            dataset_root_dir=mvtec_data_dir,
            category=category_name,
            image_size=_image_size,
            batch_size=_batch_size
        )
        self.save_sample_reconstructions(model, good_test_dataset, save_results_dir)
        self.save_all_reconstruction_pairs(model, good_test_dataset, save_results_dir, max_images=20)


def main():
    evaluator = TestReconstruction()
    evaluator.run_evaluation()
    print("\n=== Evaluation Completed Successfully! ===")


if __name__ == '__main__':
    main()

