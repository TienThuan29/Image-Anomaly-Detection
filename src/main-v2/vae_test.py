import random
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from metric import Metric
from utils import ConfigLoader, load_mvtec_test_dataset
from vae_model import VAE

config_loader = ConfigLoader("config.yml")
config = config_loader.load_config()
data_config = config_loader.get_section("data")
vae_config = config_loader.get_section("vae_model")
testing_config = config_loader.get_section("testing")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

"""Compute pixel-wise reconstruction error using MSE"""
def compute_reconstruction_error(original_images, reconstructed_images):
    mse_loss = nn.MSELoss(reduction='none') # reduction='none' : return a tensor with same shape with input
    # Tính độ khác biệt giữa từng channels
    # Input: [batch_size, channels, height, width]
    # Output: error.shape == (batch_size, channels, height, width)
    # error[i, c, h, w] = (reconstructed_images[i, c, h, w] - original_images[i, c, h, w]) ** 2
    error = mse_loss(reconstructed_images, original_images)
    # Sum across channels to get per-pixel error
    error = torch.sum(error, dim=1, keepdim=True)
    return error


"""Test running"""
def inference_vae(model, dataloader, device):
    model.eval()
    labels_list = []
    predictions = []
    anomaly_map_list = []
    ground_truth_list = []

    print("Running VAE inference ...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward
            reconstructed, mu, logvar = model(images)

            # Compute reconstruction error (anomaly map)
            anomaly_map = compute_reconstruction_error(images, reconstructed)

            # Compute image-level anomaly score (mean reconstruction error per image)
            image_scores = torch.mean(anomaly_map.view(anomaly_map.size(0), -1), dim=1)

            # Store results
            labels_list.extend(labels.cpu().numpy())
            predictions.extend(image_scores.cpu().numpy())
            anomaly_map_list.append(anomaly_map.cpu())

            # Load ground_truth
            ground_truth_masks = batch["mask"].to(device)
            ground_truth_list.append(ground_truth_masks.cpu())

    return labels_list, predictions, anomaly_map_list, ground_truth_list

def run_inference():
    device = torch.device(f"cuda:{vae_config.get('cuda')}" if vae_config.get('cuda') >= 0 and torch.cuda.is_available() else "cpu")

    test_dataset = load_mvtec_test_dataset(
        dataset_root_dir=data_config.get('mvtec_data_dir'),
        category=data_config.get('category'),
        image_size=data_config.get('image_size'),
        batch_size=data_config.get('batch_size')
    )

    model = VAE(in_channels=vae_config.get('input_channels'), latent_dim=vae_config.get('z_dim')).to(device)

    # load model
    if os.path.exists(testing_config.get('vae_model_path')):
        checkpoint = torch.load(testing_config.get('vae_model_path'), map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {testing_config.get('vae_model_path')}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {testing_config.get('vae_model_path')}")

    labels_list, predictions, anomaly_map_list, ground_truth_list = inference_vae(model, test_dataset, device)

    # Create metric
    metric = Metric(
        labels_list=labels_list,
        predictions=predictions,
        anomaly_map_list=anomaly_map_list,
        gt_list=ground_truth_list
    )
    print("\nComputing metrics...")

    image_auroc = metric.image_auroc()
    print(f"Image-level AUROC: {image_auroc:.4f}")

    pixel_auroc = metric.pixel_auroc()
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}")

    print(f"\nDataset statistics:")
    print(f"Total samples: {len(labels_list)}")
    print(f"Normal samples: {labels_list.count(0)}")
    print(f"Anomalous samples: {labels_list.count(1)}")
    print(f"Mean anomaly score: {np.mean(predictions):.4f}")
    print(f"Std anomaly score: {np.std(predictions):.4f}")

if __name__ == "__main__":
    set_seed(vae_config.get('seed'))
    run_inference()