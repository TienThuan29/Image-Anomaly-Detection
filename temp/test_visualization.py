#!/usr/bin/env python3
"""
Test script to demonstrate the visualization functionality.
This script creates dummy data and shows how the visualization functions work.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from inference import save_visualization_comparison, save_summary_visualization


def create_dummy_data(batch_size=4, image_size=256, channels=3):
    """Create dummy data for testing visualization functions."""
    
    # Create dummy images (input)
    images = torch.rand(batch_size, channels, image_size, image_size)
    
    # Create dummy VAE reconstructions (slightly different from input)
    recon_vae = images + 0.1 * torch.randn_like(images)
    recon_vae = torch.clamp(recon_vae, 0, 1)
    
    # Create dummy diffusion reconstructions (more different from input)
    recon_diffusion = images + 0.2 * torch.randn_like(images)
    recon_diffusion = torch.clamp(recon_diffusion, 0, 1)
    
    # Create dummy anomaly maps (random heat maps)
    anomaly_maps = torch.rand(batch_size, image_size, image_size)
    
    return images, recon_vae, recon_diffusion, anomaly_maps


def test_visualization_functions():
    """Test the visualization functions with dummy data."""
    
    print("Creating dummy data...")
    images, recon_vae, recon_diffusion, anomaly_maps = create_dummy_data(
        batch_size=4, image_size=128, channels=3
    )
    
    # Create test directory
    test_dir = "./test_visualizations"
    os.makedirs(test_dir, exist_ok=True)
    
    print("Testing comparison visualization...")
    save_visualization_comparison(
        images=images,
        recon_vae=recon_vae,
        recon_diffusion=recon_diffusion,
        anomaly_maps=anomaly_maps,
        save_dir=test_dir,
        batch_idx=0,
        category_name="test_category",
        max_images=4
    )
    
    print("Testing summary visualization...")
    # Create dummy labels and scores
    labels_all = [0, 0, 1, 1]  # 2 normal, 2 anomaly
    img_scores_all = [0.1, 0.15, 0.8, 0.9]  # Low scores for normal, high for anomaly
    maps_all = [anomaly_maps[i] for i in range(4)]
    
    save_summary_visualization(
        labels_all=labels_all,
        img_scores_all=img_scores_all,
        maps_all=maps_all,
        save_dir=test_dir,
        category_name="test_category"
    )
    
    print(f"Test completed! Check the '{test_dir}' directory for generated visualizations.")
    print("Files should include:")
    print("- test_category_batch_000_comparison.png")
    print("- batch_000_individual/ (directory with individual images)")
    print("- test_category_summary_analysis.png")


if __name__ == "__main__":
    test_visualization_functions() 