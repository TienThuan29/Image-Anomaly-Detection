#!/usr/bin/env python3
"""
Test script for inference_with_visualization_v2.py
This script tests the inference and visualization functionality.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_imports():
    """Test if all required modules can be imported."""
    print("[TEST] Testing imports...")
    
    try:
        from inference_with_visualization_v2 import (
            set_seed, normalize_image, tensor_to_numpy, 
            create_heatmap, overlay_heatmap_on_image,
            run_inference_with_visualization
        )
        print("[PASS] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_utility_functions():
    """Test utility functions with dummy data."""
    print("[TEST] Testing utility functions...")
    
    try:
        from inference_with_visualization_v2 import (
            set_seed, normalize_image, tensor_to_numpy, 
            create_heatmap, overlay_heatmap_on_image
        )
        
        # Test set_seed
        set_seed(42)
        print("[PASS] set_seed function works")
        
        # Test normalize_image
        dummy_tensor = torch.randn(3, 64, 64) * 2  # Random values outside [0,1]
        normalized = normalize_image(dummy_tensor)
        assert normalized.min() >= 0 and normalized.max() <= 1
        print("[PASS] normalize_image function works")
        
        # Test tensor_to_numpy
        dummy_tensor = torch.rand(1, 3, 64, 64)  # [B, C, H, W]
        numpy_array = tensor_to_numpy(dummy_tensor)
        assert numpy_array.shape == (64, 64, 3)
        assert numpy_array.min() >= 0 and numpy_array.max() <= 1
        print("[PASS] tensor_to_numpy function works")
        
        # Test create_heatmap
        dummy_heatmap = torch.rand(64, 64)
        heatmap_colored = create_heatmap(dummy_heatmap)
        assert heatmap_colored.shape == (64, 64, 3)
        print("[PASS] create_heatmap function works")
        
        # Test overlay_heatmap_on_image
        dummy_image = np.random.rand(64, 64, 3)
        dummy_heatmap = np.random.rand(64, 64, 3)
        overlay = overlay_heatmap_on_image(dummy_image, dummy_heatmap)
        assert overlay.shape == (64, 64, 3)
        print("[PASS] overlay_heatmap_on_image function works")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Utility function test error: {e}")
        return False

def test_config_loading():
    """Test if config file can be loaded."""
    print("[TEST] Testing config loading...")
    
    try:
        from config import load_config
        config = load_config("config.yml")
        
        # Check essential config sections
        assert hasattr(config, 'general')
        assert hasattr(config, 'data')
        assert hasattr(config, 'vae_model')
        assert hasattr(config, 'diffusion_model')
        assert hasattr(config, 'diffusion_testing')
        
        print("[PASS] Config loading successful")
        return True
        
    except Exception as e:
        print(f"[FAIL] Config loading error: {e}")
        return False

def test_model_paths():
    """Test if model paths exist."""
    print("[TEST] Testing model paths...")
    
    try:
        from config import load_config
        config = load_config("config.yml")
        
        # Check VAE model path
        vae_path = config.diffusion_testing.diffusion_vae_model_path
        if os.path.exists(vae_path):
            print(f"[PASS] VAE model path exists: {vae_path}")
        else:
            print(f"[WARN] VAE model path does not exist: {vae_path}")
        
        # Check diffusion model path
        diff_path = config.diffusion_testing.diffusion_model_path
        if os.path.exists(diff_path):
            print(f"[PASS] Diffusion model path exists: {diff_path}")
        else:
            print(f"[WARN] Diffusion model path does not exist: {diff_path}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Model path test error: {e}")
        return False

def test_dataset_path():
    """Test if dataset path exists."""
    print("[TEST] Testing dataset path...")
    
    try:
        from config import load_config
        config = load_config("config.yml")
        
        dataset_path = config.data.mvtec_data_dir
        if os.path.exists(dataset_path):
            print(f"[PASS] Dataset path exists: {dataset_path}")
            
            # Check if category directory exists
            category = config.data.category
            category_path = os.path.join(dataset_path, category)
            if os.path.exists(category_path):
                print(f"[PASS] Category path exists: {category_path}")
            else:
                print(f"[WARN] Category path does not exist: {category_path}")
        else:
            print(f"[WARN] Dataset path does not exist: {dataset_path}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Dataset path test error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Inference Visualization Script")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_utility_functions,
        test_config_loading,
        test_model_paths,
        test_dataset_path
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("[SUCCESS] All tests passed! The inference script should work correctly.")
        print("\nTo run the full inference with visualization:")
        print("python inference_with_visualization_v2.py")
    else:
        print("[WARNING] Some tests failed. Please check the issues above.")
        print("The inference script may not work correctly.")

if __name__ == "__main__":
    main() 