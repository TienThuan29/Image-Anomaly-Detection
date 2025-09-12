import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testing.inference_v3 import run_single_method_v3

def main():
    print("=" * 60)
    print("DiffusionVAE V3 Inference Demo")
    print("Combining Pixel + Perceptual Anomaly Detection")
    print("=" * 60)
    
    # Test different methods
    print("\n1. Testing individual methods...")
    
    # Test pixel-only method
    print("\n--- Pixel Only Method ---")
    run_single_method_v3(method='pixel_only')
    
    # Test perceptual-only method
    print("\n--- Perceptual Only Method ---")
    run_single_method_v3(method='perceptual')
    
    # Test multiscale perceptual method
    print("\n--- Multiscale Perceptual Method ---")
    run_single_method_v3(method='multiscale_perceptual')
    
    # Test combined methods with different weights
    print("\n--- Combined Methods with Different Weights ---")
    
    # Equal weights
    print("\n- Equal weights (0.5, 0.5)")
    run_single_method_v3(method='combined', pixel_weight=0.5, perceptual_weight=0.5, use_multiscale=False)
    
    # More weight on pixel
    print("\n- More pixel weight (0.7, 0.3)")
    run_single_method_v3(method='combined', pixel_weight=0.7, perceptual_weight=0.3, use_multiscale=False)
    
    # More weight on perceptual
    print("\n- More perceptual weight (0.3, 0.7)")
    run_single_method_v3(method='combined', pixel_weight=0.3, perceptual_weight=0.7, use_multiscale=False)
    
    # With multiscale perceptual
    print("\n- Equal weights with multiscale (0.5, 0.5, multiscale=True)")
    run_single_method_v3(method='combined', pixel_weight=0.5, perceptual_weight=0.5, use_multiscale=True)
    
    print("\n" + "=" * 60)
    print("Completed")
    print("=" * 60)

if __name__ == "__main__":
    main()
