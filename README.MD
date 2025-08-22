# Image Anomaly Detection

This project implements an image anomaly detection system using VAE (Variational Autoencoder) and Diffusion models.

## Features

### Visualization Support

The inference script now includes comprehensive visualization capabilities:

#### 1. Comparison Visualizations
- **Input Images**: Original test images
- **VAE Reconstructions**: Output from the VAE model
- **Diffusion Reconstructions**: Output from the diffusion model
- **Anomaly Heat Maps**: Pixel-level anomaly detection results

#### 2. Summary Analysis
- **Score Distributions**: Histograms of anomaly scores for normal vs anomalous samples
- **ROC Curves**: Performance visualization
- **Statistics Tables**: Detailed numerical analysis

#### 3. Configuration
Visualization settings can be controlled via `config.yml`:

```yaml
diffusion_testing:
  # Visualization settings
  save_visualizations: True
  max_visualization_batches: 5
  max_images_per_batch: 8
```

#### 4. Output Structure
Visualizations are saved in the following structure:
```
visualizations/
├── category_batch_000_comparison.png
├── category_batch_001_comparison.png
├── ...
├── batch_000_individual/
│   ├── category_input_00.png
│   ├── category_vae_recon_00.png
│   ├── category_diffusion_recon_00.png
│   └── category_anomaly_map_00.png
├── category_summary_analysis.png
└── category_results_summary.txt
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Inference with Visualizations

1. Configure your model paths in `config.yml`
2. Set visualization parameters as needed
3. Run the inference script:

```bash
cd src/main-v1.1
python inference.py
```

The script will automatically generate visualizations for the first few batches and save them to the `visualizations/` directory alongside the model checkpoints.

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- matplotlib >= 3.5.0 (for visualizations)
- Other dependencies listed in `requirements.txt` 