import os
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

def validate_eval_history(eval_history: Dict[str, List]) -> Dict[str, List]:
    """Validate and fix evaluation history arrays to ensure they have the same length."""
    epochs = eval_history.get('epochs', [])
    img_auroc = eval_history.get('img_auroc', [])
    px_auroc = eval_history.get('px_auroc', [])

    # Find the minimum length
    min_len = min(len(epochs), len(img_auroc), len(px_auroc))

    if min_len == 0:
        print("[WARNING] No evaluation data available")
        return eval_history

    # Check if arrays have different lengths
    if len(epochs) != len(img_auroc) or len(epochs) != len(px_auroc) or len(img_auroc) != len(px_auroc):
        print(
            f"[WARNING] Evaluation arrays have different lengths: epochs={len(epochs)}, img_auroc={len(img_auroc)}, px_auroc={len(px_auroc)}")
        print(f"[INFO] Truncating all arrays to minimum length: {min_len}")

        # Truncate arrays to minimum length
        eval_history['epochs'] = epochs[:min_len]
        eval_history['img_auroc'] = img_auroc[:min_len]
        eval_history['px_auroc'] = px_auroc[:min_len]

    return eval_history


def log_evaluation_results(epoch: int, img_auroc: float, px_auroc: float, loss: float,
                           eval_history: Dict[str, List], log_dir: str) -> None:
    """Log evaluation results to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create log entry
    log_entry = {
        "timestamp": timestamp,
        "epoch": epoch,
        "image_auroc": img_auroc,
        "pixel_auroc": px_auroc,
        "loss": loss,
        "best_image_auroc": max(eval_history["img_auroc"]) if eval_history["img_auroc"] else 0.0,
        "best_pixel_auroc": max(eval_history["px_auroc"]) if eval_history["px_auroc"] else 0.0
    }

    # Save to JSON file
    log_file = os.path.join(log_dir, "evaluation_log.json")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

    # Print detailed log
    print(f"\n{'=' * 80}")
    print(f"[EVALUATION LOG] Epoch {epoch} - {timestamp}")
    print(f"{'=' * 80}")
    print(f"Image AUROC:     {img_auroc:.4f}")
    print(f"Pixel AUROC:     {px_auroc:.4f}")
    print(f"Current Loss:    {loss:.6f}")
    print(f"Best Image AUROC: {max(eval_history['img_auroc']):.4f}" if eval_history[
        'img_auroc'] else "Best Image AUROC: N/A")
    print(f"Best Pixel AUROC: {max(eval_history['px_auroc']):.4f}" if eval_history[
        'px_auroc'] else "Best Pixel AUROC: N/A")
    print(f"{'=' * 80}")


def create_evaluation_visualizations(eval_history: Dict[str, List], loss_history: List[float],
                                     save_dir: str, category_name: str) -> None:
    """Create and save visualization charts for training progress and evaluation metrics."""
    # Validate evaluation history
    eval_history = validate_eval_history(eval_history)

    plt.style.use('default')

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Progress and Evaluation Metrics - {category_name}', fontsize=16, fontweight='bold')

    # Plot 1: Training Loss
    epochs = list(range(1, len(loss_history) + 1))
    ax1.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Image AUROC
    if eval_history['epochs'] and eval_history['img_auroc']:
        # Ensure arrays have the same length
        min_len = min(len(eval_history['epochs']), len(eval_history['img_auroc']))
        if min_len > 0:
            epochs_plot = eval_history['epochs'][:min_len]
            img_auroc_plot = eval_history['img_auroc'][:min_len]
            ax2.plot(epochs_plot, img_auroc_plot, 'g-o', linewidth=2, markersize=6, label='Image AUROC')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Image AUROC')
            ax2.set_title('Image-Level AUROC Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(0, 1)

    # Plot 3: Pixel AUROC
    if eval_history['epochs'] and eval_history['px_auroc']:
        # Ensure arrays have the same length
        min_len = min(len(eval_history['epochs']), len(eval_history['px_auroc']))
        if min_len > 0:
            epochs_plot = eval_history['epochs'][:min_len]
            px_auroc_plot = eval_history['px_auroc'][:min_len]
            ax3.plot(epochs_plot, px_auroc_plot, 'r-s', linewidth=2, markersize=6, label='Pixel AUROC')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Pixel AUROC')
            ax3.set_title('Pixel-Level AUROC Over Time')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_ylim(0, 1)

    # Plot 4: Combined AUROC comparison
    if eval_history['epochs'] and eval_history['img_auroc'] and eval_history['px_auroc']:
        # Ensure all arrays have the same length
        min_len = min(len(eval_history['epochs']), len(eval_history['img_auroc']), len(eval_history['px_auroc']))
        if min_len > 0:
            epochs_plot = eval_history['epochs'][:min_len]
            img_auroc_plot = eval_history['img_auroc'][:min_len]
            px_auroc_plot = eval_history['px_auroc'][:min_len]
            ax4.plot(epochs_plot, img_auroc_plot, 'g-o', linewidth=2, markersize=6, label='Image AUROC')
            ax4.plot(epochs_plot, px_auroc_plot, 'r-s', linewidth=2, markersize=6, label='Pixel AUROC')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('AUROC Score')
            ax4.set_title('AUROC Comparison')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_ylim(0, 1)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, f'evaluation_metrics_{category_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Evaluation visualization saved to: {plot_path}")
    plt.close()


def save_training_summary(eval_history: Dict[str, List], loss_history: List[float],
                          save_dir: str, category_name: str, config_info: Dict[str, Any]) -> None:
    """Save comprehensive training summary to file."""
    summary = {
        "category": category_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config_info,
        "training_summary": {
            "total_epochs": len(loss_history),
            "final_loss": loss_history[-1] if loss_history else None,
            "best_loss": min(loss_history) if loss_history else None,
            "final_image_auroc": eval_history['img_auroc'][-1] if eval_history['img_auroc'] else None,
            "final_pixel_auroc": eval_history['px_auroc'][-1] if eval_history['px_auroc'] else None,
            "best_image_auroc": max(eval_history['img_auroc']) if eval_history['img_auroc'] else None,
            "best_pixel_auroc": max(eval_history['px_auroc']) if eval_history['px_auroc'] else None,
            "evaluation_epochs": eval_history['epochs'],
            "image_auroc_history": eval_history['img_auroc'],
            "pixel_auroc_history": eval_history['px_auroc'],
            "evaluation_count": len(eval_history['img_auroc']) if eval_history['img_auroc'] else 0
        }
    }

    summary_path = os.path.join(save_dir, f'training_summary_{category_name}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Training summary saved to: {summary_path}")


def create_realtime_visualization(eval_history: Dict[str, List], loss_history: List[float],
                                  save_dir: str, category_name: str, current_epoch: int) -> None:
    """Create real-time visualization that can be updated during training."""
    # Validate evaluation history
    eval_history = validate_eval_history(eval_history)

    plt.style.use('default')

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Real-time Training Progress - {category_name} (Epoch {current_epoch})',
                 fontsize=16, fontweight='bold')

    # Plot 1: Training Loss
    epochs = list(range(1, len(loss_history) + 1))
    ax1.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Image AUROC
    if eval_history['epochs'] and eval_history['img_auroc']:
        # Ensure arrays have the same length
        min_len = min(len(eval_history['epochs']), len(eval_history['img_auroc']))
        if min_len > 0:
            epochs_plot = eval_history['epochs'][:min_len]
            img_auroc_plot = eval_history['img_auroc'][:min_len]
            ax2.plot(epochs_plot, img_auroc_plot, 'g-o', linewidth=2, markersize=6, label='Image AUROC')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Image AUROC')
            ax2.set_title('Image-Level AUROC Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(0, 1)

    # Plot 3: Pixel AUROC
    if eval_history['epochs'] and eval_history['px_auroc']:
        # Ensure arrays have the same length
        min_len = min(len(eval_history['epochs']), len(eval_history['px_auroc']))
        if min_len > 0:
            epochs_plot = eval_history['epochs'][:min_len]
            px_auroc_plot = eval_history['px_auroc'][:min_len]
            ax3.plot(epochs_plot, px_auroc_plot, 'r-s', linewidth=2, markersize=6, label='Pixel AUROC')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Pixel AUROC')
            ax3.set_title('Pixel-Level AUROC Over Time')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_ylim(0, 1)

    # Plot 4: Combined AUROC comparison
    if eval_history['epochs'] and eval_history['img_auroc'] and eval_history['px_auroc']:
        # Ensure all arrays have the same length
        min_len = min(len(eval_history['epochs']), len(eval_history['img_auroc']), len(eval_history['px_auroc']))
        if min_len > 0:
            epochs_plot = eval_history['epochs'][:min_len]
            img_auroc_plot = eval_history['img_auroc'][:min_len]
            px_auroc_plot = eval_history['px_auroc'][:min_len]
            ax4.plot(epochs_plot, img_auroc_plot, 'g-o', linewidth=2, markersize=6, label='Image AUROC')
            ax4.plot(epochs_plot, px_auroc_plot, 'r-s', linewidth=2, markersize=6, label='Pixel AUROC')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('AUROC Score')
            ax4.set_title('AUROC Comparison')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_ylim(0, 1)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, f'realtime_training_progress_{category_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_evaluation_report(eval_history: Dict[str, List], loss_history: List[float],
                             save_dir: str, category_name: str) -> None:
    """Create a detailed evaluation report with statistics and analysis."""
    # Validate evaluation history
    eval_history = validate_eval_history(eval_history)

    report_path = os.path.join(save_dir, f'evaluation_report_{category_name}.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"EVALUATION REPORT - {category_name.upper()}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Training Summary
        f.write("TRAINING SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Epochs: {len(loss_history)}\n")
        f.write(f"Final Loss: {loss_history[-1]:.6f}\n")
        f.write(f"Best Loss: {min(loss_history):.6f}\n")
        f.write(f"Loss Improvement: {loss_history[0] - loss_history[-1]:.6f}\n")
        f.write(f"Average Loss: {np.mean(loss_history):.6f}\n")
        f.write(f"Loss Std Dev: {np.std(loss_history):.6f}\n\n")

        # Evaluation Summary
        if eval_history['img_auroc'] and eval_history['px_auroc']:
            # Ensure arrays have the same length for analysis
            min_len = min(len(eval_history['epochs']), len(eval_history['img_auroc']), len(eval_history['px_auroc']))
            if min_len > 0:
                epochs_analysis = eval_history['epochs'][:min_len]
                img_auroc_analysis = eval_history['img_auroc'][:min_len]
                px_auroc_analysis = eval_history['px_auroc'][:min_len]

                f.write("EVALUATION SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Number of Evaluations: {len(img_auroc_analysis)}\n")
                f.write(f"Evaluation Epochs: {epochs_analysis}\n\n")

                # Image AUROC Statistics
                f.write("IMAGE AUROC STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Final Image AUROC: {img_auroc_analysis[-1]:.4f}\n")
                f.write(f"Best Image AUROC: {max(img_auroc_analysis):.4f}\n")
                f.write(f"Average Image AUROC: {np.mean(img_auroc_analysis):.4f}\n")
                f.write(f"Image AUROC Std Dev: {np.std(img_auroc_analysis):.4f}\n")
                f.write(f"Image AUROC Improvement: {img_auroc_analysis[-1] - img_auroc_analysis[0]:.4f}\n\n")

                # Pixel AUROC Statistics
                f.write("PIXEL AUROC STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Final Pixel AUROC: {px_auroc_analysis[-1]:.4f}\n")
                f.write(f"Best Pixel AUROC: {max(px_auroc_analysis):.4f}\n")
                f.write(f"Average Pixel AUROC: {np.mean(px_auroc_analysis):.4f}\n")
                f.write(f"Pixel AUROC Std Dev: {np.std(px_auroc_analysis):.4f}\n")
                f.write(f"Pixel AUROC Improvement: {px_auroc_analysis[-1] - px_auroc_analysis[0]:.4f}\n\n")

                # Performance Analysis
                f.write("PERFORMANCE ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                best_img_epoch = epochs_analysis[np.argmax(img_auroc_analysis)]
                best_px_epoch = epochs_analysis[np.argmax(px_auroc_analysis)]
                f.write(f"Best Image AUROC achieved at epoch: {best_img_epoch}\n")
                f.write(f"Best Pixel AUROC achieved at epoch: {best_px_epoch}\n")

                # Convergence Analysis
                if len(img_auroc_analysis) > 5:
                    recent_img_auroc = img_auroc_analysis[-5:]
                    recent_px_auroc = px_auroc_analysis[-5:]
                    f.write(f"Recent Image AUROC trend: {recent_img_auroc}\n")
                    f.write(f"Recent Pixel AUROC trend: {recent_px_auroc}\n")

                    img_trend = "Improving" if recent_img_auroc[-1] > recent_img_auroc[0] else "Declining"
                    px_trend = "Improving" if recent_px_auroc[-1] > recent_px_auroc[0] else "Declining"
                    f.write(f"Image AUROC trend: {img_trend}\n")
                    f.write(f"Pixel AUROC trend: {px_trend}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"[INFO] Evaluation report saved to: {report_path}")

