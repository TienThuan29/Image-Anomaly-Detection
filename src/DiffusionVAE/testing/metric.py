import os
import torch
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC


def normalize_maps_global(anomaly_maps: List[torch.Tensor]) -> List[torch.Tensor]:
    stacked = torch.cat(anomaly_maps, dim=0)
    amin = stacked.min()
    amax = stacked.max()
    if (amax - amin) < 1e-8:
        return [torch.zeros_like(m) for m in anomaly_maps]
    return [(m - amin) / (amax - amin) for m in anomaly_maps]

def compute_anomaly_map(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return (x - x_recon).abs().mean(dim=1) # Tính mean giữa 3 channels

def eval_auroc_image(labels: List[int], scores: List[float]) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))

def eval_auroc_pixel(anomaly_maps: List[torch.Tensor], gt_masks: List[torch.Tensor]) -> float:
    norm_maps = normalize_maps_global(anomaly_maps)
    pred = torch.cat([m.flatten() for m in norm_maps], dim=0).cpu()
    gt = torch.cat([g.flatten() for g in gt_masks], dim=0).cpu().bool()
    metric = AUROC(task="binary")
    return float(metric(pred, gt))

def calc_image_score(anomaly_maps, image_score_type_name):
    if isinstance(anomaly_maps, list):
        # Check if list is empty
        if len(anomaly_maps) == 0:
            raise ValueError("anomaly_maps list is empty")
        
        # Check if all tensors have the same shape
        first_shape = anomaly_maps[0].shape
        for i, tensor in enumerate(anomaly_maps):
            if tensor.shape != first_shape:
                raise ValueError(f"Tensor {i} has shape {tensor.shape}, expected {first_shape}")
        
        # Stack individual maps into batch format [N, H, W]
        anomaly_map = torch.stack(anomaly_maps, dim=0)
    else:
        anomaly_map = anomaly_maps
    
    if image_score_type_name == 'max':
        return anomaly_map.amax(dim=(1, 2))
    elif image_score_type_name == 'mean':
        return anomaly_map.float().mean(dim=(1, 2))
    elif image_score_type_name == 'std':
        return anomaly_map.float().std(dim=(1, 2), unbiased=False)
    else:
        raise ValueError(f"Unsupported score type: {image_score_type_name}")