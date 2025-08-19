import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import MvTecDataset


"""
Dataloader for MVTEC
"""
def load_mvtec_train_dataset(
        dataset_root_dir: str,
        category: str,
        image_size: int,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False
):
    train_dataset = MvTecDataset(
        dataset_root_dir=dataset_root_dir,
        category=category,
        image_size=image_size,
        is_train=True,
        is_mask=True,
        use_cutout=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return train_loader


def load_mvtec_test_dataset(
        dataset_root_dir: str,
        category: str,
        image_size: int,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = False
):
    test_dataset = MvTecDataset(
        dataset_root_dir=dataset_root_dir,
        category=category,
        image_size=image_size,
        is_train=False,
        is_mask=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return test_loader


def load_mvtec_only_good_test_dataset(
        dataset_root_dir: str,
        category: str,
        image_size: int,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = False
):
    test_dataset = MvTecDataset(
        dataset_root_dir=dataset_root_dir,
        category=category,
        image_size=image_size,
        is_train=False,
        is_mask=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return test_loader



""" Early stopping strategy """
class LossEarlyStopping:
    def __init__(self, patience: int, min_delta: float, smoothing_window: int, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.verbose = verbose
        self.best_loss = None
        self.early_stop = False
        self.loss_history = []
        self.wait = 0
        self.min_delta = min_delta

    def get_smoothed_loss(self, losses):
        if len(losses) < self.smoothing_window:
            return np.mean(losses)
        else:
            return np.mean(losses[-self.smoothing_window:])

    def __call__(self, current_loss: float):
        self.loss_history.append(current_loss)
        smoothed_loss = self.get_smoothed_loss(self.loss_history)

        if self.best_loss is None:
            self.best_loss = smoothed_loss
            if self.verbose:
                print(f"Early stopping baseline set: {smoothed_loss:.4f}")
        elif smoothed_loss > self.best_loss - self.min_delta:
            self.wait += 1
            if self.verbose:
                print(f"No improvement: {smoothed_loss:.4f} vs {self.best_loss:.4f}, patience: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best loss: {self.best_loss:.4f}")
        else:
            # Loss improvement detect
            self.best_loss = smoothed_loss
            self.wait = 0
            if self.verbose:
                print(f"Loss improved to {smoothed_loss:.4f}")
                print(100*"=")

        return self.early_stop


""" Optimizers select """
def get_optimizer(optimizer_name: str, params, **kwargs) -> torch.optim.Optimizer:
    if optimizer_name == "Adam":
        return torch.optim.Adam(params, **kwargs)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(params, **kwargs)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

