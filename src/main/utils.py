import numpy as np
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
        use_cutout=True
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


"""
Early stopping strategy
"""
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


"""
Config load from config.yml
"""
import yaml
import os
from typing import Dict, Any, Optional

class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = 'config.yml'

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        path = config_path or self.config_path

        if not path:
            raise ValueError("No config path provided")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
                return self.config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")

    # Get key
    def get(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            raise ValueError("Config not loaded. Call load_config() first.")

        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    # Get all section
    def get_section(self, section: str) -> Dict[str, Any]:
        if self.config is None:
            raise ValueError("Config not loaded. Call load_config() first.")

        return self.config.get(section, {})

    def update(self, key: str, value: Any) -> None:
        if self.config is None:
            raise ValueError("Config not loaded. Call load_config() first.")

        keys = key.split('.')
        config_ref = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]

        # Set the value
        config_ref[keys[-1]] = value

    def save_config(self, output_path: Optional[str] = None) -> None:
        if self.config is None:
            raise ValueError("Config not loaded. Call load_config() first.")

        path = output_path or self.config_path
        if not path:
            raise ValueError("No output path provided")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
