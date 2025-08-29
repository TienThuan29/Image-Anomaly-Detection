from .mvtec_dataset import MvTecDataset
from torch.utils.data import DataLoader

""" Dataloader for MVTEC """
def load_mvtec_train_dataset(
        dataset_root_dir: str,
        category: str,
        image_size: int,
        batch_size: int,
        num_workers: int = 2,
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
        num_workers: int = 2,
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

