import os.path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from .augmentation import Cutout

class MvTecDataset(Dataset):
    def __init__(
            self,
            dataset_root_dir: str,
            category: str,
            image_size: int,
            is_mask: bool,
            is_train: bool,
            # Cutout parameters
            use_cutout: bool = False,
            cutout_n_holes: int = 1,
            cutout_length: int = 16, # 24: cutout màu đen bị học
            cutout_probability: float = 0.3
    ):
        self.dataset_root_dir = dataset_root_dir
        self.category = category
        self.is_train = is_train
        self.image_size = image_size
        self.is_mask = is_mask
        self.use_cutout = use_cutout

        base_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        # Add cutout for training data if requested
        if is_train and use_cutout:
            base_transforms.append(
                Cutout(
                    n_holes=cutout_n_holes,
                    length=cutout_length,
                    probability=cutout_probability
                )
            )

        self.transform = transforms.Compose(base_transforms)

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        if is_train:
            self.image_files = glob(
                os.path.join(self.dataset_root_dir, category, 'train', 'good', '*.png')
            )
        else:
            self.image_files = glob(os.path.join(self.dataset_root_dir, category, "test", "*", "*.png"))

    def __getitem__(self, index: int):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.transform(image)

        if (image.shape[0] == 1):
            # If this is a grayscale image, expand channel 1 -> 3
            image = image.expand(3, self.image_size, self.image_size)

        if self.is_train:
            label = 0
            return {'image': image, 'label': label}
        else:
            if self.is_mask:  # mask: bool
                if os.path.dirname(image_file).endswith("good"):
                    # Create tensor with full 0
                    mask = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 0
                else:
                    mask = Image.open(
                        image_file.replace('/test/', '/ground_truth/')
                        .replace(".png", "_mask.png")
                    )
                    mask = self.mask_transform(mask)
                    label = 1
                return {'image': image, 'mask': mask, 'label': label}

            else:
                if os.path.dirname(image_file).endswith("good"):
                    label = 0
                else:
                    label = 1
                return {'image': image, 'label': label}

    def __len__(self):
        return len(self.image_files)




