import os.path
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob


class MvTecDataset(Dataset):
    def __init__(
            self,
            dataset_root_dir: str,
            category: str,
            image_size: int,
            is_mask: bool,
            is_train: bool,
            is_only_good_test: bool = False,
            # Cutout parameters
            use_cutout: bool = False,
            cutout_n_holes: int = 1,
            cutout_length: int = 16, # 24: cutout màu đen bị học luôn!!!
            cutout_probability: float = 0.3
    ):
        self.dataset_root_dir = dataset_root_dir
        self.category = category
        self.is_train = is_train
        self.image_size = image_size
        self.is_mask = is_mask
        self.is_only_good_test = is_only_good_test
        self.use_cutout = use_cutout

        # Base transforms
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
            if self.is_only_good_test:
                label = 0
                if self.is_mask:
                    mask = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    return {'image': image, 'mask': mask, 'label': label}
                else:
                    return {'image': image, 'label': label}

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


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        probability (float): Probability of applying cutout. Default is 1.0.
    """

    def __init__(self, n_holes=1, length=16, probability=1.0):
        self.n_holes = n_holes
        self.length = length
        self.probability = probability

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if random.random() > self.probability:
            return img

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            # Choose random center point
            y = np.random.randint(h)
            x = np.random.randint(w)

            # Calculate patch boundaries
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img