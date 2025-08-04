import os.path
import torch
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
            is_only_good_test: bool = False
    ):
        self.dataset_root_dir = dataset_root_dir
        self.category = category
        self.is_train = is_train
        self.image_size = image_size
        self.is_mask = is_mask
        self.is_only_good_test = is_only_good_test

        self.transform = transforms.Compose([
            # Resize image to image_size
            transforms.Resize((image_size, image_size)),
            # Scale to [0,1]
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

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

            if self.is_mask: # mask: bool
                if os.path.dirname(image_file).endswith("good"):
                    # Create tensor with full 0
                    mask = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 0
                else:
                    # if self.config.data.name == 'MVTec':
                    #     # Load ground truth
                    #     mask = Image.open(
                    #         image_file.replace('/test/',  '/ground_truth/')
                    #                   .replace(".png", "_mask.png")
                    #     )
                    # else:
                    #     mask = Image.open(image_file.replace("/test/", "/ground_truth/"))
                    mask = Image.open(
                        image_file.replace('/test/',  '/ground_truth/')
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

