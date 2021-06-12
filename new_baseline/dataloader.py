import glob
import os
import yaml
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder, VisionDataset
import torchvision.transforms as transforms

class SquarePad:
    """Square pad to make torch resize to keep aspect ratio."""

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")

DATASET_NORMALIZE_INFO = {
    "CIFAR10": {"MEAN": (0.4914, 0.4822, 0.4465), "STD": (0.2470, 0.2435, 0.2616)},
    "CIFAR100": {"MEAN": (0.5071, 0.4865, 0.4409), "STD": (0.2673, 0.2564, 0.2762)},
    "IMAGENET": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
    "TACO": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}

def simple_augment_train(
    dataset: str = "CIFAR10", img_size: float = 32
) -> transforms.Compose:
    """Simple data augmentation rule for training CIFAR100."""
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))),
            transforms.RandomResizedCrop(
                size=img_size, ratio=(0.75, 1.0, 1.3333333333333333)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )


def simple_augment_test(
    dataset: str = "CIFAR10", img_size: float = 32
) -> transforms.Compose:
    """Simple data augmentation rule for testing CIFAR100."""
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )


def create_dataloader(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Simple dataloader.

    Args:
        cfg: yaml file path or dictionary type of the data.

    Returns:
        train_loader
        valid_loader
        test_loader
    """
    # Data Setup
    train_dataset, val_dataset, test_dataset = get_dataset(
        data_path=config["DATA_PATH"],
        dataset_name=config["DATASET"],
        img_size=config["IMG_SIZE"],
    )

    return get_dataloader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config["BATCH_SIZE"],
    )


def get_dataset(
    data_path: str = "./save/data",
    dataset_name: str = "CIFAR10",
    img_size: float = 32,
    val_ratio: float=0.2,
) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    # preprocessing policies
    transform_train = simple_augment_train(dataset=dataset_name, img_size=img_size)
    transform_test = simple_augment_test(dataset=dataset_name, img_size=img_size)

    # pytorch dataset
    if dataset_name == "TACO":
        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        test_path = os.path.join(data_path, "test")

        train_dataset = ImageFolder(root=train_path, transform=transform_train)
        val_dataset = ImageFolder(root=val_path, transform=transform_test)
        test_dataset = ImageFolder(root=test_path, transform=transform_test)

    else:
        Dataset = getattr(
            __import__("torchvision.datasets", fromlist=[""]), dataset_name
        )
        train_dataset = Dataset(
            root=data_path, train=True, download=True, transform=transform_train
        )
        # from train dataset, train: 80%, val: 20%
        train_length = int(len(train_dataset) * (1.0-val_ratio))
        train_dataset, val_dataset = random_split(
            train_dataset, [train_length, len(train_dataset) - train_length]
        )
        test_dataset = Dataset(
            root=data_path, train=False, download=False, transform=transform_test
        )

    return train_dataset, val_dataset, test_dataset


def get_dataloader(
    train_dataset: VisionDataset,
    val_dataset: VisionDataset,
    test_dataset: VisionDataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get dataloader for training and testing."""

    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=val_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=4
    )
    return train_loader, valid_loader, test_loader
