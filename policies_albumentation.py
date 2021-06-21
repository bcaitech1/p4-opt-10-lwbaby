"""PyTorch transforms for data augmentation.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import torchvision.transforms as transforms
from src.augmentation.methods import RandAugmentation, SequentialAugmentation
from src.augmentation.transforms import FILLCOLOR, SquarePad
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
    OneOf, Compose,HorizontalFlip,
    Flip, ShiftScaleRotate, GridDistortion, ElasticTransform,
    RandomGamma, RandomContrast, RandomBrightness, RandomBrightnessContrast,
    Blur, MedianBlur, MotionBlur, GaussianBlur,
    CLAHE, IAASharpen, GaussNoise,CoarseDropout,MultiplicativeNoise,
    RandomSizedCrop, CropNonEmptyMaskIfExists,
    RandomSunFlare, Resize, Normalize,
    HueSaturationValue, RGBShift)
import cv2

DATASET_NORMALIZE_INFO = {
    "CIFAR10": {"MEAN": (0.4914, 0.4822, 0.4465), "STD": (0.2470, 0.2435, 0.2616)},
    "CIFAR100": {"MEAN": (0.5071, 0.4865, 0.4409), "STD": (0.2673, 0.2564, 0.2762)},
    "IMAGENET": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
    "TACO": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}

def albumentation_train(
    dataset: str = "TACO", img_size: float = 256
) -> transforms.Compose:
    """Simple data augmentation rule for training CIFAR100."""
    aug = Compose([
        Resize(height = img_size, width = img_size, interpolation=cv2.INTER_AREA),
        #Blur(),
        CLAHE(),
        OneOf([RandomGamma(p=1),], p=0.5),
        RandomBrightnessContrast(p=0.5, brightness_by_max=False),
        OneOf([ShiftScaleRotate(shift_limit=0.1,
                                scale_limit=0.1,
                                rotate_limit=10,
                                p=0.25,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(0,0,0)),], 
                p=0.25),
        OneOf([
            CoarseDropout(max_holes=3, max_height=7, max_width=7, fill_value=0, p=1),
            CoarseDropout(max_holes=6, max_height=5, max_width=5, fill_value=0, p=1),
            CoarseDropout(max_holes=9, max_height=3, max_width=3, fill_value=0, p=1),
        ], p=0.2),
        
        Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
            ),
        ToTensorV2()
    ], p=1)
    return aug

def simple_augment_train(
    dataset: str = "CIFAR10", img_size: float = 32
) -> transforms.Compose:
    """Simple data augmentation rule for training CIFAR100."""
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size * 1.2, img_size * 1.2)),
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


def randaugment_train(
    dataset: str = "CIFAR10",
    img_size: float = 32,
    n_select: int = 2,
    level: int = 14,
    n_level: int = 31,
) -> transforms.Compose:
    """Random augmentation policy for training CIFAR100."""
    operators = [
        "Identity",
        "AutoContrast",
        "Equalize",
        "Rotate",
        "Solarize",
        "Color",
        "Posterize",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            RandAugmentation(operators, n_select, level, n_level),
            transforms.RandomHorizontalFlip(),
            SequentialAugmentation([("Cutout", 0.8, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )

def randaugment_albu(
    dataset: str = "TACO",
    img_size: float = 256,
    n_select: int = 2,
    level: int = 14,
    n_level: int = 31,
) -> transforms.Compose:
    """Random augmentation policy for training CIFAR100."""
    operators = [
        "Identity",
        "AutoContrast",
        "Equalize",
        "Rotate",
        "Solarize",
        "Color",
        "Posterize",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            RandAugmentation(operators, n_select, level, n_level),
            transforms.RandomHorizontalFlip(),
            SequentialAugmentation([("Cutout", 0.8, 5)]),
            transforms.ToTensor(),
        ]
    )
