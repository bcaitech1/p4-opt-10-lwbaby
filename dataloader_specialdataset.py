"""Tune Model.

- Author: Junghoon Kim, Jongkuk Lim, Jimyeong Kim
- Contact: placidus36@gmail.com, lim.jeikei@gmail.com, wlaud1001@snu.ac.kr
- Reference
    https://github.com/j-marple-dev/model_compression
"""
import glob
import os
from typing import Any, Dict, List, Tuple, Union
from alist import alist
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision.datasets import ImageFolder, VisionDataset
import torchvision
import yaml
from torchvision.datasets.folder import default_loader

from src.utils.data import weights_for_balanced_classes
from src.utils.torch_utils import split_dataset_index
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy



class MySpecialDataset(ImageFolder):  
    def __init__(self, root, transform, t2, loader=default_loader,
                 batch_size=128, is_valid_file=None):    
        super(MySpecialDataset, self).__init__(root=root, 
                                               loader=loader, 
                                               is_valid_file=is_valid_file)
        
        self.transform = transform
        self.t2 = t2;
        
    def __getitem__(self, index):
        image_path, target = self.samples[index]
        # do your magic here
        im = cv2.imread(image_path)
        image = Image.open(image_path).convert('RGB') 
        t2im = self.t2(image)
        im = t2im
        for i in range(3):
            im[i] = im[i] - torch.min(im[i]) 
            im[i] = 255*im[i]/(torch.max(im[i])-torch.min(im[i]))
        im = torch.as_tensor(t2im, dtype=torch.uint8)

        im = im.permute(1, 2, 0)
        im = im.numpy()
        augmented = self.transform(image=im) 
        im = augmented['image']

        im = im.float()

        
        return im, target

    
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
        val_ratio=config["VAL_RATIO"],
        transform_train=config["AUG_TRAIN"],
        transform_test=config["AUG_TEST"],
        transform_train_params=config["AUG_TRAIN_PARAMS"],
        transform_test_params=config.get("AUG_TEST_PARAMS"),
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
    transform_train: str = "simple_augment_train",
    transform_test: str = "simple_augment_test",
    transform_train_params: Dict[str, int] = None,
    transform_test_params: Dict[str, int] = None,
) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    """Get dataset for training and testing."""
    if not transform_train_params:
        transform_train_params = dict()
    if not transform_test_params:
        transform_test_params = dict()

    # preprocessing policies
    transform_train = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_train,
    )(dataset=dataset_name, img_size=img_size, **transform_train_params)
    
    transform_train_albu = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        'albumentation_train',
    )(dataset=dataset_name, img_size=img_size)

    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_test,
    )(dataset=dataset_name, img_size=img_size, **transform_test_params)

    transform_albu = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        "randaugment_albu",
    )(dataset=dataset_name, img_size=img_size, **transform_test_params)
    label_weights = None
    # pytorch dataset
    if dataset_name == "TACO":
        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        test_path = os.path.join(data_path, "test")

        train_dataset_1 = ImageFolder(root=train_path, transform=transform_train)
        train_dataset_2 = ImageFolder(root=train_path, transform=transform_train)
        train_dataset_3 = ImageFolder(root=train_path, transform=transform_train)
        train_dataset_4 = MySpecialDataset(train_path, transform = transform_train_albu,t2=transform_albu)
        train_dataset_5 = MySpecialDataset(train_path, transform = transform_train_albu,t2=transform_albu)
        train_dataset_6 = MySpecialDataset(train_path, transform = transform_train_albu,t2=transform_albu)
#         alist = []
#         print("Start make dataset")
#         print(len(train_dataset_4))
#         for i, (data, label) in enumerate(train_dataset_4):
#             if(i%500 == 0): print(i)
#             if label in [0,1,2,3,5,8]: alist.append(i)
#         print(alist)
        train_dataset_2 = Subset(train_dataset_2, alist)
        train_dataset_3 = Subset(train_dataset_3, alist)
        train_dataset_5 = Subset(train_dataset_5, alist)
        train_dataset_6 = Subset(train_dataset_6, alist)
        print(len(train_dataset_1))
        print(len(train_dataset_2))
        train_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4, train_dataset_5, train_dataset_6])
        val_dataset = ImageFolder(root=val_path, transform=transform_test)
        test_dataset = ImageFolder(root=test_path, transform=transform_test)
        print(len(train_dataset))
        print("trainset = " + str(type(train_dataset)))
        print("validationset = " + str(type(val_dataset)))

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
        num_workers=10,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=val_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=5
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=5
    )
    return train_loader, valid_loader, test_loader
