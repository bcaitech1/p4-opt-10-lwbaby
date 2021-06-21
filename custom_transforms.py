"""Image transformations for augmentation.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import random
from typing import Callable, Dict, Tuple

import PIL
from PIL.Image import Image
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
import torchvision.transforms.functional as F

from abc import ABC
from typing import List, Tuple

FILLCOLOR = (128, 128, 128)
FILLCOLOR_RGBA = (128, 128, 128, 128)

def get_rand_bbox_coord(
    w: int, h: int, len_ratio: float
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get a coordinate of random box."""
    size_hole_w = int(len_ratio * w)
    size_hole_h = int(len_ratio * h)
    x = random.randint(0, w)  # [0, w]
    y = random.randint(0, h)  # [0, h]

    x0 = max(0, x - size_hole_w // 2)
    y0 = max(0, y - size_hole_h // 2)
    x1 = min(w, x + size_hole_w // 2)
    y1 = min(h, y + size_hole_h // 2)
    return (x0, y0), (x1, y1)

def transforms_info() -> Dict[
    str, Tuple[Callable[[Image, float], Image], float, float]
]:
    """Return augmentation functions and their ranges."""
    transforms_list = [
        (Identity, 0.0, 0.0),
        (Invert, 0.0, 0.0),
        (Contrast, 0.0, 0.9),
        (AutoContrast, 0.0, 0.0),
        (Rotate, 0.0, 30.0),
        (TranslateX, 0.0, 150 / 331),
        (TranslateY, 0.0, 150 / 331),
        (Sharpness, 0.0, 0.9),
        (ShearX, 0.0, 0.3),
        (ShearY, 0.0, 0.3),
        (Color, 0.0, 0.9),
        (Brightness, 0.0, 0.9),
        (Equalize, 0.0, 0.0),
        (Solarize, 256.0, 0.0),
        (Posterize, 8, 4),
        (Cutout, 0, 0.5),
    ]
    return {f.__name__: (f, low, high) for f, low, high in transforms_list}


def Identity(img: Image, _: float) -> Image:
    """Identity map."""
    return img


def Invert(img: Image, _: float) -> Image:
    """Invert the image."""
    return PIL.ImageOps.invert(img)


def Contrast(img: Image, magnitude: float) -> Image:
    """Put contrast effect on the image."""
    return PIL.ImageEnhance.Contrast(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def AutoContrast(img: Image, _: float) -> Image:
    """Put contrast effect on the image."""
    return PIL.ImageOps.autocontrast(img)


def Rotate(img: Image, magnitude: float) -> Image:
    """Rotate the image (degree)."""
    rot = img.convert("RGBA").rotate(magnitude)
    return PIL.Image.composite(
        rot, PIL.Image.new("RGBA", rot.size, FILLCOLOR_RGBA), rot
    ).convert(img.mode)


def TranslateX(img: Image, magnitude: float) -> Image:
    """Translate the image on x-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
        fillcolor=FILLCOLOR,
    )


def TranslateY(img: Image, magnitude: float) -> Image:
    """Translate the image on y-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
        fillcolor=FILLCOLOR,
    )


def Sharpness(img: Image, magnitude: float) -> Image:
    """Adjust the sharpness of the image."""
    return PIL.ImageEnhance.Sharpness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def ShearX(img: Image, magnitude: float) -> Image:
    """Shear the image on x-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )


def ShearY(img: Image, magnitude: float) -> Image:
    """Shear the image on y-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )


def Color(img: Image, magnitude: float) -> Image:
    """Adjust the color balance of the image."""
    return PIL.ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))


def Brightness(img: Image, magnitude: float) -> Image:
    """Adjust brightness of the image."""
    return PIL.ImageEnhance.Brightness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def Equalize(img: Image, _: float) -> Image:
    """Equalize the image."""
    return PIL.ImageOps.equalize(img)


def Solarize(img: Image, magnitude: float) -> Image:
    """Solarize the image."""
    return PIL.ImageOps.solarize(img, magnitude)


def Posterize(img: Image, magnitude: float) -> Image:
    """Posterize the image."""
    magnitude = int(magnitude)
    return PIL.ImageOps.posterize(img, magnitude)


def Cutout(img: Image, magnitude: float) -> Image:
    """Cutout some region of the image."""
    if magnitude == 0.0:
        return img
    w, h = img.size
    xy = get_rand_bbox_coord(w, h, magnitude)

    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fill=FILLCOLOR)
    return img


class SquarePad:
    """Square pad to make torch resize to keep aspect ratio."""

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")

class Augmentation(ABC):
    """Abstract class used by all augmentation methods."""

    def __init__(self, n_level: int = 10) -> None:
        """Initialize."""
        self.transforms_info = transforms_info()
        self.n_level = n_level

    def _apply_augment(self, img: Image, name: str, level: int) -> Image:
        """Apply and get the augmented image.

        Args:
            img (Image): an image to augment
            level (int): magnitude of augmentation in [0, n_level]

        returns:
            Image: an augmented image
        """
        assert 0 <= level <= self.n_level
        augment_fn, low, high = self.transforms_info[name]
        return augment_fn(img.copy(), level * (high - low) / self.n_level + low)


class SequentialAugmentation(Augmentation):
    """Sequential augmentation class."""

    def __init__(
        self,
        policies: List[Tuple[str, float, int]],
        n_level: int = 10,
    ) -> None:
        """Initialize."""
        super().__init__(n_level)
        self.policies = policies

    def __call__(self, img: Image) -> Image:
        """Run augmentations."""
        for name, pr, level in self.policies:
            if random.random() > pr:
                continue
            img = self._apply_augment(img, name, level)
        return img


class RandAugmentation(Augmentation):
    """Random augmentation class.

    References:
        RandAugment: Practical automated data augmentation with a reduced search space
        (https://arxiv.org/abs/1909.13719)

    """
    def __init__(
        self,
        transforms: List[str],
        level: int = 14,
        n_level: int = 31,
    ) -> None:
        """Initialize."""
        super().__init__(n_level)
        self.level = level if isinstance(level, int) and 0 <= level <= n_level else None
        self.transforms = transforms

    def __call__(self, img: Image) -> Image:
        """Run augmentations."""
        random.shuffle(self.transforms)
        for transf in self.transforms:
            level = self.level if self.level else random.randint(0, self.n_level)
            img = self._apply_augment(img, transf, level)
        return img
