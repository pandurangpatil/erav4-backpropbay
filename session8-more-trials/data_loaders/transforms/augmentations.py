"""
Albumentations-based data augmentation transforms.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class AlbumentationsTransforms:
    """
    Training phase transformations with Albumentations.
    Supports different augmentation strengths.
    """

    def __init__(self, mean, std, augmentation='strong'):
        """
        Initialize augmentation pipeline.

        Args:
            mean: Tuple of mean values for normalization
            std: Tuple of std values for normalization
            augmentation: Augmentation strength ('none', 'weak', 'strong')
        """
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        self.aug = self._build_pipeline()

    def _build_pipeline(self):
        """Build augmentation pipeline based on strength."""
        transforms = []

        if self.augmentation == 'strong':
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.5
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(8, 8),
                    hole_width_range=(8, 8),
                    fill=128,
                    p=0.5
                ),  # Cutout
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.3
                ),
            ])
        elif self.augmentation == 'weak':
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
                    p=0.3
                ),
            ])
        # 'none' augmentation: only normalize and convert to tensor

        # Always add normalization and tensor conversion
        transforms.extend([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

        return A.Compose(transforms)

    def __call__(self, image):
        """Apply augmentation pipeline to image."""
        image = np.array(image)
        return self.aug(image=image)["image"]


class TestTransformWrapper:
    """Test phase transformations (no augmentation, only normalization)."""

    def __init__(self, mean, std):
        """
        Initialize test transforms.

        Args:
            mean: Tuple of mean values for normalization
            std: Tuple of std values for normalization
        """
        self.aug = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def __call__(self, img):
        """Apply test transforms to image."""
        img = np.array(img)
        return self.aug(image=img)["image"]


def get_cifar_transforms(mean, std, train=True, augmentation='strong'):
    """
    Get CIFAR transforms based on train/test mode.

    Args:
        mean: Tuple of mean values for normalization
        std: Tuple of std values for normalization
        train: Whether to get train or test transforms
        augmentation: Augmentation strength for training ('none', 'weak', 'strong')

    Returns:
        Callable transform object
    """
    if train:
        return AlbumentationsTransforms(mean, std, augmentation)
    else:
        return TestTransformWrapper(mean, std)
