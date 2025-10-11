"""
Data loading and augmentation for CIFAR-100
Supports both standard PyTorch transforms and Albumentations
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available. Using standard transforms.")


# CIFAR-100 dataset statistics
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)


class AlbumentationsTransforms:
    """Albumentations-based transforms wrapper for CIFAR-100"""

    def __init__(self, mean=CIFAR100_MEAN, std=CIFAR100_STD, is_training=True):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required for this transform")

        if is_training:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                                 rotate_limit=15, p=0.5),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8,
                               p=0.5, fill_value=0),  # Cutout
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                          contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                    val_shift_limit=10, p=0.3),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

    def __call__(self, img):
        image = np.array(img)
        return self.aug(image=image)["image"]


def get_train_transforms(use_albumentations=True):
    """
    Get training data transforms

    Args:
        use_albumentations: Whether to use Albumentations (if available)

    Returns:
        Transform composition
    """
    if use_albumentations and ALBUMENTATIONS_AVAILABLE:
        return AlbumentationsTransforms(mean=CIFAR100_MEAN, std=CIFAR100_STD,
                                       is_training=True)
    else:
        # Fallback to standard PyTorch transforms
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        ])


def get_test_transforms():
    """Get test/validation data transforms"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    ])


def get_dataloaders(data_dir='./data', batch_size=256, num_workers=4,
                   use_albumentations=True, pin_memory=True):
    """
    Create CIFAR-100 data loaders

    Args:
        data_dir: Directory to store/load CIFAR-100 data
        batch_size: Batch size for training and testing
        num_workers: Number of data loading workers
        use_albumentations: Use Albumentations for augmentation
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, test_loader, num_classes)
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Get transforms
    train_transforms = get_train_transforms(use_albumentations)
    test_transforms = get_test_transforms()

    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transforms
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transforms
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for consistent training
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader, 100  # CIFAR-100 has 100 classes


def get_cifar100_classes():
    """Get CIFAR-100 class names"""
    return datasets.CIFAR100(root='./data', train=False, download=True).classes


class CIFAR100Dataset:
    """
    Convenience class for CIFAR-100 dataset management
    Useful for SageMaker integration
    """

    def __init__(self, data_dir='./data', batch_size=256, num_workers=4,
                 use_albumentations=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_albumentations = use_albumentations
        self.num_classes = 100
        self.mean = CIFAR100_MEAN
        self.std = CIFAR100_STD

    def get_loaders(self, pin_memory=True):
        """Get train and test data loaders"""
        return get_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            use_albumentations=self.use_albumentations,
            pin_memory=pin_memory
        )

    def get_class_names(self):
        """Get class names"""
        return get_cifar100_classes()


if __name__ == "__main__":
    # Test data loading
    print("Testing CIFAR-100 data loading...")

    train_loader, test_loader, num_classes = get_dataloaders(
        batch_size=32,
        num_workers=2
    )

    print(f"Number of classes: {num_classes}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test batch loading
    data, target = next(iter(train_loader))
    print(f"Batch data shape: {data.shape}")
    print(f"Batch target shape: {target.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")

    # Show class names
    classes = get_cifar100_classes()
    print(f"First 10 classes: {classes[:10]}")
