"""
CIFAR-100 dataset with Albumentations transforms.
"""
from torchvision import datasets
from .base import BaseDataset
from .transforms import get_cifar_transforms


class CIFAR100Dataset(BaseDataset):
    """
    CIFAR-100 dataset wrapper with Albumentations augmentation.

    Attributes:
        MEAN: Normalization mean values
        STD: Normalization std values
        NUM_CLASSES: Number of classes in CIFAR-100
        IMAGE_SIZE: Image dimensions (height, width)
    """

    # CIFAR-100 dataset statistics
    MEAN = (0.5071, 0.4865, 0.4409)
    STD = (0.2673, 0.2564, 0.2761)
    NUM_CLASSES = 100
    IMAGE_SIZE = (32, 32)

    def __init__(self, train=True, data_dir='../data', augmentation='strong', download=True):
        """
        Initialize CIFAR-100 dataset.

        Args:
            train: Whether to load train or test split
            data_dir: Directory to store/load dataset
            augmentation: Augmentation strength ('none', 'weak', 'strong')
                         Only applies to training data
            download: Whether to download dataset if not present
        """
        self.train = train
        self.data_dir = data_dir
        self.augmentation = augmentation if train else 'none'
        self.download = download

        # Get appropriate transforms
        self.transform = self.get_transforms(self.augmentation)

        # Load CIFAR-100 dataset
        self.dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=self.train,
            download=self.download,
            transform=self.transform
        )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            tuple: (image_tensor, label)
        """
        return self.dataset[idx]

    def get_transforms(self, augmentation='strong'):
        """
        Get transforms for this dataset.

        Args:
            augmentation: Augmentation strength ('none', 'weak', 'strong')

        Returns:
            Callable transform
        """
        return get_cifar_transforms(
            mean=self.MEAN,
            std=self.STD,
            train=self.train,
            augmentation=augmentation
        )

    @classmethod
    def get_info(cls):
        """
        Get dataset metadata.

        Returns:
            dict: Metadata including num_classes, image_size, mean, std, etc.
        """
        return {
            'name': 'CIFAR-100',
            'num_classes': cls.NUM_CLASSES,
            'image_size': cls.IMAGE_SIZE,
            'mean': cls.MEAN,
            'std': cls.STD,
            'train_samples': 50000,
            'test_samples': 10000,
            'description': '100-class image classification dataset with 600 images per class'
        }
