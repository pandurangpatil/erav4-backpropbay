"""
Base dataset interface for all datasets.
"""
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """
    Abstract base class for all datasets.
    All dataset implementations should inherit from this class.
    """

    @abstractmethod
    def __init__(self, train=True, data_dir='../data', augmentation='strong', **kwargs):
        """Initialize the dataset."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Get a sample by index."""
        pass

    @classmethod
    @abstractmethod
    def get_info(cls):
        """
        Get dataset metadata.

        Returns:
            dict: Metadata including num_classes, image_size, mean, std, etc.
        """
        pass

    @abstractmethod
    def get_transforms(self, augmentation='strong'):
        """
        Get transforms for this dataset.

        Args:
            augmentation: Augmentation strength ('none', 'weak', 'strong')

        Returns:
            Callable transform
        """
        pass
