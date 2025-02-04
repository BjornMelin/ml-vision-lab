"""Base dataset implementations for vision tasks."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets.

    Provides common functionality and enforces a standard interface
    for all dataset implementations.
    """

    def __init__(
        self, root: Union[str, Path], transforms: Optional[List] = None, **kwargs
    ):
        """Initialize dataset.

        Args:
            root: Root directory containing the dataset
            transforms: List of transforms to apply to data
            **kwargs: Additional dataset-specific arguments
        """
        self.root = Path(root)
        self.transforms = transforms
        self.data: List = []
        self.targets: List = []

        # Validate and setup
        self._validate_directory()
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self, **kwargs) -> None:
        """Setup the dataset.

        Implement dataset-specific initialization logic here.
        """
        pass

    def _validate_directory(self) -> None:
        """Validate dataset directory exists."""
        if not self.root.exists():
            raise ValueError(f"Dataset directory {self.root} does not exist")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single data item.

        Args:
            index: Index of the data item to get

        Returns:
            Dict containing the data item and its metadata
        """
        pass

    def apply_transforms(self, data: Any) -> Any:
        """Apply transformation pipeline to data.

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        if self.transforms:
            for t in self.transforms:
                data = t(data)
        return data


class VisionDataset(BaseDataset):
    """Base class for vision datasets.

    Implements common functionality for image-based datasets.
    """

    def __init__(
        self,
        root: Union[str, Path],
        transforms: Optional[List] = None,
        target_transforms: Optional[List] = None,
        **kwargs,
    ):
        """Initialize vision dataset.

        Args:
            root: Root directory containing the dataset
            transforms: List of transforms to apply to images
            target_transforms: List of transforms to apply to targets
            **kwargs: Additional dataset-specific arguments
        """
        self.target_transforms = target_transforms
        super().__init__(root, transforms, **kwargs)

    def apply_target_transforms(self, target: Any) -> Any:
        """Apply transformation pipeline to target.

        Args:
            target: Target to transform

        Returns:
            Transformed target
        """
        if self.target_transforms:
            for t in self.target_transforms:
                target = t(target)
        return target

    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image from path.

        Args:
            path: Path to image file

        Returns:
            Loaded image as numpy array
        """
        import cv2

        # Read image
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor.

        Args:
            data: Numpy array to convert

        Returns:
            PyTorch tensor
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(data)}")

        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0

        return torch.from_numpy(data)

    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics.

        Returns:
            Dict containing dataset statistics (e.g., mean, std)
        """
        # Calculate on a subset if dataset is large
        max_samples = min(len(self), 1000)
        indices = np.random.choice(len(self), max_samples, replace=False)

        # Collect samples
        samples = []
        for idx in indices:
            data = self.__getitem__(idx)
            if isinstance(data, dict):
                data = data["image"]
            samples.append(data)

        # Calculate statistics
        samples = np.stack(samples)
        return {
            "mean": samples.mean(axis=(0, 1, 2)).tolist(),
            "std": samples.std(axis=(0, 1, 2)).tolist(),
        }
