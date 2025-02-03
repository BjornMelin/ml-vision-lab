"""Base transform implementations for vision tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class BaseTransform(ABC):
    """Abstract base class for all transforms.

    Provides a standard interface for data transformations.
    """

    def __init__(self, **kwargs):
        """Initialize transform.

        Args:
            **kwargs: Transform-specific parameters
        """
        self.params = kwargs

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply the transform to data.

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        pass

    def __repr__(self) -> str:
        """Get string representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class VisionTransform(BaseTransform):
    """Base class for vision transforms.

    Implements common functionality for image transformations.
    """

    def __init__(self, always_apply: bool = False, p: float = 0.5, **kwargs):
        """Initialize vision transform.

        Args:
            always_apply: Whether to always apply the transform
            p: Probability of applying the transform
            **kwargs: Additional transform-specific parameters
        """
        super().__init__(**kwargs)
        self.always_apply = always_apply
        self.p = p

    def __call__(
        self, data: Union[np.ndarray, Dict[str, Any]]
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Apply the transform to image data.

        Args:
            data: Image data to transform (numpy array or dict with 'image' key)

        Returns:
            Transformed image data
        """
        if isinstance(data, dict):
            if "image" not in data:
                raise KeyError("Dict input must contain 'image' key")
            should_apply = self.always_apply or np.random.random() < self.p
            if should_apply:
                data["image"] = self.apply(data["image"])
            return data
        else:
            should_apply = self.always_apply or np.random.random() < self.p
            if should_apply:
                return self.apply(data)
            return data

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the transform to an image.

        Args:
            image: Image to transform (HxWxC numpy array)

        Returns:
            Transformed image
        """
        pass


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: List[BaseTransform]):
        """Initialize transform composition.

        Args:
            transforms: List of transforms to apply sequentially
        """
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        """Apply all transforms sequentially.

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        """Get string representation."""
        format_string = self.__class__.__name__ + "(["
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n])"
        return format_string


def create_transform_pipeline(
    config: Dict[str, Any], transform_registry: Dict[str, type]
) -> Compose:
    """Create a transform pipeline from configuration.

    Args:
        config: Transform configuration dictionary
        transform_registry: Registry mapping transform names to classes

    Returns:
        Composed transform pipeline

    Example config:
        {
            'transforms': [
                {
                    'name': 'Resize',
                    'params': {'height': 224, 'width': 224}
                },
                {
                    'name': 'RandomHorizontalFlip',
                    'params': {'p': 0.5}
                }
            ]
        }
    """
    transforms = []

    for t_config in config["transforms"]:
        name = t_config["name"]
        if name not in transform_registry:
            raise ValueError(f"Unknown transform: {name}")

        transform_cls = transform_registry[name]
        params = t_config.get("params", {})
        transforms.append(transform_cls(**params))

    return Compose(transforms)
