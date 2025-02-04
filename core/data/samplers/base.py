"""Base sampler implementations for balanced and custom sampling strategies."""

from typing import Iterator, List, Optional
import numpy as np
from torch.utils.data import Sampler
from collections import Counter


class BaseSampler(Sampler):
    """Abstract base class for all samplers."""

    def __init__(self, data_source):
        """Initialize sampler.

        Args:
            data_source: Dataset to sample from
        """
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        """Get iterator over indices."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.data_source)


class BalancedSampler(BaseSampler):
    """Sampler that ensures balanced sampling across classes.

    Useful for handling imbalanced datasets by oversampling
    minority classes and/or undersampling majority classes.
    """

    def __init__(
        self,
        data_source,
        labels: List[int],
        replacement: bool = False,
        num_samples: Optional[int] = None,
        class_weights: Optional[List[float]] = None,
    ):
        """Initialize balanced sampler.

        Args:
            data_source: Dataset to sample from
            labels: List of class labels for each sample
            replacement: Whether to sample with replacement
            num_samples: Number of samples to draw (default: len(data_source))
            class_weights: Optional weights for each class
        """
        super().__init__(data_source)

        if len(labels) != len(data_source):
            raise ValueError(
                f"Length of labels ({len(labels)}) does not match "
                f"length of dataset ({len(data_source)})"
            )

        self.labels = labels
        self.replacement = replacement
        self.num_samples = num_samples or len(data_source)

        # Get class distribution
        self.class_counts = Counter(labels)
        self.classes = sorted(self.class_counts.keys())

        # Set up class weights
        if class_weights is None:
            # Default to inverse frequency weighting
            max_count = max(self.class_counts.values())
            self.class_weights = {
                cls: max_count / count for cls, count in self.class_counts.items()
            }
        else:
            if len(class_weights) != len(self.classes):
                raise ValueError(
                    f"Length of class_weights ({len(class_weights)}) does not match "
                    f"number of classes ({len(self.classes)})"
                )
            self.class_weights = dict(zip(self.classes, class_weights))

        # Create index map for each class
        self.class_indices = {
            cls: np.where(np.array(labels) == cls)[0] for cls in self.classes
        }

    def __iter__(self) -> Iterator[int]:
        """Get iterator over indices.

        Returns indices that ensure balanced sampling across classes.
        """
        # Calculate number of samples per class
        if self.replacement:
            # With replacement: can oversample minority classes
            samples_per_class = {
                cls: int(
                    self.num_samples
                    * (self.class_weights[cls] / sum(self.class_weights.values()))
                )
                for cls in self.classes
            }
        else:
            # Without replacement: limited by size of smallest class
            min_class_size = min(
                len(indices) for indices in self.class_indices.values()
            )
            samples_per_class = {cls: min_class_size for cls in self.classes}

        # Generate indices
        indices = []
        for cls in self.classes:
            class_indices = self.class_indices[cls]
            if self.replacement:
                sampled_indices = np.random.choice(
                    class_indices, size=samples_per_class[cls], replace=True
                )
            else:
                sampled_indices = np.random.choice(
                    class_indices, size=samples_per_class[cls], replace=False
                )
            indices.extend(sampled_indices)

        # Shuffle final indices
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Get number of samples to draw."""
        return self.num_samples


class SubsetSampler(BaseSampler):
    """Sampler that samples from a subset of indices."""

    def __init__(self, data_source, indices: List[int]):
        """Initialize subset sampler.

        Args:
            data_source: Dataset to sample from
            indices: List of indices to sample from
        """
        super().__init__(data_source)
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        """Get iterator over subset indices."""
        return iter(self.indices)

    def __len__(self) -> int:
        """Get number of samples in subset."""
        return len(self.indices)
