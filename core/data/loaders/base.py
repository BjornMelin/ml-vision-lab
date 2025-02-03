"""Base data loader implementations for vision tasks."""

from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate


class BaseDataLoaderConfig:
    """Configuration for data loaders."""

    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        **kwargs
    ):
        """Initialize loader configuration.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory in GPU training
            drop_last: Whether to drop last incomplete batch
            prefetch_factor: Number of batches to prefetch
            persistent_workers: Whether to maintain worker processes
            **kwargs: Additional configuration parameters
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.additional_params = kwargs


class BaseDataLoader(DataLoader):
    """Base class for all data loaders.

    Extends PyTorch's DataLoader with additional functionality.
    """

    def __init__(
        self,
        dataset: Dataset,
        config: Optional[BaseDataLoaderConfig] = None,
        sampler: Optional[Sampler] = None,
        collate_fn: Optional[callable] = None,
        **kwargs
    ):
        """Initialize data loader.

        Args:
            dataset: Dataset to load from
            config: Loader configuration
            sampler: Data sampler
            collate_fn: Function to collate samples
            **kwargs: Additional arguments passed to DataLoader
        """
        if config is None:
            config = BaseDataLoaderConfig()

        if collate_fn is None:
            collate_fn = self.default_collate

        super().__init__(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
            **kwargs
        )

        self.config = config

    @staticmethod
    def default_collate(batch: list) -> Any:
        """Default collate function.

        Handles both tensor data and dictionaries of tensors.

        Args:
            batch: List of samples to collate

        Returns:
            Collated batch
        """
        if isinstance(batch[0], dict):
            return {
                key: default_collate([d[key] for d in batch]) for key in batch[0].keys()
            }
        return default_collate(batch)

    def get_statistics(self) -> Dict[str, float]:
        """Get data loader statistics.

        Returns:
            Dict containing statistics about the data
        """
        stats = {}

        # Basic stats
        stats["num_samples"] = len(self.dataset)
        stats["num_batches"] = len(self)
        stats["batch_size"] = self.batch_size

        # Get dataset stats if available
        if hasattr(self.dataset, "get_statistics"):
            stats.update(self.dataset.get_statistics())

        return stats


def create_data_loader(dataset: Dataset, config: Dict[str, Any]) -> BaseDataLoader:
    """Create a data loader from configuration.

    Args:
        dataset: Dataset to load from
        config: Configuration dictionary

    Returns:
        Configured data loader

    Example config:
        {
            'batch_size': 32,
            'num_workers': 4,
            'shuffle': True,
            'pin_memory': True
        }
    """
    loader_config = BaseDataLoaderConfig(**config)
    return BaseDataLoader(dataset, config=loader_config)
