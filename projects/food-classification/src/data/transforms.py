from typing import Dict, Any, List

import torch
import torchvision.transforms as T


def build_transforms(cfg: Dict[str, Any]) -> Dict[str, torch.nn.Module]:
    """Build training and validation transforms from config.

    Args:
        cfg: Transform configuration dictionary

    Returns:
        dict: Contains 'train' and 'test' transform pipelines
    """
    train_transforms = []
    test_transforms = []

    # Build training transforms
    for transform in cfg["train"]:
        for name, params in transform.items():
            transform_fn = getattr(T, name)
            train_transforms.append(transform_fn(**params))

    # Build test transforms
    for transform in cfg["test"]:
        for name, params in transform.items():
            transform_fn = getattr(T, name)
            test_transforms.append(transform_fn(**params))

    return {"train": T.Compose(train_transforms), "test": T.Compose(test_transforms)}


def get_default_transforms() -> Dict[str, torch.nn.Module]:
    """Get default transforms for Food-101 dataset.

    Returns:
        dict: Contains 'train' and 'test' transform pipelines
    """
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return {"train": train_transform, "test": test_transform}
