from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class Food101Dataset(Dataset):
    """Food-101 PyTorch Dataset implementation."""

    def __init__(
        self, root: str, split: str = "train", transform: Optional[Callable] = None
    ):
        """Initialize Food-101 dataset.

        Args:
            root: Path to Food-101 dataset root directory
            split: Either 'train' or 'test'
            transform: Optional transform to apply to images
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Load class mapping
        with open(self.root / "meta" / "classes.txt") as f:
            self.classes = [line.strip() for line in f.readlines()]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load split data
        split_path = self.root / "meta" / f"{split}.json"
        with open(split_path) as f:
            self.samples = json.load(f)

        # Convert sample paths to (path, class_idx) tuples
        self.samples = [
            (
                self.root / "images" / f"{sample}.jpg",
                self.class_to_idx[sample.split("/")[0]],
            )
            for sample in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            tuple: (image, class_idx)
        """
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, class_idx
