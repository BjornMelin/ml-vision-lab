import os
from typing import Any, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, root_dir: str, transform: Any = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self) -> None:
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(label)
