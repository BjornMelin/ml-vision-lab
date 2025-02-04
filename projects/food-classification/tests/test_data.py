import pytest
from src.data.dataset import ImageDataset

def test_dataset_loading():
    dataset = ImageDataset("data/test")
    assert len(dataset) > 0
    
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
