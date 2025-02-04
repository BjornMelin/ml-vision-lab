import pytest
import torch

@pytest.fixture
def sample_batch():
    return {
        'input': torch.randn(4, 3, 224, 224),
        'target': torch.randint(0, 10, (4,))
    }
