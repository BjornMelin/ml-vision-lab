import pytest
import torch
from src.models.model import Model

def test_model_forward(sample_batch):
    model = Model(num_classes=10)
    outputs = model(sample_batch['input'])
    assert outputs.shape == (4, 10)
