# Contributing to ML Vision Lab

Thank you for your interest in contributing to ML Vision Lab! This document provides guidelines for contributing ML/CV projects following industry best practices.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [ML Project Structure](#ml-project-structure)
- [ML Development Standards](#ml-development-standards)
- [Project Requirements](#project-requirements)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Development Setup

1. **Fork and Clone**

   ```bash
   git clone https://github.com/YOUR-USERNAME/ml-vision-lab.git
   cd ml-vision-lab
   ```

2. **Set Up Environment**

   Using Poetry (recommended):

   ```bash
   # Install Poetry
   pip install poetry

   # Install dependencies
   poetry install --with dev,ui

   # Activate virtual environment
   poetry shell
   ```

   Using pip:

   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/MacOS
   # or
   .venv\Scripts\activate     # Windows

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install Development Tools**

   ```bash
   # Install pre-commit hooks
   pre-commit install

   # Set up DVC
   dvc init
   dvc remote add -d storage s3://your-bucket/path

   # Configure MLflow
   export MLFLOW_TRACKING_URI=http://localhost:5000
   ```

## ML Project Structure

When adding a new ML project, follow this structure:

```
projects/your-project/
├── README.md              # Project documentation
├── pyproject.toml        # Poetry configuration
├── requirements.txt      # Pip requirements
├── scripts/              # Execution scripts
│   ├── train.py          # Training entry point
│   ├── evaluate.py       # Evaluation script
│   ├── predict.py        # Inference script
│   └── utils/            # Script utilities
├── configs/              # Configuration files
│   ├── model.yaml        # Model architecture
│   ├── data.yaml         # Data processing
│   ├── train.yaml        # Training parameters
│   └── experiments/      # Experiment configs
├── data/                 # Dataset files (DVC-tracked)
│   ├── raw/              # Original data
│   │   ├── train/        # Training data
│   │   ├── val/          # Validation data
│   │   └── test/         # Test data
│   └── processed/        # Processed data
├── src/                  # Source code
│   ├── data/             # Data processing
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── utils.py
│   ├── models/           # Model implementations
│   │   ├── model.py
│   │   ├── layers.py
│   │   └── heads/        # Model heads
│   └── utils/            # Utilities
│       ├── metrics.py
│       ├── visualization.py
│       └── logging.py
├── notebooks/            # Jupyter notebooks
│   ├── exploration/      # Data exploration
│   ├── modeling/         # Model prototyping
│   └── evaluation/       # Model evaluation
├── ui/                   # User interface code
│   ├── streamlit/        # Streamlit interface
│   │   ├── app.py        # Main app
│   │   ├── pages/        # App pages
│   │   └── assets/       # UI resources
│   └── static/           # Shared assets
├── experiments/          # Experiment tracking
│   ├── runs/             # MLflow/experiment runs
│   │   ├── baseline/     # Experiment instance
│   │   └── improved/     # Another experiment
│   ├── models/           # Trained models (DVC-tracked)
│   └── results/          # Evaluation results (DVC-tracked)
├── tests/                # Testing suite
│   ├── conftest.py       # Test configuration
│   ├── test_data.py
│   ├── test_models.py
│   └── test_utils.py
├── docs/                 # Documentation
│   ├── index.md          # Documentation home
│   ├── api/              # API documentation
│   └── guides/           # User guides
├── artifacts/            # Temporary outputs (not tracked)
│   ├── predictions/      # Model predictions
│   ├── checkpoints/      # Training checkpoints
│   └── logs/            # Training logs
├── .dvc/                # DVC configuration
│   ├── cache/           # DVC cache (auto-managed)
│   ├── tmp/             # DVC temporary files
│   └── config           # DVC settings
├── .dvcignore           # DVC ignore patterns
├── .env.example         # Environment variables template
└── .gitignore           # Git ignore patterns
```

## ML Development Standards

### Version Control Strategy

1. **Git-Tracked Content**

   - Source code (src/)
   - Configuration files (configs/)
   - Documentation (docs/)
   - Notebooks (notebooks/)
   - UI code (ui/)
   - Tests (tests/)

2. **DVC-Tracked Content**

   - Dataset files (data/)
   - Trained models (experiments/models/)
   - Important results (experiments/results/)
   - Large binary files

3. **Untracked Content**
   - Temporary files (artifacts/predictions/)
   - Training checkpoints (artifacts/checkpoints/)
   - Log files (artifacts/logs/)
   - Local configs (.env)

### Dependency Management

1. **Poetry Configuration**

   ```toml
   [tool.poetry.dependencies]
   python = "^3.11"
   torch = "^2.3.0"
   opencv-python = "^5.0.0"

   [tool.poetry.group.ui.dependencies]
   streamlit = "^1.32.0"
   gradio = "^4.19.0"

   [tool.poetry.group.dev.dependencies]
   pytest = "^7.0.0"
   black = "^23.0.0"
   mypy = "^1.0.0"
   ```

2. **Requirements Files**

   ```txt
   # requirements.txt - Core dependencies
   torch>=2.3.0
   opencv-python>=5.0.0
   mlflow>=2.10.0
   dvc>=3.30.0
   hydra-core>=1.3.0

   # requirements-ui.txt - UI dependencies
   streamlit>=1.32.0
   gradio>=4.19.0

   # requirements-dev.txt - Development dependencies
   pytest>=7.0.0
   black>=23.0.0
   mypy>=1.0.0
   ```

### Code Quality

```python
# Example with type hints and docstrings
from typing import List, Optional
import torch
import numpy as np

def process_batch(
    images: List[np.ndarray],
    threshold: Optional[float] = None
) -> torch.Tensor:
    """Process a batch of images through the model.

    Args:
        images: List of numpy arrays representing images
        threshold: Optional confidence threshold

    Returns:
        Processed tensor of shape (N, C, H, W)
    """
    # Implementation
    pass
```

### UI Integration

1. **Streamlit App Structure**

   ```python
   # ui/streamlit/app.py
   import streamlit as st
   from src.models import Model

   def main():
       st.title("ML Vision Demo")
       # Implementation
   ```

2. **Asset Organization**
   ```
   ui/
   ├── static/           # Shared assets
   │   ├── css/
   │   └── images/
   └── streamlit/
       ├── app.py
       └── pages/
   ```

## Testing Guidelines

### Unit Tests

```python
# tests/test_models.py
import pytest
import torch

def test_model_output():
    model = Model(config)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, num_classes)
```

### Testing UI Components

```python
# tests/test_ui.py
def test_streamlit_app():
    # Test UI components
    pass
```

## Pull Request Process

1. **Initial Checks**

   - [ ] Code follows project structure
   - [ ] Dependencies documented in both pyproject.toml and requirements.txt
   - [ ] Tests added for both ML and UI components
   - [ ] Documentation updated

2. **Documentation**

   - [ ] Update relevant README files
   - [ ] Document UI components
   - [ ] Add usage examples

3. **Testing**

   - [ ] Run tests: `pytest`
   - [ ] Check coverage: `pytest --cov`
   - [ ] Test UI: `streamlit run ui/streamlit/app.py`

4. **Submit PR**
   - Reference related issues
   - Include test results
   - Add reviewers

## Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Streamlit Documentation](https://docs.streamlit.io/)

Thank you for contributing to ML Vision Lab! 🚀

---

Remember:

- Keep ML and UI code separate
- Document all dependencies
- Test thoroughly
- Follow best practices
