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
├── pyproject.toml          # Poetry configuration
├── requirements.txt        # Pip requirements
├── configs/               # Configuration files
│   ├── model.yaml         # Model hyperparameters
│   ├── data.yaml          # Data processing settings
│   └── train.yaml         # Training parameters
├── data/                 # Dataset files
│   ├── raw/                 # Original data
│   └── processed/           # Processed data
├── src/                 # Source code
│   ├── data/               # Data processing
│   ├── models/             # Model implementations
│   ├── utils/              # Utilities
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── predict.py         # Inference script
├── ui/                  # User interface code
│   ├── streamlit/         # Streamlit interface
│   │   ├── app.py         # Main app
│   │   └── pages/         # App pages
│   └── static/           # Shared assets
├── tests/               # Testing suite
├── experiments/         # Experiment tracking
│   ├── runs/             # MLflow/experiment runs
│   └── results/          # Evaluation results
├── docs/               # Documentation
├── artifacts/          # Generated files
└── .dvc/              # DVC configuration
```

## ML Development Standards

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
