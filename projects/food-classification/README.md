# food-classification

> Brief description of the ML/CV project and its objectives

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-red.svg)](https://pytorch.org/)
[![DVC](https://img.shields.io/badge/dvc-3.30%2B-violet.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/mlflow-2.10%2B-yellow.svg)](https://mlflow.org/)

## Overview

This project provides a complete, reproducible pipeline for training and deploying a food classification model using the Food-101 dataset. It leverages PyTorch (with timm) for model development, Hydra for flexible configuration, and integrates DVC for data/model versioning alongside MLflow for experiment tracking. A Streamlit UI offers an intuitive interface for uploading images and visualizing classification results in real time.

## Quick Start

```bash
# Clone project
git clone https://github.com/username/project.git
cd project

# Set up environment
cp .env.example .env  # Edit with your settings
poetry install       # or: pip install -r requirements.txt

# Download data
dvc pull

# Train model
python scripts/train.py
```

## Project Structure

```
.
├── scripts/          # Execution scripts
│   ├── train.py     # Training entry point
│   ├── evaluate.py  # Evaluation script
│   ├── predict.py   # Inference script
│   └── utils/       # Script utilities
├── configs/         # Configuration files
│   ├── model.yaml   # Model architecture
│   ├── data.yaml    # Data processing
│   ├── train.yaml   # Training parameters
│   └── experiments/ # Experiment configs
├── data/           # Dataset files (DVC-tracked)
│   ├── raw/        # Original data
│   │   ├── train/  # Training data
│   │   ├── val/    # Validation data
│   │   └── test/   # Test data
│   └── processed/  # Processed data
├── src/            # Source code
│   ├── data/       # Data processing
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── utils.py
│   ├── models/     # Model implementations
│   │   ├── model.py
│   │   ├── layers.py
│   │   └── heads/  # Model heads
│   └── utils/      # Utilities
│       ├── metrics.py
│       ├── visualization.py
│       └── logging.py
├── notebooks/      # Jupyter notebooks
│   ├── exploration/# Data exploration
│   ├── modeling/   # Model prototyping
│   └── evaluation/ # Model evaluation
├── ui/             # User interface code
│   ├── streamlit/  # Streamlit interface
│   │   ├── app.py  # Main app
│   │   ├── pages/  # App pages
│   │   └── assets/ # UI resources
│   └── static/     # Shared assets
├── experiments/    # Experiment tracking
│   ├── runs/       # MLflow/experiment runs
│   │   ├── baseline/
│   │   └── improved/
│   ├── models/     # Trained models (DVC-tracked)
│   └── results/    # Evaluation results (DVC-tracked)
├── tests/          # Testing suite
│   ├── conftest.py # Test configuration
│   ├── test_data.py
│   ├── test_models.py
│   └── test_utils.py
├── docs/           # Documentation
│   ├── index.md    # Documentation home
│   ├── api/        # API documentation
│   └── guides/     # User guides
├── artifacts/      # Temporary outputs (not tracked)
│   ├── predictions/# Model predictions
│   ├── checkpoints/# Training checkpoints
│   └── logs/      # Training logs
├── .dvc/          # DVC configuration
│   ├── cache/     # DVC cache (auto-managed)
│   ├── tmp/       # DVC temporary files
│   └── config     # DVC settings
├── .dvcignore     # DVC ignore patterns
├── .env.example   # Environment variables template
└── .gitignore     # Git ignore patterns
```

## Development

### Environment Setup

```bash
# DVC configuration
dvc init
dvc remote add -d storage s3://bucket/path

# Configure DVC
# .dvcignore
artifacts/          # Ignore temporary outputs
*.pyc              # Ignore Python cache
__pycache__/       # Ignore Python cache directories
.ipynb_checkpoints # Ignore Jupyter checkpoints

# .dvc/config
[core]
    remote = storage
    autostage = true
[cache]
    type = "hardlink,symlink"
    dir = .dvc/cache
```

### Dependencies

```toml
# pyproject.toml
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
```

### Training

```python
# scripts/train.py
from src.models import Model
from src.data import DataLoader

def train():
    model = Model(config)
    trainer = Trainer(model)
    trainer.train()

if __name__ == "__main__":
    train()
```

### Experiment Tracking

```python
# experiments/runs/baseline/run.py
with mlflow.start_run():
    mlflow.log_params(config)
    train()
    mlflow.log_metrics(metrics)
```

## Results

### Performance

| Model    | Accuracy | FPS | Memory |
| -------- | -------- | --- | ------ |
| Baseline | 85.5%    | 120 | 2.4 GB |
| Improved | 89.2%    | 95  | 3.8 GB |

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- List key contributors
- Cite papers/repos
- Credit data sources

---

Made with 🧠 by Generated via GitHub Actions
