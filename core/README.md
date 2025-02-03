# Core Components 🛠️

> Central hub for shared ML vision components, utilities, and best practices

## 📑 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Core ML Components](#core-ml-components)
- [Guidelines](#guidelines)
- [Usage](#usage)
- [Integration](#integration)

## Overview

This directory contains shared components, utilities, and base implementations used across multiple ML vision projects, focusing on reproducibility, maintainability, and production-grade standards.

## Directory Structure

```
core/
├── models/               # Base model architectures and components
│   ├── architectures/    # Neural network architectures
│   ├── blocks/          # Reusable model blocks
│   ├── heads/           # Task-specific model heads
│   ├── backbones/       # Feature extractors
│   └── versioning/      # Model version control utilities
├── data/                # Data processing components
│   ├── datasets/        # Base dataset classes
│   ├── transforms/      # Data augmentation
│   ├── loaders/         # DataLoader utilities
│   ├── samplers/        # Sampling strategies
│   └── monitoring/      # Data drift detection
├── pipelines/           # ML processing pipelines
│   ├── training/        # Training workflows
│   ├── inference/       # Inference optimization
│   ├── evaluation/      # Metrics computation
│   └── deployment/      # Model serving
└── utils/               # Common utilities
    ├── metrics/         # Evaluation metrics
    ├── visualization/   # Result plotting
    ├── logging/         # Experiment logging
    └── optimization/    # Performance tools
```

## Core ML Components

### 🧠 Model Components

```python
from core.models.architectures import BaseArchitecture
from core.models.blocks import ResidualBlock, AttentionBlock
from core.models.heads import ClassificationHead, DetectionHead

class CustomModel(BaseArchitecture):
    def __init__(self, config):
        super().__init__()
        self.backbone = self.build_backbone(config)
        self.head = self.build_head(config)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
```

### 📊 Data Components

```python
from core.data.datasets import VisionDataset
from core.data.transforms import get_transforms
from core.data.loaders import get_dataloader

class CustomDataset(VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.transforms = get_transforms(config)
        self.loader = get_dataloader(self, batch_size=32)
```

### 📈 Training Pipeline

```python
from core.pipelines.training import Trainer
from core.utils.metrics import MetricLogger
from core.utils.logging import MLflowLogger

class CustomTrainer(Trainer):
    def __init__(self, model, config):
        super().__init__()
        self.logger = MLflowLogger(config)
        self.metric_logger = MetricLogger()

    def train_epoch(self):
        # Custom training logic
        self.logger.log_metrics(self.metric_logger.metrics)
```

## Guidelines

### 🔧 Code Organization

1. **Modularity**

   - Clear interfaces
   - Single responsibility
   - Dependency injection
   - Configuration-driven

2. **ML Best Practices**

   - Reproducible components
   - Experiment tracking
   - Metrics logging
   - Model versioning

3. **Performance**
   - GPU optimization
   - Memory efficiency
   - Batch processing
   - Mixed precision

## Integration

### 🔄 Project Integration

Each project in the `projects/` directory should:

1. **Import Core Components**

```python
# Import base classes
from core.models.architectures import BaseArchitecture
from core.data.datasets import VisionDataset
from core.pipelines.training import BaseTrainer

# Extend for project needs
class ProjectModel(BaseArchitecture):
    def __init__(self, config):
        super().__init__()
        # Project-specific implementation
```

2. **Use Core Utilities**

```python
# Use shared utilities
from core.utils.metrics import calculate_metrics
from core.utils.visualization import plot_results
from core.utils.logging import setup_logging

# Project-specific usage
logger = setup_logging(__name__)
metrics = calculate_metrics(predictions, targets)
plot_results(metrics, save_dir="experiments/results")
```

3. **Follow Project Structure**

```
project-name/
├── src/              # Project-specific implementations
│   ├── models/       # Extends core.models
│   ├── data/         # Uses core.data
│   └── utils/        # Project utilities
├── tests/            # Project-specific tests
├── experiments/      # Project experiments
└── configs/          # Project configs
```

### 📦 Dependencies

Core components require:

```python
# Base requirements
python_requires='>=3.8'
install_requires=[
    'torch>=1.12.0',    # Deep learning
    'numpy>=1.21.0',    # Array operations
    'opencv-python>=4.5.0',  # Image processing
    'albumentations>=1.0.0', # Augmentations
]
```

Projects should include these in their `requirements.txt` or `pyproject.toml`.

### 🔍 Best Practices

1. **Code Reuse**

   - Import from core instead of copying code
   - Extend base classes for customization
   - Use shared utilities consistently

2. **Testing**

   - Test project-specific implementations
   - Verify core component integration
   - Maintain test coverage

3. **Documentation**
   - Document custom implementations
   - Reference core components
   - Provide usage examples

Remember: Core components provide the foundation - extend them in your projects, don't copy them! 💪
