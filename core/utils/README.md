# Utility Components 🛠️

> Common utilities and helper functions for ML vision tasks

## 📑 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Components](#components)
- [Usage Guidelines](#usage-guidelines)
- [Best Practices](#best-practices)

## Overview

This directory contains shared utilities and helper functions used across different vision projects, providing common functionality for metrics, visualization, logging, and optimization.

## Directory Structure

```
utils/
├── metrics/           # Evaluation metrics
│   ├── classification.py # Classification metrics
│   ├── detection.py     # Detection metrics
│   └── segmentation.py  # Segmentation metrics
├── visualization/     # Result visualization
│   ├── plotting.py     # Plot generation
│   ├── images.py       # Image visualization
│   └── tensorboard.py  # TensorBoard logging
├── logging/          # Experiment logging
│   ├── logger.py      # Base logger
│   ├── mlflow.py      # MLflow integration
│   └── wandb.py       # Weights & Biases
└── optimization/     # Performance tools
    ├── profiler.py    # Code profiling
    ├── memory.py      # Memory optimization
    └── cuda.py        # GPU utilities
```

## Components

### Metrics

```python
from core.utils.metrics import (
    calculate_accuracy,
    calculate_precision_recall,
    calculate_map
)

# Classification metrics
accuracy = calculate_accuracy(predictions, targets)
precision, recall = calculate_precision_recall(predictions, targets)

# Detection metrics
map_score = calculate_map(
    predictions,
    targets,
    iou_threshold=0.5
)
```

### Visualization

```python
from core.utils.visualization import (
    plot_results,
    visualize_predictions,
    log_to_tensorboard
)

# Plot results
plot_results(
    metrics_dict,
    save_path="results/plot.png"
)

# Visualize predictions
visualize_predictions(
    images,
    predictions,
    targets,
    save_dir="results/viz"
)
```

### Logging

```python
from core.utils.logging import setup_logging, MLflowLogger

# Setup logging
logger = setup_logging(
    name=__name__,
    log_file="logs/training.log"
)

# MLflow logging
mlflow_logger = MLflowLogger(
    experiment_name="vision_experiment",
    tracking_uri="mlruns"
)
mlflow_logger.log_params(config)
mlflow_logger.log_metrics(metrics)
```

## Usage Guidelines

### 1. Metrics Tracking

```python
from core.utils.metrics import MetricsTracker

# Initialize tracker
tracker = MetricsTracker(
    metrics=['accuracy', 'loss'],
    save_dir='logs'
)

# Update metrics
tracker.update({
    'accuracy': 0.85,
    'loss': 0.32
})

# Get summary
summary = tracker.get_summary()
```

### 2. Visualization

```python
from core.utils.visualization import (
    plot_learning_curves,
    create_confusion_matrix
)

# Plot learning curves
plot_learning_curves(
    train_metrics,
    val_metrics,
    save_path='plots/learning_curves.png'
)

# Create confusion matrix
create_confusion_matrix(
    predictions,
    targets,
    save_path='plots/confusion_matrix.png'
)
```

### 3. Optimization

```python
from core.utils.optimization import (
    profile_model,
    optimize_memory_usage
)

# Profile model
profile_results = profile_model(
    model,
    input_shape=(1, 3, 224, 224)
)

# Optimize memory
optimize_memory_usage(
    model,
    batch_size=32
)
```

## Best Practices

### 1. Logging

- Use structured logging
- Include timestamps
- Set appropriate levels
- Handle exceptions
- Rotate log files

### 2. Visualization

- Use consistent styling
- Add proper labels
- Include legends
- Save high quality
- Enable interactivity

### 3. Metrics

- Validate inputs
- Handle edge cases
- Use appropriate metrics
- Track uncertainties
- Save raw data

### 4. Optimization

- Profile before optimizing
- Monitor memory usage
- Track GPU utilization
- Benchmark changes
- Document optimizations

## Integration

### With Core Components

```python
from core.models import BaseModel
from core.utils.optimization import optimize_model
from core.utils.logging import log_model_summary

class OptimizedModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = optimize_model(self.model)
        log_model_summary(self.model)
```

### With Projects

```python
from core.utils import setup_project

# Setup project
logger, metrics, viz = setup_project(
    name="vision_project",
    config=config
)

# Use utilities
logger.info("Starting training...")
metrics.update(train_metrics)
viz.plot_results(results)
```

Remember: Good utilities make development faster and more reliable! 💪
