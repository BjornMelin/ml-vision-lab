# Projects Directory 🚀

> Collection of modular computer vision implementations following ML/CV best practices

## 📑 Table of Contents

- [Overview](#overview)
- [Project Organization](#project-organization)
- [ML Development Standards](#ml-development-standards)
- [Project Creation Checklist](#project-creation-checklist)

## Overview

This directory contains individual computer vision projects, each following industry-standard ML project organization and best practices for reproducibility, maintainability, and production deployment.

## Project Organization

### Standard Project Structure

```
project-name/
├── README.md          # Project documentation
├── pyproject.toml    # Poetry/project metadata
├── requirements.txt  # Pip requirements (alternative to Poetry)
├── configs/          # Configuration files
│   ├── model.yaml    # Model architecture
│   ├── data.yaml     # Data processing
│   └── train.yaml    # Training parameters
├── data/             # Dataset files
│   ├── raw/          # Original data
│   └── processed/    # Processed data
├── src/              # Source code
│   ├── data/         # Data processing
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/       # Model implementations
│   │   ├── model.py
│   │   └── layers.py
│   ├── utils/        # Utilities
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── train.py      # Training script
│   ├── evaluate.py   # Evaluation script
│   └── predict.py    # Inference script
├── ui/               # User interface code
│   ├── streamlit/    # Streamlit interface
│   │   ├── app.py    # Main Streamlit app
│   │   ├── pages/    # App pages
│   │   └── assets/   # Images, css, etc.
│   ├── gradio/       # Gradio interface (optional)
│   └── static/       # Shared static files
├── experiments/      # Experiment tracking
│   ├── runs/         # MLflow/experiment runs
│   ├── notebooks/    # Analysis notebooks
│   └── results/      # Evaluation results
│       ├── metrics/  # Performance metrics
│       └── plots/    # Visualizations
├── tests/            # Testing suite
│   ├── test_data.py
│   ├── test_models.py
│   └── test_utils.py
├── docs/             # Additional documentation
│   ├── api.md        # API documentation
│   └── guides/       # User/dev guides
├── artifacts/        # Generated files
│   ├── models/       # Saved models
│   └── logs/         # Training logs
├── .dvc/            # Data version control
├── .env.example     # Example environment variables
└── .gitignore       # Git ignore patterns
```

### Dependencies Management

1. **Using Poetry (Recommended)**

   ```toml
   # pyproject.toml
   [tool.poetry]
   name = "project-name"
   version = "0.1.0"

   [tool.poetry.dependencies]
   python = "^3.11"
   torch = "^2.3.0"

   [tool.poetry.group.ui.dependencies]
   streamlit = "^1.32.0"
   gradio = "^4.19.0"
   ```

2. **Using Pip (Alternative)**

   ```txt
   # requirements.txt
   torch>=2.3.0
   opencv-python-headless>=5.0.0

   # UI dependencies
   streamlit>=1.32.0
   gradio>=4.19.0
   ```

### User Interface Integration

1. **Streamlit App**

   ```python
   # ui/streamlit/app.py
   import streamlit as st
   from src.models import Model
   from src.utils.visualization import visualize_results

   def main():
       st.title("ML Vision Demo")

       # File upload
       image = st.file_uploader("Upload image", type=["jpg", "png"])

       if image:
           # Process image
           model = Model.load("artifacts/models/best.pt")
           results = model.predict(image)

           # Display results
           st.image(visualize_results(results))

   if __name__ == "__main__":
       main()
   ```

2. **Running the UI**

   ```bash
   # Start Streamlit app
   streamlit run ui/streamlit/app.py

   # Start MLflow UI (separate terminal)
   mlflow ui
   ```

## Project Creation Checklist

### 🚀 Initial Setup

1. **Project Structure**

   ```bash
   # Create project
   mkdir project-name
   cd project-name

   # Initialize Poetry
   poetry init

   # Generate requirements.txt (alternative)
   poetry export -f requirements.txt --output requirements.txt

   # Create directories
   mkdir -p src/{data,models,utils}
   mkdir -p ui/streamlit/pages
   mkdir -p experiments/{runs,notebooks,results}
   mkdir -p tests docs artifacts
   ```

2. **Version Control**

   ```bash
   # Initialize Git and DVC
   git init
   dvc init

   # Configure DVC storage
   dvc remote add -d storage s3://bucket/path
   ```

3. **UI Setup**
   ```bash
   # Add UI dependencies
   poetry add --group ui streamlit gradio
   # or
   pip install streamlit gradio
   ```

### Best Practices

1. **Dependency Management**

   - Use Poetry for development
   - Maintain requirements.txt for compatibility
   - Group UI dependencies separately

2. **UI Organization**

   - Keep UI code separate from ML logic
   - Use shared static assets
   - Modular UI components

3. **Documentation**
   - Document UI setup and usage
   - Include screenshots/demos
   - Provide API documentation

Remember: Keep ML and UI code separate but well-integrated. This makes both components easier to maintain and deploy! 💪
