# ML Vision Lab 👁️

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/opencv-5.0%2B-green.svg)](https://opencv.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.2%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Projects](https://img.shields.io/badge/projects-5%2B-brightgreen.svg)](https://github.com/BjornMelin/ml-vision-lab/pulse)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Modular computer vision implementations - A collection of production-grade vision systems spanning multiple domains.

[Featured Projects](#-project-matrix) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Contributing](#-contributing)

## 📑 Table of Contents

- [Project Organization](#-project-organization)
- [Core Features](#-core-features)
- [Prerequisites](#-prerequisites)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Matrix](#-project-matrix)
- [Development Standards](#-development-standards)
- [Contributing](#-contributing)
- [Documentation](#-documentation)
- [Benchmarks](#-benchmarks)
- [Versioning](#-versioning)
- [Authors](#-authors)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🗂️ Project Organization

```mermaid
graph TD
    A[ML Vision Lab] --> B[projects]
    A --> C[core]
    A --> D[docs]
    B --> E[food-classification]
    B --> F[object-detection]
    B --> G[medical-imaging]
    B --> H[satellite-analysis]
    C --> I[utils]
    C --> J[models]
    C --> K[pipelines]
    D --> L[api]
    D --> M[guides]
    D --> N[architecture]
```

```
ml-vision-lab/
├── projects/               # Individual vision projects
│   ├── food-classification/  # Food analysis system
│   ├── object-detection/      # Real-time detection
│   ├── medical-imaging/       # DICOM processing
│   └── satellite-analysis/    # Geospatial vision
├── core/                   # Shared vision components
│   ├── utils/              # Common utilities
│   ├── models/             # Base model architectures
│   └── pipelines/          # Processing workflows
└── docs/                   # Project documentation
```

## ✨ Core Features

```mermaid
mindmap
  root((ML Vision Lab))
    Cross-Project
      Modular architecture
      Shared pipelines
      Hardware optimization
      Standardized metrics
    Project Types
      Classification
      Detection
      Medical
      Satellite
    Optimization
      GPU acceleration
      TensorRT
      Memory efficiency
    Development
      MLflow tracking
      DVC versioning
      CI/CD pipelines
```

**Cross-Project Capabilities**

- Modular project architecture
- Shared preprocessing pipelines
- Hardware-optimized inference
- Standardized evaluation metrics
- GPU-accelerated processing
- Production deployment examples
- Memory-efficient inference
- TensorRT integration

**Project Types**

- Image Classification
- Object Detection & Tracking
- Medical Imaging Analysis
- Satellite Imagery Processing
- Industrial Quality Inspection

## 🔧 Prerequisites

- Python 3.11+
- CUDA 12.2+
- OpenCV 5.0+
- PyTorch 2.3+
- TensorFlow 2.15+
- NVIDIA GPU (Compute Capability 6.0+)

## 🛠️ Tech Stack

```mermaid
graph TD
    A[Tech Stack] --> B[Core Libraries]
    A --> C[Project Libraries]
    B --> D[PyTorch]
    B --> E[TensorFlow]
    B --> F[OpenCV]
    B --> G[CUDA]
    C --> H[MONAI]
    C --> I[RasterIO]
    C --> J[DeepSORT]
    C --> K[MLflow]
```

**Core Libraries**

- PyTorch - Deep learning framework
- TensorFlow - Machine learning platform
- OpenCV - Computer vision operations
- CUDA - GPU acceleration
- TensorRT - Inference optimization
- NumPy - Numerical computing
- Pandas - Data manipulation
- Scikit-learn - Machine learning utilities
- Matplotlib - Visualization
- Plotly - Visualization
- Pillow - Image processing

**Project-Specific Libraries**

- MONAI - Medical imaging
- RasterIO - Geospatial analysis
- DeepSORT - Object tracking
- Albumentations - Image augmentation
- MLflow - Experiment tracking
- DVC - Data version control

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/BjornMelin/ml-vision-lab.git
cd ml-vision-lab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# or
.venv\Scripts\activate  # Windows

# Install core requirements
pip install -r requirements.txt

# Install project-specific requirements (optional)
pip install -r projects/food-classification/requirements.txt
```

## 🚀 Quick Start

**Food Classification**

```python
from projects.food_classification import predict

result = predict("pizza.jpg")
print(f"Identified: {result.label} ({result.confidence:.1%})")
```

**Object Detection**

```python
from projects.object_detection import VideoAnalyzer

analyzer = VideoAnalyzer(model="yolov9")
analyzer.process_stream("input.mp4", output="results.mp4")
```

## 📊 Project Matrix

| Project                                             | Task                 | Models              | Input Types   |
| --------------------------------------------------- | -------------------- | ------------------- | ------------- |
| [Food Classification](projects/food-classification) | Image Classification | EfficientNetV2, ViT | JPEG/PNG      |
| [Object Detection](projects/object-detection)       | Real-time Tracking   | YOLOv9, DeepSORT    | Video Streams |
| [Medical Imaging](projects/medical-imaging)         | DICOM Analysis       | UNet3+, MONAI       | CT/MRI Scans  |
| [Satellite Analysis](projects/satellite-analysis)   | Geospatial ML        | ResNet50-ADE20K     | GeoTIFF       |

## 🔧 Development Standards

```mermaid
flowchart TD
    A[Development] --> B[Code Quality]
    A --> C[Testing]
    A --> D[Documentation]
    B --> E[Black]
    B --> F[MyPy]
    C --> G[PyTest]
    C --> H[Coverage]
    D --> I[Docstrings]
    D --> J[Examples]
```

**Code Quality**

```bash
# Format all projects
black projects/

# Type checking
mypy projects/

# Run tests
pytest projects/ --cov
```

**Project Structure Template**

```
projects/new-project/
├── app/          # Application interface
├── engine/       # Core logic
├── models/       # Trained weights
├── tests/        # Unit tests
├── README.md     # Project docs
└── requirements.txt # Local dependencies
```

## 🤝 Contributing

**Adding New Projects**

1. Create project folder in `projects/`
2. Follow structure template
3. Add cross-links to:
   - Core utilities (avoid duplication)
   - Related projects
4. Submit PR with:
   - [ ] Black-formatted code
   - [ ] Google-style docstrings
   - [ ] Unit tests (≥80% coverage)

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## 📚 Documentation

### Pipeline Optimization

```mermaid
graph LR
    A[Input] --> B[Preprocessing]
    B --> C[Inference]
    C --> D[Postprocessing]
    B --> E[GPU Pipeline]
    C --> F[TensorRT]
    D --> G[Batch Processing]
```

- GPU-accelerated preprocessing
- Batch processing optimization
- Memory-efficient inference
- TensorRT integration
- Multi-GPU support
- Mixed precision training

### Models

| Model      | Task         | Performance | Speed (FPS) |
| ---------- | ------------ | ----------- | ----------- |
| YOLOv8     | Detection    | mAP: 52.3   | 120         |
| Mask R-CNN | Segmentation | mAP: 47.8   | 45          |
| DeepSORT   | Tracking     | MOTA: 76.5  | 80          |

## 📊 Benchmarks

Performance on standard datasets:

| Task         | Dataset | Model      | GPU  | FPS | Accuracy   |
| ------------ | ------- | ---------- | ---- | --- | ---------- |
| Detection    | COCO    | YOLOv8     | A100 | 120 | mAP: 52.3  |
| Segmentation | COCO    | Mask R-CNN | V100 | 45  | mAP: 47.8  |
| Tracking     | MOT17   | DeepSORT   | 3090 | 80  | MOTA: 76.5 |

## 📌 Versioning

We use [SemVer](http://semver.org/) for versioning. For available versions, see the [tags on this repository](https://github.com/BjornMelin/ml-vision-lab/tags).

## ✍️ Authors

**Bjorn Melin**

- GitHub: [@BjornMelin](https://github.com/BjornMelin)
- LinkedIn: [Bjorn Melin](https://linkedin.com/in/bjorn-melin)

## 📝 Citation

```bibtex
@misc{melin2024mlvisionlab,
  author = {Melin, Bjorn},
  title = {ML Vision Lab: Production Computer Vision Implementations},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/BjornMelin/ml-vision-lab}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community
- YOLO authors and contributors
- Deep SORT implementation team
- Medical imaging community (MONAI)
- Satellite imagery processing teams
- TensorFlow and PyTorch teams
- NVIDIA for CUDA and TensorRT support

---

![Architecture Overview](docs/architecture/overview.png)

Made with 👁️ and ❤️ by Bjorn Melin
