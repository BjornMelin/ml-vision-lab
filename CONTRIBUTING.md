# Contributing to ML Vision Lab

Thank you for your interest in contributing to ML Vision Lab! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
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

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/MacOS
   # or
   .venv\Scripts\activate  # Windows

   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Project Structure

When adding a new project, follow this structure:

```
projects/your-project/
├── app/              # Application interface
├── engine/           # Core logic
├── models/           # Trained weights
├── tests/            # Unit tests
├── README.md         # Project documentation
└── requirements.txt  # Project dependencies
```

Each project must include:

- Clear documentation in README.md
- Requirements file listing dependencies
- Unit tests with ≥80% coverage
- Type hints for all functions
- Google-style docstrings

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for formatting
- Include type hints ([PEP 484](https://www.python.org/dev/peps/pep-0484/))
- Use Google-style docstrings

### Example Code Style

```python
from typing import List, Optional

def process_images(
    images: List[np.ndarray],
    model_type: str = "yolov9",
    confidence: Optional[float] = None
) -> Dict[str, Any]:
    """Process a batch of images through detection model.

    Args:
        images: List of numpy arrays representing images.
        model_type: Type of model to use for detection.
        confidence: Optional confidence threshold.

    Returns:
        Dictionary containing detection results.

    Raises:
        ValueError: If images list is empty.
    """
    if not images:
        raise ValueError("Images list cannot be empty")

    # Implementation
    return results
```

### Code Quality Tools

```bash
# Format code
black projects/

# Type checking
mypy projects/

# Linting
ruff projects/

# Run tests
pytest projects/ --cov
```

## Pull Request Process

1. **Create Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Development Checklist**

   - [ ] Follow project structure template
   - [ ] Add unit tests
   - [ ] Update documentation
   - [ ] Run code quality tools
   - [ ] Test on supported Python versions

3. **Commit Guidelines**

   - Use semantic commit messages:
     - feat: New feature
     - fix: Bug fix
     - docs: Documentation changes
     - style: Formatting changes
     - refactor: Code restructuring
     - test: Adding tests
     - chore: Maintenance tasks

4. **Documentation**

   - Update relevant README.md files
   - Add docstrings to new functions/classes
   - Include examples for new features

5. **Submit PR**
   - Fill out PR template
   - Link related issues
   - Add project maintainers as reviewers

## Testing Guidelines

### Unit Tests

- Use pytest for testing
- Maintain ≥80% code coverage
- Test edge cases and error conditions
- Mock external dependencies

### Example Test

```python
import pytest
from your_project import process_images

def test_process_images_empty_input():
    with pytest.raises(ValueError, match="Images list cannot be empty"):
        process_images([])

def test_process_images_valid_input(mock_model):
    images = [np.zeros((224, 224, 3))]
    result = process_images(images)
    assert isinstance(result, dict)
    assert "detections" in result
```

### Performance Testing

- Include benchmarks for critical operations
- Test with various input sizes
- Document performance characteristics

## Getting Help

If you need help, you can:

- Open an issue for questions
- Join our community discussions
- Review existing documentation

Thank you for contributing to ML Vision Lab! 🚀

---

## Additional Resources

- [Python Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
