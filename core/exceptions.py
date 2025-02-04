"""Core exception classes for ML vision library.

This module defines custom exceptions used throughout the library for handling
various error cases in data processing, model operations, pipeline execution,
and configuration management.
"""


class CoreException(Exception):
    """Base exception class for all custom exceptions."""

    pass


class DataError(CoreException):
    """Base class for data-related exceptions."""

    pass


class DataLoadError(DataError):
    """Raised when data loading fails.

    Examples:
        >>> try:
        ...     dataset = load_dataset("path/to/data")
        ... except DataLoadError as e:
        ...     logger.error(f"Failed to load dataset: {e}")
    """

    pass


class DataTransformError(DataError):
    """Raised when data transformation fails.

    Examples:
        >>> try:
        ...     transformed = transform(image)
        ... except DataTransformError as e:
        ...     logger.error(f"Transform failed: {e}")
    """

    pass


class ModelError(CoreException):
    """Base class for model-related exceptions."""

    pass


class ModelInitError(ModelError):
    """Raised when model initialization fails.

    Examples:
        >>> try:
        ...     model = create_model(config)
        ... except ModelInitError as e:
        ...     logger.error(f"Model initialization failed: {e}")
    """

    pass


class ModelLoadError(ModelError):
    """Raised when loading model weights fails.

    Examples:
        >>> try:
        ...     model.load_state_dict(checkpoint)
        ... except ModelLoadError as e:
        ...     logger.error(f"Failed to load weights: {e}")
    """

    pass


class PipelineError(CoreException):
    """Base class for pipeline-related exceptions."""

    pass


class TrainingError(PipelineError):
    """Raised when training pipeline encounters an error.

    Examples:
        >>> try:
        ...     trainer.train(train_loader)
        ... except TrainingError as e:
        ...     logger.error(f"Training failed: {e}")
    """

    pass


class InferenceError(PipelineError):
    """Raised when inference pipeline encounters an error.

    Examples:
        >>> try:
        ...     predictions = model.predict(data)
        ... except InferenceError as e:
        ...     logger.error(f"Inference failed: {e}")
    """

    pass


class EvaluationError(PipelineError):
    """Raised when evaluation pipeline encounters an error.

    Examples:
        >>> try:
        ...     metrics = evaluator.evaluate(val_loader)
        ... except EvaluationError as e:
        ...     logger.error(f"Evaluation failed: {e}")
    """

    pass


class ConfigError(CoreException):
    """Raised when configuration validation fails.

    Examples:
        >>> try:
        ...     config = load_config("config.yaml")
        ... except ConfigError as e:
        ...     logger.error(f"Invalid config: {e}")
    """

    pass


class ResourceError(CoreException):
    """Raised when resource allocation/access fails.

    Examples:
        >>> try:
        ...     device = get_device()
        ... except ResourceError as e:
        ...     logger.error(f"Failed to get device: {e}")
    """

    pass


class ValidationError(CoreException):
    """Raised when input validation fails.

    Examples:
        >>> try:
        ...     validate_input(data)
        ... except ValidationError as e:
        ...     logger.error(f"Invalid input: {e}")
    """

    pass
