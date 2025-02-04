"""
Custom exception classes for ML Vision Lab.

These exceptions provide descriptive error messages and allow for
consistent error handling throughout the repository.
"""


class MLBaseException(Exception):
    """Base exception for all errors in the ML Vision Lab."""

    pass


class DataLoadError(MLBaseException):
    """Raised when there is an error loading or processing data."""

    def __init__(self, message="Error occurred while loading data.", errors=None):
        super().__init__(message)
        self.errors = errors


class ModelError(MLBaseException):
    """Raised when model training, inference, or evaluation fails."""

    def __init__(
        self, message="An error occurred in the model operation.", errors=None
    ):
        super().__init__(message)
        self.errors = errors


class PipelineError(MLBaseException):
    """Raised when a step in the data or model pipeline fails."""

    def __init__(self, message="Pipeline execution failed.", errors=None):
        super().__init__(message)
        self.errors = errors


class ConfigError(MLBaseException):
    """Raised when there is an issue with configuration or parameter parsing."""

    def __init__(self, message="Configuration error encountered.", errors=None):
        super().__init__(message)
        self.errors = errors
