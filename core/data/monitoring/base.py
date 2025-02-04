"""Base monitoring implementations for data drift detection and statistics tracking."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy import stats


class DatasetStatistics:
    """Track and compute dataset statistics."""

    def __init__(self):
        """Initialize statistics tracker."""
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self._count = 0
        self._mean = None
        self._m2 = None  # For online variance computation
        self._min = None
        self._max = None
        self._histogram_bins = None
        self._histogram_counts = None

    def update(self, batch: np.ndarray):
        """Update statistics with new batch of data.

        Uses Welford's online algorithm for computing mean and variance.

        Args:
            batch: New data batch (N x D array)
        """
        if len(batch.shape) == 1:
            batch = batch.reshape(-1, 1)

        if self._mean is None:
            self._mean = np.zeros(batch.shape[1])
            self._m2 = np.zeros(batch.shape[1])
            self._min = np.full(batch.shape[1], np.inf)
            self._max = np.full(batch.shape[1], -np.inf)

        for x in batch:
            self._count += 1
            delta = x - self._mean
            self._mean += delta / self._count
            delta2 = x - self._mean
            self._m2 += delta * delta2

            # Update min/max
            self._min = np.minimum(self._min, x)
            self._max = np.maximum(self._max, x)

    def compute_histogram(self, data: np.ndarray, bins: int = 50):
        """Compute histogram for data distribution.

        Args:
            data: Data to compute histogram for
            bins: Number of histogram bins
        """
        self._histogram_bins, self._histogram_counts = np.histogram(
            data, bins=bins, density=True
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics.

        Returns:
            Dictionary of statistics
        """
        if self._count == 0:
            return {}

        stats = {
            "count": self._count,
            "mean": self._mean.tolist(),
            "std": np.sqrt(self._m2 / (self._count - 1)).tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
        }

        if self._histogram_bins is not None:
            stats.update(
                {
                    "histogram_bins": self._histogram_bins.tolist(),
                    "histogram_counts": self._histogram_counts.tolist(),
                }
            )

        return stats


class DriftDetector:
    """Detect distribution drift in data streams."""

    def __init__(
        self, window_size: int = 1000, threshold: float = 0.05, test: str = "ks"
    ):
        """Initialize drift detector.

        Args:
            window_size: Size of reference window
            threshold: P-value threshold for drift detection
            test: Statistical test to use ('ks' or 'ttest')
        """
        self.window_size = window_size
        self.threshold = threshold
        self.test = test

        self.reference_window = []
        self.current_window = []

    def update_reference(self, data: np.ndarray):
        """Update reference window.

        Args:
            data: New reference data
        """
        self.reference_window = list(data[-self.window_size :])

    def add_sample(self, sample: Union[float, np.ndarray]):
        """Add new sample to current window.

        Args:
            sample: New data sample
        """
        self.current_window.append(sample)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)

    def detect_drift(self) -> Dict[str, Any]:
        """Check for drift between reference and current windows.

        Returns:
            Dictionary containing drift detection results
        """
        if len(self.current_window) < self.window_size:
            return {"drift_detected": False, "message": "Insufficient data"}

        if not self.reference_window:
            return {"drift_detected": False, "message": "No reference data"}

        # Convert to numpy arrays
        reference = np.array(self.reference_window)
        current = np.array(self.current_window)

        # Perform statistical test
        if self.test == "ks":
            statistic, p_value = stats.ks_2samp(reference, current)
        elif self.test == "ttest":
            statistic, p_value = stats.ttest_ind(reference, current)
        else:
            raise ValueError(f"Unknown test: {self.test}")

        drift_detected = p_value < self.threshold

        return {
            "drift_detected": drift_detected,
            "p_value": float(p_value),
            "statistic": float(statistic),
            "test": self.test,
            "threshold": self.threshold,
        }


class DataMonitor:
    """Monitor dataset characteristics and drift."""

    def __init__(
        self, feature_names: Optional[List[str]] = None, window_size: int = 1000
    ):
        """Initialize data monitor.

        Args:
            feature_names: Names of features to monitor
            window_size: Window size for drift detection
        """
        self.feature_names = feature_names
        self.statistics = DatasetStatistics()
        self.drift_detector = DriftDetector(window_size=window_size)

    def update(self, batch: np.ndarray):
        """Update monitoring with new batch.

        Args:
            batch: New data batch
        """
        # Update statistics
        self.statistics.update(batch)

        # Update drift detector
        for sample in batch:
            self.drift_detector.add_sample(sample)

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status.

        Returns:
            Dictionary containing monitoring results
        """
        stats = self.statistics.get_statistics()
        drift_results = self.drift_detector.detect_drift()

        return {"statistics": stats, "drift": drift_results}
