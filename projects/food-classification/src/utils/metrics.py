from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torchmetrics import Metric


class TopKAccuracy(Metric):
    """Custom metric for top-k accuracy."""

    def __init__(self, k: int = 5, dist_sync_on_step: bool = False):
        """Initialize metric.

        Args:
            k: Number of top predictions to consider
            dist_sync_on_step: Sync across GPUs on step
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric state.

        Args:
            preds: Predicted logits of shape (B, C)
            target: Ground truth labels of shape (B,)
        """
        with torch.no_grad():
            _, pred_labels = torch.topk(preds, self.k, dim=1)
            correct = torch.any(pred_labels == target.view(-1, 1), dim=1)

            self.correct += torch.sum(correct)
            self.total += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute final metric value.

        Returns:
            torch.Tensor: Top-k accuracy value
        """
        return self.correct.float() / self.total


def calculate_metrics(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> Dict[str, float]:
    """Calculate various metrics for model predictions.

    Args:
        logits: Model output logits of shape (B, C)
        targets: Ground truth labels of shape (B,)
        k: K value for top-k accuracy

    Returns:
        dict: Dictionary of metric names and values
    """
    with torch.no_grad():
        # Top-1 accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean().item()

        # Top-k accuracy
        _, pred_labels = torch.topk(logits, k, dim=1)
        top_k_acc = (
            torch.any(pred_labels == targets.view(-1, 1), dim=1).float().mean().item()
        )

        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets).item()

        return {"accuracy": acc, f"top_{k}_accuracy": top_k_acc, "loss": loss}
