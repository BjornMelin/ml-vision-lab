import torch
import torch.nn.functional as F

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute the accuracy of predictions.

    Args:
        output: Model predictions
        target: Ground truth labels

    Returns:
        Accuracy as a float
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0)

def top_k_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """Compute the top-k accuracy of predictions.

    Args:
        output: Model predictions
        target: Ground truth labels
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy as a float
    """
    with torch.no_grad():
        _, pred = output.topk(k, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred)).sum().item()
        return correct / target.size(0)

def cross_entropy_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the cross-entropy loss.

    Args:
        output: Model predictions
        target: Ground truth labels

    Returns:
        Cross-entropy loss as a tensor
    """
    return F.cross_entropy(output, target)
