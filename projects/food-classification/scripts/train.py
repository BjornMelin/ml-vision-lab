import logging
from pathlib import Path
from typing import Dict, Any

import hydra
import mlflow
import torch
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.dataset import Food101Dataset
from src.data.transforms import build_transforms
from src.models.efficientnet import FoodClassifier
from src.utils.metrics import calculate_metrics

log = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    cfg: DictConfig,
) -> Dict[str, float]:
    """Train model for one epoch.

    Args:
        model: PyTorch model
        loader: Training data loader
        optimizer: PyTorch optimizer
        scaler: Gradient scaler for mixed precision
        epoch: Current epoch number
        cfg: Training configuration

    Returns:
        dict: Training metrics
    """
    model.train()
    total_loss = 0
    total_metrics = {}

    for i, (images, targets) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()

        # Mixed precision training
        with autocast():
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if cfg.model.get("grad_clip"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.model.grad_clip.max_norm
            )
        scaler.step(optimizer)
        scaler.update()

        # Log metrics
        if i % cfg.model.metrics.log_interval == 0:
            metrics = calculate_metrics(logits, targets, k=cfg.model.metrics.top_k)
            mlflow.log_metrics(
                {f"train/{k}": v for k, v in metrics.items()},
                step=epoch * len(loader) + i,
            )

            # Update running averages
            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

    # Compute epoch averages
    num_batches = len(loader)
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = total_loss / num_batches

    return avg_metrics


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # Set random seed
    torch.manual_seed(cfg.training.seed)

    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.start_run(run_name=cfg.mlflow.run_name)

    # Log config
    mlflow.log_params(
        {
            f"{k1}.{k2}": str(v2)
            for k1, v1 in cfg.items()
            if isinstance(v1, dict)
            for k2, v2 in v1.items()
        }
    )

    # Create datasets
    transforms = build_transforms(cfg.data.transforms)
    train_dataset = Food101Dataset(
        cfg.data.dataset.root, split="train", transform=transforms["train"]
    )
    val_dataset = Food101Dataset(
        cfg.data.dataset.root, split="test", transform=transforms["test"]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.data.dataloader.num_workers,
        pin_memory=cfg.data.dataloader.pin_memory,
        drop_last=cfg.data.dataloader.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.data.dataloader.num_workers,
        pin_memory=cfg.data.dataloader.pin_memory,
    )

    # Create model
    model = FoodClassifier(cfg.model)
    model = model.cuda()

    # Create optimizer and scheduler
    optimizer = model.get_optimizer(cfg.model.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=cfg.model.scheduler.eta_min
    )

    # Initialize gradient scaler
    scaler = GradScaler()

    # Create checkpoint directory
    ckpt_dir = Path(cfg.checkpoints.dirpath)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0
    for epoch in range(cfg.training.epochs):
        log.info(f"Epoch {epoch+1}/{cfg.training.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, epoch, cfg)
        mlflow.log_metrics(
            {f"train/{k}": v for k, v in train_metrics.items()}, step=epoch
        )

        # Validate
        if (epoch + 1) % cfg.training.val_interval == 0:
            model.eval()
            val_metrics = {}
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.cuda()
                    targets = targets.cuda()
                    logits = model(images)
                    metrics = calculate_metrics(
                        logits, targets, k=cfg.model.metrics.top_k
                    )
                    for k, v in metrics.items():
                        val_metrics[k] = val_metrics.get(k, 0) + v

            # Average validation metrics
            val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
            mlflow.log_metrics(
                {f"val/{k}": v for k, v in val_metrics.items()}, step=epoch
            )

            # Save checkpoint if improved
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                ckpt_path = ckpt_dir / f"best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": best_val_acc,
                    },
                    ckpt_path,
                )
                mlflow.log_artifact(ckpt_path)

        # Update learning rate
        scheduler.step()

    mlflow.end_run()


if __name__ == "__main__":
    main()
