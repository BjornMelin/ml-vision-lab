from pathlib import Path
import sys
import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.model import Model
from src.data.utils import create_dataloaders
from src.utils.metrics import accuracy, top_k_accuracy

def evaluate():
    # Load config
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load model
    model = Model(num_classes=config["model"]["num_classes"])
    model.load("experiments/models/model.pt")
    model.eval()

    # Create dataloader
    dataloaders = create_dataloaders(
        config["data"]["root"],
        batch_size=config["train"]["batch_size"]
    )
    val_loader = dataloaders['val']

    # Evaluate
    total_acc = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            acc = accuracy(output, target)
            total_acc += acc * len(target)
            total_samples += len(target)

    print(f"Validation Accuracy: {total_acc/total_samples:.4f}")

if __name__ == "__main__":
    evaluate()
