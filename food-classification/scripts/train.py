import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from src.data.dataset import FoodDataset
from src.data.transforms import train_transforms, val_transforms
from src.models.efficientnet import EfficientNetModel
from src.utils.metrics import accuracy, cross_entropy_loss

@hydra.main(config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Training entry point with Hydra configuration"""
    # Initialize dataset and dataloaders
    train_dataset = FoodDataset(cfg.data.train_dir, transform=train_transforms)
    val_dataset = FoodDataset(cfg.data.val_dir, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    # Initialize model, optimizer, and loss function
    model = EfficientNetModel(cfg.model.architecture, num_classes=cfg.model.num_classes, pretrained=cfg.model.pretrained, freeze_backbone=cfg.model.freeze_backbone)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.model.optimizer.weight_decay)
    criterion = cross_entropy_loss

    # Training loop
    device = torch.device(cfg.train.device)
    model.to(device)
    best_val_acc = 0.0

    for epoch in range(cfg.train.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % cfg.train.log_interval == 0:
                print(f"Epoch [{epoch+1}/{cfg.train.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation loop
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_acc += accuracy(outputs, labels)
        val_acc /= len(val_loader)
        print(f"Epoch [{epoch+1}/{cfg.train.epochs}], Validation Accuracy: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if cfg.train.save_model:
                torch.save(model.state_dict(), cfg.train.save_path)
                print(f"Model saved to {cfg.train.save_path}")

if __name__ == "__main__":
    main()
