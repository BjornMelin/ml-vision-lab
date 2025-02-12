import torch
import torch.nn as nn
import timm

class EfficientNetModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
        super(EfficientNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
