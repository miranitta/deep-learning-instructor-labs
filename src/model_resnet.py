from __future__ import annotations

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model