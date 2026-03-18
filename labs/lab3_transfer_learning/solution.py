from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
