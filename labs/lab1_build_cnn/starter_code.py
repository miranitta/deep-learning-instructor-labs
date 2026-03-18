import torch
from torch import nn


class StudentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # TODO: add Conv2d -> ReLU -> MaxPool2d blocks
        )
        self.classifier = nn.Sequential(
            # TODO: flatten and classify
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    x = torch.randn(8, 3, 32, 32)
    model = StudentCNN()
    print(model(x).shape)
