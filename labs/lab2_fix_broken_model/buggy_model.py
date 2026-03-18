import torch
from torch import nn


class BuggyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.relu = nn.Sigmoid()
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(16 * 10 * 10, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    x = torch.randn(8, 3, 32, 32)
    model = BuggyCNN()
    print(model(x).shape)
