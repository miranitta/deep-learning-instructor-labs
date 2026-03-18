from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    batch_size: int = 64
    image_size: int = 32
    num_workers: int = 2


def get_cifar10_loaders(config: DataConfig):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomCrop(config.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_ds = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    test_ds = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, test_loader, train_ds.classes