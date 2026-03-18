from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from data_loader import DataConfig, get_cifar10_loaders
from model_cnn import SimpleCNN
from model_resnet import build_resnet18
from utils import (
    ensure_dir,
    get_device,
    plot_history,
    save_checkpoint,
    save_history,
    set_seed,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def build_model(model_name: str, num_classes: int):
    if model_name == 'cnn':
        return SimpleCNN(num_classes=num_classes)
    if model_name == 'resnet':
        return build_resnet18(num_classes=num_classes)
    raise ValueError(f'Unknown model: {model_name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cnn', 'resnet'], default='cnn')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    train_loader, test_loader, classes = get_cifar10_loaders(DataConfig(batch_size=args.batch_size))
    model = build_model(args.model, num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    ensure_dir('results/models')
    ensure_dir('results/logs')
    ensure_dir('results/plots')

    history = []
    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, test_loader, criterion, device)
        scheduler.step()
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        print(
            f'Epoch {epoch}/{args.epochs} | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                Path('results/models') / f'{args.model}_best.pt',
                metadata={
                    'model_name': args.model,
                    'num_classes': len(classes),
                    'classes': classes,
                    'best_val_acc': best_val_acc,
                },
            )

    save_checkpoint(
        model,
        Path('results/models') / f'{args.model}_latest.pt',
        metadata={
            'model_name': args.model,
            'num_classes': len(classes),
            'classes': classes,
            'best_val_acc': best_val_acc,
        },
    )
    save_history(history, Path('results/logs') / f'{args.model}_history.csv', Path('results/logs') / f'{args.model}_history.json')
    plot_history(history, Path('results/plots') / f'{args.model}_training_curve.png')


if __name__ == '__main__':
    main()
