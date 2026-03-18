from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import DataConfig, get_cifar10_loaders
from model_cnn import SimpleCNN
from model_resnet import build_resnet18
from utils import get_device, load_checkpoint, load_checkpoint_payload


def build_model(model_name: str, num_classes: int):
    if model_name == 'cnn':
        return SimpleCNN(num_classes=num_classes)
    if model_name == 'resnet':
        return build_resnet18(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    raise ValueError(f'Unsupported model: {model_name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model', choices=['cnn', 'resnet'], default=None)
    args = parser.parse_args()

    device = get_device()
    _, test_loader, default_classes = get_cifar10_loaders(DataConfig())

    payload = load_checkpoint_payload(args.model_path, map_location=device)
    metadata = payload.get('metadata', {})
    model_name = args.model or metadata.get('model_name', 'cnn')
    classes = metadata.get('classes', default_classes)
    num_classes = int(metadata.get('num_classes', len(classes)))

    model = build_model(model_name, num_classes=num_classes).to(device)
    model = load_checkpoint(model, args.model_path, map_location=device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    print('Model:', model_name)
    print(classification_report(y_true, y_pred, target_names=classes))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    out_path = Path('results/logs') / f"evaluation_{Path(args.model_path).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f'Saved evaluation report to {out_path}')


if __name__ == '__main__':
    main()
