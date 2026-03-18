from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    ensure_dir(Path(path).parent)
    payload = {
        'state_dict': model.state_dict(),
        'metadata': metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, map_location=None) -> torch.nn.Module:
    payload = torch.load(path, map_location=map_location)
    state_dict = payload['state_dict'] if isinstance(payload, dict) and 'state_dict' in payload else payload
    model.load_state_dict(state_dict)
    return model


def load_checkpoint_payload(path: str | Path, map_location=None) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and 'state_dict' in payload:
        return payload
    return {'state_dict': payload, 'metadata': {}}


def save_history(history: list[dict[str, Any]], csv_path: str | Path, json_path: str | Path) -> None:
    ensure_dir(Path(csv_path).parent)
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)
    Path(json_path).write_text(json.dumps(history, indent=2))


def plot_history(history: list[dict[str, Any]], out_path: str | Path) -> None:
    df = pd.DataFrame(history)
    ensure_dir(Path(out_path).parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df['epoch'], df['train_loss'], label='train_loss')
    axes[0].plot(df['epoch'], df['val_loss'], label='val_loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(df['epoch'], df['train_acc'], label='train_acc')
    axes[1].plot(df['epoch'], df['val_acc'], label='val_acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
