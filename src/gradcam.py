from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    class_idx: int
    confidence: float


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> GradCAMResult:
        self.model.eval()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, class_idx].item())

        self.model.zero_grad(set_to_none=True)
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError('Grad-CAM hooks did not capture activations/gradients.')

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return GradCAMResult(heatmap=cam, class_idx=class_idx, confidence=confidence)


def tensor_to_rgb_image(x: torch.Tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)) -> np.ndarray:
    image = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    mean_arr = np.array(mean)
    std_arr = np.array(std)
    image = image * std_arr + mean_arr
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError('Expected uint8 image array.')
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return overlay
