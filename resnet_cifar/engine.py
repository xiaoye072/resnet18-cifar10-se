"""Train and evaluate loops. Metrics are accumulated over examples, not batches."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """One pass over ``loader``.

    ``amp_dtype`` applies only to the training step. Test metrics are
    computed in fp32 so a saved checkpoint matches ``evaluate.py``.
    """
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    use_amp = training and amp_dtype is not None
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(images)
                    loss = criterion(logits, targets)
            else:
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            batch = targets.size(0)
            loss_sum += loss.item() * batch
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += batch
    elapsed = time.perf_counter() - start
    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "seconds": elapsed,
        "examples": total,
    }


@torch.inference_mode()
def confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    model.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1).cpu()
        for truth, pred in zip(targets.tolist(), preds.tolist()):
            matrix[truth][pred] += 1
    return matrix
