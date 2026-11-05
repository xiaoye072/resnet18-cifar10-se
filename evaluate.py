#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the CIFAR-10 test set.

Writes overall accuracy, per-class accuracy, and a confusion matrix.
Numbers are computed from the checkpoint; nothing is hardcoded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from resnet_cifar.data import build_loaders
from resnet_cifar.engine import confusion_matrix, run_epoch
from resnet_cifar.models import VARIANTS, build_model
from resnet_cifar.scene import CIFAR10_CLASSES
from resnet_cifar.utils import device_of, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 ResNet checkpoint.")
    parser.add_argument("--model", choices=VARIANTS, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--all", action="store_true", help="Evaluate every checkpoints/*_best.pth.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Eval JSON path. Defaults to runs/<model>/eval.json.")
    return parser.parse_args()


def evaluate_one(checkpoint_path: Path, args: argparse.Namespace) -> dict:
    device = device_of()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = args.model or checkpoint["model"]
    se_reduction = 16
    if "args" in checkpoint and checkpoint["args"]:
        se_reduction = int(checkpoint["args"].get("se_reduction", 16))
    model = build_model(model_name, se_reduction=se_reduction).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    _, test_loader = build_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.seed,
        download=True,
    )
    criterion = nn.CrossEntropyLoss()
    stats = run_epoch(model, test_loader, criterion, device, optimizer=None)
    matrix = confusion_matrix(model, test_loader, device, num_classes=10)
    per_class = []
    for index, name in enumerate(CIFAR10_CLASSES):
        support = sum(matrix[index])
        correct = matrix[index][index]
        per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "support": support,
                "correct": correct,
                "acc": correct / support if support else 0.0,
            }
        )
    payload = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_logged_test_acc": checkpoint.get("test_acc"),
        "test_loss": stats["loss"],
        "test_acc": stats["acc"],
        "test_acc_percent": stats["acc"] * 100.0,
        "examples": stats["examples"],
        "per_class": per_class,
        "confusion_matrix": matrix,
        "device": str(device),
        "classes": CIFAR10_CLASSES,
    }
    output = Path(args.output) if args.output and not args.all else Path("runs") / model_name / "eval.json"
    write_json(output, payload)
    logged = checkpoint.get("test_acc")
    match = ""
    if isinstance(logged, float):
        delta = abs(logged - stats["acc"])
        match = f" delta_vs_logged={delta:.6f}"
    print(
        f"{model_name}: test_acc={stats['acc']*100:.2f}% "
        f"test_loss={stats['loss']:.4f} examples={stats['examples']}{match} -> {output}"
    )
    return payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.all:
        paths = sorted(Path(args.checkpoint_dir).glob("*_best.pth"))
        if not paths:
            raise SystemExit(f"No checkpoints in {args.checkpoint_dir}")
        for path in paths:
            evaluate_one(path, args)
        return
    if not args.checkpoint:
        if not args.model:
            raise SystemExit("Pass --checkpoint, or --model with --checkpoint-dir, or --all.")
        args.checkpoint = str(Path(args.checkpoint_dir) / f"{args.model}_best.pth")
    evaluate_one(Path(args.checkpoint), args)


if __name__ == "__main__":
    main()
