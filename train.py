#!/usr/bin/env python3
"""Train a CIFAR-10 ResNet-18 variant from scratch.

Example:
    python train.py --model resnet18 --epochs 16 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn

from resnet_cifar.data import build_loaders
from resnet_cifar.engine import run_epoch
from resnet_cifar.models import VARIANTS, build_model
from resnet_cifar.stats import model_complexity
from resnet_cifar.utils import device_of, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 ResNet-18 variants from scratch.")
    parser.add_argument("--model", choices=VARIANTS, default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--t-max",
        type=int,
        default=None,
        help="Cosine annealing period. Defaults to --epochs. The submitted comparison uses 24.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--amp",
        choices=("off", "bf16"),
        default="off",
        help="bf16 autocast for the training step only. Test accuracy stays fp32.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None, help="Defaults to runs/<model>.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--se-reduction", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/last.pt if it exists.")
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print parameter and MAC counts, then exit without training.",
    )
    return parser.parse_args()


def append_metrics(path: Path, row: dict, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_last(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")

    set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = device_of()
    output_dir = Path(args.output_dir or f"runs/{args.model}")
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, se_reduction=args.se_reduction).to(device)
    complexity = model_complexity(model)
    print(
        f"model={args.model} device={device} params={complexity['params']} "
        f"macs={complexity['macs']} flops={complexity['flops']}"
    )
    if args.print_stats:
        return

    train_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.seed
    )
    criterion = nn.CrossEntropyLoss()
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    t_max = args.epochs if args.t_max is None else args.t_max
    if t_max < 1:
        raise SystemExit("--t-max must be >= 1")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    config = {
        "model": args.model,
        "epochs": args.epochs,
        "t_max": t_max,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "nesterov": True,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "se_reduction": args.se_reduction,
        "optimizer": "SGD",
        "scheduler": "CosineAnnealingLR",
        "scheduler_step": "per-epoch, after the test evaluation",
        "loss": "CrossEntropyLoss",
        "amp": args.amp,
        "amp_note": "Training step only. Evaluation is fp32 so checkpoints match evaluate.py.",
        "augmentation": ["RandomCrop(32, padding=4)", "RandomHorizontalFlip"],
        "normalize_mean": [0.4914, 0.4822, 0.4465],
        "normalize_std": [0.2023, 0.1994, 0.2010],
        "stem": "Conv3x3 stride 1, no max-pool",
        "init": "kaiming_normal fan_out relu; last BN gamma of each block set to 0",
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "complexity": complexity,
    }
    write_json(output_dir / "config.json", config)

    history: list[dict] = []
    start_epoch = 1
    best_acc = -1.0
    best_epoch = 0
    last_path = output_dir / "last.pt"
    metrics_path = output_dir / "metrics.csv"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_acc = checkpoint["best_test_acc"]
        best_epoch = checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"resumed at epoch {start_epoch} best_test_acc={best_acc:.4f}")
    elif metrics_path.exists() and start_epoch == 1:
        metrics_path.unlink()

    best_path = checkpoint_dir / f"{args.model}_best.pth"
    train_started = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_stats = run_epoch(
            model, train_loader, criterion, device, optimizer, amp_dtype=amp_dtype
        )
        test_stats = run_epoch(model, test_loader, criterion, device, optimizer=None)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": f"{lr:.8f}",
            "train_loss": f"{train_stats['loss']:.6f}",
            "train_acc": f"{train_stats['acc']:.6f}",
            "test_loss": f"{test_stats['loss']:.6f}",
            "test_acc": f"{test_stats['acc']:.6f}",
            "train_seconds": f"{train_stats['seconds']:.3f}",
            "test_seconds": f"{test_stats['seconds']:.3f}",
        }
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "test_loss": test_stats["loss"],
                "test_acc": test_stats["acc"],
                "train_seconds": train_stats["seconds"],
                "test_seconds": test_stats["seconds"],
            }
        )
        append_metrics(metrics_path, row, write_header=epoch == 1 and start_epoch == 1)
        write_json(output_dir / "history.json", history)
        improved = test_stats["acc"] > best_acc
        if improved:
            best_acc = test_stats["acc"]
            best_epoch = epoch
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_stats["acc"],
                    "test_loss": test_stats["loss"],
                    "args": vars(args),
                    "complexity": complexity,
                },
                best_path,
            )
        save_last(
            last_path,
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_test_acc": best_acc,
                "best_epoch": best_epoch,
                "history": history,
            },
        )
        print(
            f"epoch {epoch:03d}/{args.epochs} lr={lr:.5f} "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']*100:.2f}% "
            f"test_loss={test_stats['loss']:.4f} test_acc={test_stats['acc']*100:.2f}% "
            f"{'*' if improved else ''}"
        )

    total_seconds = time.perf_counter() - train_started
    summary = {
        "model": args.model,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "best_test_acc_percent": best_acc * 100.0,
        "final_test_acc": history[-1]["test_acc"] if history else None,
        "final_test_loss": history[-1]["test_loss"] if history else None,
        "final_train_acc": history[-1]["train_acc"] if history else None,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "total_train_seconds_this_process": total_seconds,
        "device": str(device),
        "torch": torch.__version__,
        "complexity": complexity,
        "checkpoint": str(best_path),
        "metrics_csv": str(metrics_path),
        "history_json": str(output_dir / "history.json"),
    }
    write_json(output_dir / "summary.json", summary)
    print(
        f"done {args.model}: best_test_acc={best_acc*100:.2f}% at epoch {best_epoch} "
        f"checkpoint={best_path}"
    )


if __name__ == "__main__":
    main()
