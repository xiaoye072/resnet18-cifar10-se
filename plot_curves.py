#!/usr/bin/env python3
"""Plot training curves and write a comparison table from logged metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from resnet_cifar.models import VARIANTS
from resnet_cifar.utils import read_json, write_json

DISPLAY = {
    "resnet18": "ResNet-18",
    "resnet18_dw": "ResNet-18-DW",
    "resnet18_se": "ResNet-18-SE",
    "resnet18_gelu": "ResNet-18-GELU",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot curves and summarize run logs.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="artifacts")
    return parser.parse_args()


def load_histories(runs_dir: Path) -> dict[str, list[dict]]:
    histories = {}
    for name in VARIANTS:
        path = runs_dir / name / "history.json"
        if path.exists():
            histories[name] = read_json(path)
    return histories


def load_summaries(runs_dir: Path) -> list[dict]:
    rows = []
    for name in VARIANTS:
        path = runs_dir / name / "summary.json"
        if path.exists():
            rows.append(read_json(path))
    return rows


def comparison_rows(summaries: list[dict]) -> list[dict]:
    baseline = next((row for row in summaries if row["model"] == "resnet18"), None)
    table = []
    for row in summaries:
        complexity = row["complexity"]
        item = {
            "model": row["model"],
            "display": DISPLAY.get(row["model"], row["model"]),
            "params": complexity["params"],
            "macs": complexity["macs"],
            "flops": complexity["flops"],
            "best_epoch": row["best_epoch"],
            "best_test_acc": row["best_test_acc"],
            "final_test_acc": row["final_test_acc"],
            "final_test_loss": row["final_test_loss"],
            "epochs_completed": row["epochs_completed"],
            "device": row["device"],
            "seed": row["seed"],
        }
        if baseline is not None:
            item["delta_best_acc_vs_baseline"] = row["best_test_acc"] - baseline["best_test_acc"]
            item["param_ratio_vs_baseline"] = complexity["params"] / baseline["complexity"]["params"]
            item["mac_ratio_vs_baseline"] = complexity["macs"] / baseline["complexity"]["macs"]
        table.append(item)
    return table


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    header = (
        "| 模型 | 参数量 | MACs | FLOPs | 最佳测试准确率 | 最佳 epoch | 末轮测试准确率 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {display} | {params:,} | {macs:,} | {flops:,} | {acc:.2f}% | {epoch} | {final:.2f}% |\n".format(
                display=row["display"],
                params=row["params"],
                macs=row["macs"],
                flops=row["flops"],
                acc=row["best_test_acc"] * 100.0,
                epoch=row["best_epoch"],
                final=row["final_test_acc"] * 100.0,
            )
        )
    path.write_text("".join(lines), encoding="utf-8")


def plot_confusion(runs_dir: Path, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    for name in VARIANTS:
        path = runs_dir / name / "eval.json"
        if not path.exists():
            continue
        payload = read_json(path)
        matrix = np.array(payload["confusion_matrix"], dtype=float)
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = matrix / np.maximum(totals, 1.0)
        figure, axis = plt.subplots(figsize=(7.2, 6.2))
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(10), payload["classes"], rotation=45, ha="right")
        axis.set_yticks(range(10), payload["classes"])
        axis.set_xlabel("预测")
        axis.set_ylabel("真实")
        axis.set_title(f"{DISPLAY[name]} 测试集混淆矩阵")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(out_dir / f"confusion_{name}.png", dpi=140)
        plt.close(figure)


def plot(histories: dict[str, list[dict]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    if not histories:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "resnet18": "#1f77b4",
        "resnet18_dw": "#ff7f0e",
        "resnet18_se": "#2ca02c",
        "resnet18_gelu": "#d62728",
    }

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        acc = [row["test_acc"] * 100.0 for row in history]
        axis.plot(epochs, acc, label=DISPLAY[name], color=colors[name], linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("测试准确率 (%)")
    axis.set_title("CIFAR-10 测试准确率")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "curves_test_acc.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axis.plot(
            epochs,
            [row["train_loss"] for row in history],
            label=DISPLAY[name],
            color=colors[name],
            linewidth=2,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("训练损失")
    axis.set_title("CIFAR-10 训练损失")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "curves_train_loss.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axis.plot(
            epochs,
            [row["test_loss"] for row in history],
            label=DISPLAY[name],
            color=colors[name],
            linewidth=2,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("测试损失")
    axis.set_title("CIFAR-10 测试损失")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "curves_test_loss.png", dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    histories = load_histories(runs_dir)
    summaries = load_summaries(runs_dir)
    rows = comparison_rows(summaries)
    if rows:
        write_json(runs_dir / "comparison.json", rows)
        write_csv(runs_dir / "comparison.csv", rows)
        write_markdown(runs_dir / "comparison.md", rows)
        print(f"wrote {runs_dir / 'comparison.json'}")
    plot(histories, out_dir)
    plot_confusion(runs_dir, out_dir)
    if histories:
        print(f"wrote curves to {out_dir}")
    if not histories and not rows:
        raise SystemExit("No history.json or summary.json found under runs/.")


if __name__ == "__main__":
    main()
