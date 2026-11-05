#!/usr/bin/env python3
"""Classify sample images with a trained CIFAR-10 ResNet checkpoint.

The demo assigns one outdoor-scene category to a whole image.
It does not draw boxes, track objects, or run detection.

Examples:
    python infer_demo.py --export-samples
    python infer_demo.py --checkpoint checkpoints/resnet18_best.pth --image samples/00_airplane.png
    python infer_demo.py --checkpoint checkpoints/resnet18_best.pth --samples-dir samples --save-grid artifacts/demo_grid.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from resnet_cifar.data import eval_transform, export_class_samples
from resnet_cifar.models import build_model
from resnet_cifar.scene import CIFAR10_CLASSES, scene_of
from resnet_cifar.utils import device_of, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 scene-category inference demo.")
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_best.pth")
    parser.add_argument("--image", action="append", default=None, help="Image path. Repeatable.")
    parser.add_argument("--samples-dir", default=None, help="Classify every PNG in this directory.")
    parser.add_argument("--export-samples", action="store_true", help="Write one CIFAR-10 test PNG per class.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-dir", default="samples")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--save-grid", default=None, help="Save a matplotlib contact sheet when classifying a directory.")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["model"]
    se_reduction = 16
    if checkpoint.get("args"):
        se_reduction = int(checkpoint["args"].get("se_reduction", 16))
    model = build_model(model_name, se_reduction=se_reduction).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_image(model, image_path: Path, device: torch.device, topk: int) -> dict:
    image = Image.open(image_path).convert("RGB")
    if image.size != (32, 32):
        image = image.resize((32, 32), Image.BILINEAR)
    tensor = eval_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    scores, indices = torch.topk(probabilities, k=min(topk, probabilities.numel()))
    top = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        name = CIFAR10_CLASSES[index]
        scene = scene_of(name)
        top.append(
            {
                "class_index": index,
                "class_name": name,
                "probability": score,
                "scene_zh": scene["zh"],
                "scene_en": scene["en"],
                "group_zh": scene["group_zh"],
            }
        )
    best = top[0]
    return {
        "image": str(image_path),
        "prediction": best["class_name"],
        "probability": best["probability"],
        "scene_zh": best["scene_zh"],
        "scene_en": best["scene_en"],
        "group_zh": best["group_zh"],
        "topk": top,
    }


def save_grid(results: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    columns = 5
    rows = (len(results) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.4, rows * 2.8))
    axes = axes.ravel() if len(results) > 1 else [axes]
    for axis in axes:
        axis.axis("off")
    for axis, result in zip(axes, results):
        image = Image.open(result["image"]).convert("RGB")
        axis.imshow(image, interpolation="nearest")
        axis.set_title(
            f"{result['prediction']}\n{result['scene_zh']}\n{result['probability']*100:.1f}%",
            fontsize=8,
        )
    figure.suptitle("CIFAR-10 场景类别推理示例", fontsize=12)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.export_samples:
        manifest = export_class_samples(args.data_dir, args.sample_dir)
        print(f"exported {len(manifest)} samples to {args.sample_dir}")
        if not args.image and not args.samples_dir:
            return

    paths: list[Path] = []
    if args.image:
        paths.extend(Path(item) for item in args.image)
    if args.samples_dir:
        paths.extend(sorted(Path(args.samples_dir).glob("*.png")))
    if not paths:
        raise SystemExit("Pass --image, --samples-dir, or --export-samples.")

    device = device_of()
    model, checkpoint = load_model(Path(args.checkpoint), device)
    results = [predict_image(model, path, device, args.topk) for path in paths]
    payload = {
        "checkpoint": args.checkpoint,
        "model": checkpoint["model"],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_test_acc": checkpoint.get("test_acc"),
        "task": "image classification into CIFAR-10 outdoor scene categories",
        "results": results,
    }
    for result in results:
        print(
            f"{result['image']}: {result['prediction']} "
            f"({result['scene_zh']}) p={result['probability']*100:.2f}%"
        )
    if args.save_json:
        write_json(Path(args.save_json), payload)
        print(f"wrote {args.save_json}")
    elif len(results) > 1:
        default_json = Path("artifacts/demo_predictions.json")
        write_json(default_json, payload)
        print(f"wrote {default_json}")
    else:
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    if args.save_grid:
        save_grid(results, Path(args.save_grid))
        print(f"wrote {args.save_grid}")


if __name__ == "__main__":
    main()
