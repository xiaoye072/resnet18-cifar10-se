"""CIFAR-10 loaders and the fixed preprocessing used by train, eval, and demo."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet_cifar.scene import CIFAR10_CLASSES
from resnet_cifar.utils import seed_worker

# Standard CIFAR-10 channel statistics.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_datasets(data_dir: str | Path, download: bool = True):
    root = str(data_dir)
    train_set = datasets.CIFAR10(
        root=root, train=True, download=download, transform=train_transform()
    )
    test_set = datasets.CIFAR10(
        root=root, train=False, download=download, transform=eval_transform()
    )
    return train_set, test_set


def build_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_set, test_set = build_datasets(data_dir, download=download)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def export_class_samples(data_dir: str | Path, out_dir: str | Path) -> list[dict]:
    """Save the first test image of each class as a 32x32 PNG.

    Filenames and the manifest are deterministic so the demo inputs
    committed in the repo match a known CIFAR-10 test index.
    """
    dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=None)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    found: dict[int, tuple[int, object]] = {}
    for index, (image, label) in enumerate(dataset):
        if label not in found:
            found[int(label)] = (index, image)
        if len(found) == len(CIFAR10_CLASSES):
            break

    manifest = []
    for label in range(len(CIFAR10_CLASSES)):
        index, image = found[label]
        name = CIFAR10_CLASSES[label]
        filename = f"{label:02d}_{name}.png"
        image.save(destination / filename)
        manifest.append(
            {
                "file": filename,
                "cifar10_test_index": index,
                "label_index": label,
                "label_name": name,
            }
        )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
