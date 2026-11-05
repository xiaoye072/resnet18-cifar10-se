"""Parameter and multiply-accumulate counts.

MACs count multiply-accumulates in Conv2d and Linear only.
Batch-norm, pooling, activations, and elementwise adds are omitted.
FLOPs are reported as 2 * MACs (one multiply and one add).
The count is for a single 3x32x32 image.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_macs(model: nn.Module, image_size: int = 32) -> int:
    macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, inputs, output: torch.Tensor) -> None:
        nonlocal macs
        _, _, height, width = output.shape
        kernel_h, kernel_w = module.kernel_size
        macs += (
            (module.in_channels // module.groups)
            * kernel_h
            * kernel_w
            * module.out_channels
            * height
            * width
        )

    def linear_hook(module: nn.Linear, inputs, output: torch.Tensor) -> None:
        nonlocal macs
        macs += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size))
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    return int(macs)


def model_complexity(model: nn.Module, image_size: int = 32) -> dict:
    params = count_parameters(model)
    macs = count_macs(model, image_size=image_size)
    return {
        "params": params,
        "macs": macs,
        "flops": macs * 2,
        "image_size": image_size,
        "macs_note": "Conv2d and Linear multiply-accumulates for one image; BN and activations excluded.",
        "flops_note": "FLOPs = 2 * MACs.",
    }
