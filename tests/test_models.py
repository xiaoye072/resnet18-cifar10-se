"""Sanity checks for the CIFAR ResNet-18 skeleton and the three variants."""

import torch

from resnet_cifar.models import BasicBlock, build_model
from resnet_cifar.stats import model_complexity


def test_baseline_parameter_count_matches_standard_cifar_resnet18():
    model = build_model("resnet18")
    stats = model_complexity(model)
    assert stats["params"] == 11_173_962


def test_forward_shape_and_stem():
    model = build_model("resnet18")
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert not any(isinstance(module, torch.nn.MaxPool2d) for module in model.modules())
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)


def test_variants_change_cost_in_the_expected_direction():
    baseline = model_complexity(build_model("resnet18"))
    depthwise = model_complexity(build_model("resnet18_dw"))
    squeeze = model_complexity(build_model("resnet18_se"))
    gelu = model_complexity(build_model("resnet18_gelu"))
    assert depthwise["params"] < baseline["params"]
    assert depthwise["macs"] < baseline["macs"]
    assert squeeze["params"] > baseline["params"]
    assert squeeze["macs"] > baseline["macs"]
    assert gelu["params"] == baseline["params"]
    assert gelu["macs"] == baseline["macs"]


def test_each_stage_has_two_basic_blocks():
    model = build_model("resnet18_se")
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        blocks = [module for module in stage if isinstance(module, BasicBlock)]
        assert len(blocks) == 2
        assert all(not isinstance(block.se, torch.nn.Identity) for block in blocks)
