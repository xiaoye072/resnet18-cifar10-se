"""Outdoor-scene reading of the ten CIFAR-10 categories.

The mapping is an application framing for image-level classification.
It does not localize, track, or detect objects inside an image.
"""

from __future__ import annotations

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Coarse outdoor categories used by the inference demo.
SCENE = {
    "airplane": {
        "zh": "空中目标 / 飞行器",
        "en": "airborne aircraft",
        "group_zh": "空中",
        "group_en": "air",
    },
    "automobile": {
        "zh": "地面车辆 / 小型机动车",
        "en": "ground light vehicle",
        "group_zh": "地面车辆",
        "group_en": "ground_vehicle",
    },
    "bird": {
        "zh": "低空生物 / 鸟类",
        "en": "low-altitude bird",
        "group_zh": "动物",
        "group_en": "animal",
    },
    "cat": {
        "zh": "地面动物 / 猫",
        "en": "ground animal (cat)",
        "group_zh": "动物",
        "group_en": "animal",
    },
    "deer": {
        "zh": "地面动物 / 鹿",
        "en": "ground animal (deer)",
        "group_zh": "动物",
        "group_en": "animal",
    },
    "dog": {
        "zh": "地面动物 / 犬",
        "en": "ground animal (dog)",
        "group_zh": "动物",
        "group_en": "animal",
    },
    "frog": {
        "zh": "近地小型生物 / 蛙",
        "en": "ground small fauna (frog)",
        "group_zh": "动物",
        "group_en": "animal",
    },
    "horse": {
        "zh": "地面动物 / 马",
        "en": "ground animal (horse)",
        "group_zh": "动物",
        "group_en": "animal",
    },
    "ship": {
        "zh": "水面目标 / 船只",
        "en": "surface vessel",
        "group_zh": "水面",
        "group_en": "water",
    },
    "truck": {
        "zh": "地面车辆 / 货车",
        "en": "ground heavy vehicle",
        "group_zh": "地面车辆",
        "group_en": "ground_vehicle",
    },
}


def scene_of(class_name: str) -> dict:
    if class_name not in SCENE:
        raise KeyError(f"Unknown CIFAR-10 class: {class_name}")
    info = dict(SCENE[class_name])
    info["class_name"] = class_name
    info["class_index"] = CIFAR10_CLASSES.index(class_name)
    return info
