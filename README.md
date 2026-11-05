# CIFAR-10 图像分类：ResNet-18 及其结构变体

# CIFAR-10 Image Classification: ResNet-18 and Structural Variants

本项目从零训练一个面向 CIFAR-10 的 ResNet-18，并在**同一训练配方**下比较三种结构改动：深度可分离卷积、SE 通道注意力、GELU 激活。任务是**整图分类**（每张图一个类别），不是目标检测，也不包含边界框或检测头。

This project trains a CIFAR-adapted ResNet-18 from scratch and compares three structural changes under one training recipe: depthwise separable convolutions, Squeeze-and-Excitation, and GELU. The task is **image classification** (one label per image), not object detection.

课程报告见 [`大作业报告.md`](大作业报告.md)。报告中的准确率、损失和耗时均来自 `runs/` 下的训练日志，而不是手写估计。

The course report is [`大作业报告.md`](大作业报告.md). Accuracies, losses, and timings in that report are copied from the logs under `runs/`.

## 模型 / Models

| 名称 | 说明 |
| --- | --- |
| `resnet18` | 基线。3×3 stride-1 stem，无 max-pool；四阶段各 2 个标准残差块（64-128-256-512）。 |
| `resnet18_dw` | 残差块内的 3×3 卷积换成深度可分离卷积（depthwise 3×3 + pointwise 1×1）。Stem 与 1×1 投影捷径保持普通卷积。 |
| `resnet18_se` | 在第二个 BN 之后、残差相加之前插入 SE 模块（reduction=16）。 |
| `resnet18_gelu` | 将 stem 与残差块中的 ReLU 换成 GELU，其余结构与基线相同。 |

## 环境 / Setup

需要 Python 3.10+。CIFAR-10 由 torchvision 自动下载到 `data/`（已写入 `.gitignore`）。官方站点有时很慢。也可以自行把 `cifar-10-python.tar.gz`（MD5 `c58f30108f718f92721af3b95e74349a`，170,498,071 字节）放到 `data/` 下，torchvision 会校验后直接使用。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

有 GPU 时训练脚本会自动使用 CUDA；没有 GPU 时使用 CPU。本仓库中提交的曲线和 `summary.json` 记录了实际使用的设备。

## 训练 / Train

仓库里已经有三份完整日志：`resnet18`、`resnet18_dw`、`resnet18_se`。配方是 SGD + Nesterov（动量 0.9）、权重衰减 `5e-4`、初始学习率 `0.1`、余弦退火周期 `T_max=24`、实际训练 **12 个 epoch**、batch size `128`、随机种子 `42`、随机裁剪与水平翻转。训练步用 bfloat16 autocast，测试准确率用 fp32 计算。机器没有 GPU，日志里的 `device` 为 `cpu`。

基线本来按 24 个 epoch 启动。测到第 11 个 epoch 时测试准确率已经到 84.52%，于是在第 12 个 epoch 结束后停止。深度可分离和 SE 直接训练 12 个 epoch，余弦周期同样是 `T_max=24`，因此第 k 轮的学习率与基线一致。`resnet18_gelu` 已在代码里实现，本次提交没有它的训练日志。下面的 `train_all.sh` 会连 GELU 一起重训。

按同一配方重跑（会覆盖 `runs/` 和 `checkpoints/`）：

```bash
EPOCHS=12 T_MAX=24 WORKERS=0 AMP=bf16 bash scripts/train_all.sh
```

单模型命令：

```bash
python train.py --model resnet18 --epochs 12 --t-max 24 --seed 42 --num-workers 0 --amp bf16
python train.py --model resnet18_dw --epochs 12 --t-max 24 --seed 42 --num-workers 0 --amp bf16
python train.py --model resnet18_se --epochs 12 --t-max 24 --seed 42 --num-workers 0 --amp bf16
python train.py --model resnet18_gelu --epochs 12 --t-max 24 --seed 42 --num-workers 0 --amp bf16
```

不传 `--amp bf16` 时训练步使用 fp32，其余配方不变。有 CUDA 时会自动用 GPU。CPU 上的卷积累加顺序不保证逐位复现，以仓库中的 `metrics.csv` 为准。

画曲线、复评并跑推理示例：

```bash
python plot_curves.py
python evaluate.py --all
python infer_demo.py --checkpoint checkpoints/resnet18_best.pth --samples-dir samples --save-json artifacts/demo_predictions.json --save-grid artifacts/demo_grid.png
```

中断后接着训（`--epochs` 和 `--t-max` 须与原来一致）：

```bash
python train.py --model resnet18_dw --epochs 12 --t-max 24 --seed 42 --num-workers 0 --amp bf16 --resume
```

只查看参数量与 MACs，不训练：

```bash
python train.py --model resnet18 --print-stats
python train.py --model resnet18_dw --print-stats
python train.py --model resnet18_se --print-stats
python train.py --model resnet18_gelu --print-stats
```

每个模型会写出：

- `runs/<model>/config.json`：配方与模型复杂度
- `runs/<model>/metrics.csv`、`runs/<model>/history.json`：逐 epoch 的训练/测试损失和准确率
- `runs/<model>/summary.json`：最佳测试准确率、对应 epoch、设备
- `checkpoints/<model>_best.pth`：测试准确率最高的权重（约数十 MB）
- `runs/<model>/last.pt`：含优化器状态的断点，已 gitignore

## 本次日志 / Logged results

数字来自 `runs/*/summary.json`，与 `runs/comparison.md` 一致。

| 模型 | 参数量 | MACs | 最佳测试准确率 |
| --- | ---: | ---: | ---: |
| ResNet-18 | 11,173,962 | 555,422,720 | 84.52%（第 11 轮） |
| ResNet-18-DW | 1,439,626 | 73,049,088 | 80.84%（第 12 轮） |
| ResNet-18-SE | 11,261,002 | 555,509,760 | 82.87%（第 12 轮） |

`evaluate.py` 复评这三个 checkpoint，与上表的差为 0。说明见 [`大作业报告.md`](大作业报告.md)。

## 评估 / Evaluate

在 CIFAR-10 测试集上重算准确率、每类准确率和混淆矩阵：

```bash
python evaluate.py --checkpoint checkpoints/resnet18_best.pth
python evaluate.py --all
```

结果写入 `runs/<model>/eval.json`。`checkpoint_logged_test_acc` 是训练时记录的准确率，`test_acc` 是本次复评结果，二者应当一致。

画曲线和对比表：

```bash
python plot_curves.py
```

输出 `artifacts/curves_*.png`、`artifacts/confusion_*.png` 以及 `runs/comparison.md`。

## 场景推理演示 / Scene demo

把 CIFAR-10 的 10 个类别解释为户外场景中的粗粒度目标（飞行器、地面车辆、动物、船只等），对整张图给出一个类别。演示只做分类。

导出每个类别一张 CIFAR-10 测试图（仓库里已经附带 `samples/`）：

```bash
python infer_demo.py --export-samples
```

对单张图分类：

```bash
python infer_demo.py --checkpoint checkpoints/resnet18_best.pth --image samples/00_airplane.png
```

对 `samples/` 下全部示例分类，并保存 JSON 与拼图：

```bash
python infer_demo.py \
  --checkpoint checkpoints/resnet18_best.pth \
  --samples-dir samples \
  --save-json artifacts/demo_predictions.json \
  --save-grid artifacts/demo_grid.png
```

非 32×32 的图片会先双线性缩放到 32×32，再使用与测试集相同的归一化。

## 复现 / Reproducibility

- 随机种子默认 `42`，并写入每个 `config.json`。
- 数据增强与归一化写在 `resnet_cifar/data.py`，训练、评估、演示共用测试预处理。
- 指标以「按样本加权」累计，不是对各 batch 准确率取平均。
- CPU 上的线程调度可能带来极小的数值差异；以仓库中的 `runs/*/metrics.csv` 为准核对报告表格。

## 测试 / Tests

```bash
python -m pytest tests/test_models.py
```

`pytest` 未列入训练依赖。只检查结构时也可以：

```bash
python -c "import tests.test_models as t; t.test_baseline_parameter_count_matches_standard_cifar_resnet18(); t.test_forward_shape_and_stem(); t.test_variants_change_cost_in_the_expected_direction(); print('ok')"
```

## 目录 / Layout

```
train.py  evaluate.py  infer_demo.py  plot_curves.py
resnet_cifar/          模型、数据、训练循环、MACs
runs/<model>/          训练日志（报告引用这些文件）
checkpoints/           各变体最佳权重
samples/               推理示例图
artifacts/             曲线、混淆矩阵、演示拼图
大作业报告.md           中文课程报告
```
