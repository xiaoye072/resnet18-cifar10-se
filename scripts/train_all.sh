#!/usr/bin/env bash
# Train the baseline and the three structural variants with one recipe.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

EPOCHS="${EPOCHS:-12}"
T_MAX="${T_MAX:-24}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-0}"
AMP="${AMP:-bf16}"

for model in resnet18 resnet18_dw resnet18_se resnet18_gelu; do
  python train.py \
    --model "$model" \
    --epochs "$EPOCHS" \
    --t-max "$T_MAX" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$WORKERS" \
    --amp "$AMP" \
    --output-dir "runs/${model}" \
    --checkpoint-dir checkpoints
done

python plot_curves.py --runs-dir runs --out-dir artifacts
python evaluate.py --all --data-dir data --num-workers "$WORKERS"
python infer_demo.py --export-samples --data-dir data --sample-dir samples
python infer_demo.py \
  --checkpoint checkpoints/resnet18_best.pth \
  --samples-dir samples \
  --save-json artifacts/demo_predictions.json \
  --save-grid artifacts/demo_grid.png
