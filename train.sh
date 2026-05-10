#!/bin/bash
# Training script for bimanual SO-101 donut packing ACT policy.
#
# Usage:
#   bash train.sh                        # defaults
#   bash train.sh --steps 50000          # override steps
#   SLURM: sbatch --gres=gpu:1 train.sh

#SBATCH --job-name=act_bi_so101
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATASET_NAME="${DATASET_NAME:-local/bi_so101_donut_packing}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/data/${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/train/act_bi_so101_donut}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-8}"

conda activate act 2>/dev/null || true

lerobot-train \
  --policy.type=act \
  --dataset.repo_id="${DATASET_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.image_transforms.enable=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.vision_backbone=resnet18 \
  --policy.dim_model=512 \
  --policy.use_vae=true \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval_freq=5000 \
  --save_freq=5000 \
  --log_freq=100 \
  --output_dir="${OUTPUT_DIR}" \
  --wandb.enable=false \
  "$@"
