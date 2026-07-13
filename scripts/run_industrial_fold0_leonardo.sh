#!/usr/bin/env bash
# =============================================================================
# Industrial fold-0 runs that OOM on 24GB -> Leonardo (bigger GPUs / SLURM).
#
#   DCAMA baseline 40/80-shot      -- voting=10 needs >21.6GB   (4 runs)
#   DistillFSS-DCAMA resnet50 80   -- OOMs at step 0 on 24GB    (1 run)
#     (swin all 4 + resnet50 10/20/40 already ran on the dev box; only this
#      one distill cell OOMs. Its swin-80 got 59.23.)
#
# --parallel submits each run as its own SLURM job (one big GPU each).
# Usage (repo root, Leonardo env):  bash scripts/run_industrial_fold0_leonardo.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUN() { echo "==> $*"; uv run python refine.py grid --parameters "$1" --parallel || echo "FAILED: $1"; }

RUN parameters/baselines/industrial_fold0/DCAMA_highshot.yaml  # baseline dcama 40/80
RUN parameters/distill/industrial_fold0/DCAMA_rn50_80.yaml     # DistillFSS-DCAMA rn50 80

echo "Leonardo Industrial fold-0 OOM runs submitted."
