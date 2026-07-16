#!/usr/bin/env bash
# =============================================================================
# Industrial fold-0 runs for gandalf (single big GPU, reap-free).
#
# STATUS: SegIC, INSID3 baseline, and DistillFSS-DCAMA (swin all 4 + resnet50
# 10/20/40) already ran here and synced to wandb. Only DistillFSS-PAHNet is
# left (the earlier run aborted when DCAMA-resnet50-80 OOM'd). The OOM runs go
# on Leonardo (scripts/run_industrial_fold0_leonardo.sh).
#
#   DistillFSS-PAHNet  teacher pahnet, student pahnet_distillator (num_classes 5),
#                      @473, 2500 iters, substitutor null -> fits 24GB   (4 runs)
#
# Runs log to wandb (project FSSWeed, group Industrial) so results/1_download
# + 2_filter regenerate Industrial.csv.
#
# Usage (repo root on gandalf):  bash scripts/run_industrial_fold0_gandalf.sh
# Append --resume to continue after an interruption.  || true keeps one failure
# from aborting the rest.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUN() { echo "==> $*"; uv run python refine.py grid --parameters "$1" || echo "FAILED: $1"; }

RUN parameters/distill/pahnet/Industrial.yaml   # DistillFSS-PAHNet

echo "gandalf Industrial fold-0 runs complete."
