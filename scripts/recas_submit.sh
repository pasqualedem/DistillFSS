#!/usr/bin/env bash
# recas_submit.sh — launch a DistillFSS grid on ReCaS-Bari (HTCondor) in one command.
#
# Why not just `refine.py grid --scheduler condor` on ReCaS? The ReCaS frontend SIGKILLs
# `import torch` (login-node memory cap), so the grid *driver* can't run there. This script
# does the torch-heavy expansion LOCALLY, ships the tiny run configs, and condor_submits
# each run on the frontend (submit is lightweight). See ~/recas/llm_readme.md.
#
# Usage:  scripts/recas_submit.sh <grid.yaml> [--no-code-sync]
# Env:    RECAS_ALIAS (default: recas)   RECAS_REMOTE (default: DistillFSS)
#         PY (default: .venv/bin/python)
# Prereq: the grid's datasets live in ReCaS ~/$RECAS_REMOTE/data/ and its checkpoints in
#         ~/$RECAS_REMOTE/checkpoints/ (stage once; reuse ~/FSSAffinityExplainer/checkpoints).
set -euo pipefail

PARAMS="${1:?usage: recas_submit.sh <grid.yaml> [--no-code-sync]}"
CODE_SYNC=1; [ "${2:-}" = "--no-code-sync" ] && CODE_SYNC=0
ALIAS="${RECAS_ALIAS:-recas}"
REMOTE="${RECAS_REMOTE:-DistillFSS}"
PY="${PY:-.venv/bin/python}"
SSH="ssh -o BatchMode=yes"
RSH="$SSH"

echo ">> [1/3] expanding grid locally (--only_create)…"
log=$(mktemp)
"$PY" refine.py grid --parameters "$PARAMS" --parallel --scheduler condor --only_create >"$log" 2>&1 || { cat "$log"; rm -f "$log"; exit 1; }
G=$(grep -oaE 'out/[^/ ]+/run_0\.yaml' "$log" | head -1 | xargs -r dirname)
rm -f "$log"
[ -n "${G:-}" ] && [ -d "$G" ] || { echo "ERROR: could not locate created grid dir"; exit 1; }
N=$(ls "$G"/run_*.yaml 2>/dev/null | wc -l)
[ "$N" -gt 0 ] || { echo "ERROR: no run_*.yaml in $G"; exit 1; }
echo "   grid = $G   ($N runs)"

echo ">> [2/3] shipping to ReCaS ($ALIAS:$REMOTE)…"
if [ "$CODE_SYNC" = 1 ]; then
  # incremental code sync (small, fast). Anchored excludes so code dirs named data/logs/
  # wandb/out are NOT dropped (unanchored patterns match distillfss/data etc.).
  rsync -az \
    --exclude='/.git' --exclude='/.venv' --exclude='/out' --exclude='/data' \
    --exclude='/checkpoints' --exclude='/tmp' --exclude='/wandb' --exclude='/preds' \
    --exclude='/artifacts' --exclude='/logs' --exclude='/notebooks' \
    --exclude='.ipynb_checkpoints' --exclude='*.zip' --exclude='*.pt' --exclude='*.pth' \
    --exclude='*.ipynb' -e "$RSH" ./ "$ALIAS:$REMOTE/"
fi
$SSH "$ALIAS" "mkdir -p '$REMOTE/$G'"
rsync -az -e "$RSH" "$G/" "$ALIAS:$REMOTE/$G/"

echo ">> [3/3] submitting $N jobs on the frontend…"
$SSH "$ALIAS" "cd '$REMOTE' && for i in \$(seq 0 $((N-1))); do \
  condor_submit output=$G/run_\$i.log error=$G/run_\$i.log log=$G/run_\$i.log \
    \"arguments=refine.py run --parameters=$G/run_\$i.yaml --disable_log_params --disable_log_model --disable_log_on_file\" \
    slurm/condor 2>&1 | grep -E 'submitted to cluster'; done"

echo ">> done. track with:  scripts/recas_status.sh $G"
