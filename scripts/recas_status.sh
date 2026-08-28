#!/usr/bin/env bash
# recas_status.sh — fast progress/results for a DistillFSS grid on ReCaS.
# Reads the per-run logs directly over the shared FS (instant); does NOT call
# condor_history (which scans ReCaS' huge shared history file and is very slow).
#
# Usage:  scripts/recas_status.sh <grid-dir>          # e.g. out/2026-..._DistillDMTNetKVASIR
# Env:    RECAS_ALIAS (default recas)  RECAS_REMOTE (default DistillFSS)
set -euo pipefail
G="${1:?usage: recas_status.sh <grid-dir>}"
ALIAS="${RECAS_ALIAS:-recas}"; REMOTE="${RECAS_REMOTE:-DistillFSS}"

ssh -o BatchMode=yes "$ALIAS" "cd '$REMOTE' 2>/dev/null || exit 1
G='$G'
echo '== live queue (this user) =='
condor_q \$USER -nobatch 2>/dev/null | grep -E 'run_|Total for query' || echo '  (none queued/running)'
echo
echo '== per-run outcome (from logs) =='
n=\$(ls \$G/run_*.yaml 2>/dev/null | wc -l)
for i in \$(seq 0 \$((n-1))); do
  f=\$G/run_\$i.log
  [ -f \"\$f\" ] || { echo \"  run_\$i: <no log>\"; continue; }
  term=\$(grep -aoE 'exit-code [0-9]+' \"\$f\" | tail -1)
  err=\$(grep -aE 'Traceback|ModuleNotFound|CUDA error|RuntimeError' \"\$f\" | tail -1)
  fg=\$(grep -aE 'MulticlassJaccardIndex_fg [0-9.]+' \"\$f\" | grep -aoE '[0-9.]+\$' | tail -1)
  if [ -n \"\$fg\" ]; then metric=\"fg=\$(awk \"BEGIN{printf \\\"%.1f\\\", \$fg*100}\")\"; else metric=''; fi
  state=\"\${term:-RUNNING}\"; [ -n \"\$err\" ] && state=\"\$state  ERR:\${err:0:60}\"
  echo \"  run_\$i: \$state  \$metric\"
done"
