#!/usr/bin/env bash
# Trains every agent in the paper at its tuned settings (10 seeds, 100 episodes) and compares it with the organoids.
# Usage (from organoids-study/, venv active, dataset in data/): bash scripts/paper_runs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -I scripts/organoid_runs.py data/aggregated_experiment_data.pkl
#     agent        lr     episodes/update
RUNS="mlp          3e-3   4
      gru          3e-3   1
      snn          3e-4   1
      snn5         3e-3   4
      snn10        3e-3   4
      snn5x2       3e-3   4
      res5x2       3e-3   4
      snn_out0     1e-3   4
      snn5_out0    3e-3   4
      snn10_out0   3e-3   4
      snn5x2_out0  3e-3   4
      res5x2_out0  3e-3   4
      if_mlp       3e-3   4
      if_snn5      3e-3   4
      if_gru       1e-2   4"
PATTERNS=()
while read -r a lr epu; do
    python -I scripts/train.py "$a" --seeds 0-9 --lr "$lr" --epu "$epu"
    PATTERNS+=("${a}_lr$(python -c "print(f'{$lr:g}')")_epu${epu}_e100_s*")
done <<< "$RUNS"
python -I scripts/compare.py "${PATTERNS[@]}"
python -I scripts/reference_policies.py
