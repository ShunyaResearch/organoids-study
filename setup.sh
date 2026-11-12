#!/usr/bin/env bash
# Sets up Python 3.10.12 and the pinned packages with uv. Run from this folder: ./setup.sh   (safe to re-run)
# The dataset is not downloaded: get aggregated_experiment_data.pkl from https://zenodo.org/records/17684862
# and put it in data/ (setup.sh checks it if it is there).
set -euo pipefail
cd "$(dirname "$0")"

PY=3.10.12
DATA=data/aggregated_experiment_data.pkl
MD5=af99f1f859136ca4a6dad47a1aa7c86d

# uv if installed, otherwise from nix, otherwise ask the user to install one of them
if command -v uv >/dev/null 2>&1; then UV=uv
elif command -v nix >/dev/null 2>&1; then UV="nix --extra-experimental-features nix-command --extra-experimental-features flakes run nixpkgs#uv --"
else echo "Please install uv (https://docs.astral.sh/uv/) or nix, then re-run ./setup.sh"; exit 1; fi

[ -x .venv/bin/python ] || $UV venv --python "$PY" .venv
$UV pip install --python .venv/bin/python -r requirements.txt --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu

mkdir -p data
if [ -f "$DATA" ]; then
    if [ "$(md5sum "$DATA" | cut -d' ' -f1)" = "$MD5" ]; then echo "dataset: checksum OK"
    else echo "dataset: checksum does NOT match the Zenodo file ($DATA)"; exit 1; fi
else
    echo "dataset: not found. Download aggregated_experiment_data.pkl from https://zenodo.org/records/17684862"
    echo "         and save it as $DATA"
fi
echo "Ready. Activate with: source .venv/bin/activate   (commands in README.md)"
