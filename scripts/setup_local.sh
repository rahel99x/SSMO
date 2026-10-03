#!/bin/bash
# Local/cloud CPU preparation. CARC installations use the Slurm install stage.
set -euo pipefail
SSMO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export SSMO_ROOT
source "$SSMO_ROOT/scripts/carc_env.sh"
if [[ "$SSMO_ROOT" = /home1/aadaniel/projects/SSMO ]]; then
    ssmo_error 'Use the module-aware Slurm install stage on CARC'
    exit 2
fi
ssmo_env
ssmo_check_tree "$SSMO_ROOT/.venv"
exec 9> "$SSMO_ROOT/local/venv.lock"
flock -n -x 9 || { ssmo_error 'venv is in use; install after existing tasks finish'; exit 3; }
python3 -m venv --copies "$SSMO_ROOT/.venv"
ssmo_check_venv
"$SSMO_PYTHON" -m pip install --requirement "$SSMO_ROOT/requirements/base.txt"
"$SSMO_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu --requirement "$SSMO_ROOT/requirements/cpu.txt"
"$SSMO_PYTHON" -m pip check
ssmo_mkdir "$SSMO_ROOT/local/setup"
"$SSMO_PYTHON" -m pip freeze --all > "$SSMO_ROOT/local/setup/cpu-freeze.txt"
ssmo_path "$SSMO_ROOT/.venv/ssmo-dependency-freeze.txt" >/dev/null
cp -- "$SSMO_ROOT/local/setup/cpu-freeze.txt" "$SSMO_ROOT/.venv/ssmo-dependency-freeze.txt"
export SSMO_PROJECT_ROOT="$SSMO_ROOT"
"$SSMO_PYTHON" -c 'import torch; print("CPU environment:", torch.__version__); assert torch.ones(2).sum().item() == 2'
