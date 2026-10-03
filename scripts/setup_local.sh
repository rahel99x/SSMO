#!/bin/bash
# Local/cloud CPU preparation. CARC installations use the Slurm install stage.
set -euo pipefail
SSMO_PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export SSMO_PROJECT_ROOT
source "$SSMO_PROJECT_ROOT/scripts/carc_env.sh"
if [[ "$SSMO_PROJECT_ROOT" = /home1/aadaniel/projects/SSMO ]]; then
    ssmo_error 'Use the module-aware Slurm install stage on CARC'
    exit 2
fi
ssmo_env
ssmo_check_tree "$SSMO_PROJECT_ROOT/.venv"
exec 9<> "$SSMO_PROJECT_ROOT/local/venv.lock"
ssmo_lock_venv -x 'venv is in use; install after existing tasks finish'
python3 -m venv --copies "$SSMO_PROJECT_ROOT/.venv"
ssmo_check_venv
"$SSMO_PYTHON" -m pip install --requirement "$SSMO_PROJECT_ROOT/requirements/base.txt"
"$SSMO_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu --requirement "$SSMO_PROJECT_ROOT/requirements/cpu.txt"
"$SSMO_PYTHON" -m pip check
ssmo_mkdir "$SSMO_PROJECT_ROOT/local/setup"
"$SSMO_PYTHON" -m pip freeze --all > "$SSMO_PROJECT_ROOT/local/setup/cpu-freeze.txt"
ssmo_path "$SSMO_PROJECT_ROOT/.venv/ssmo-dependency-freeze.txt" >/dev/null
cp -- "$SSMO_PROJECT_ROOT/local/setup/cpu-freeze.txt" "$SSMO_PROJECT_ROOT/.venv/ssmo-dependency-freeze.txt"
"$SSMO_PYTHON" -c 'import torch; print("CPU environment:", torch.__version__); assert torch.ones(2).sum().item() == 2'
