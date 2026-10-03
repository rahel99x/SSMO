#!/bin/bash
# Optional bounded CPU validation-only search in the cloud/local checkout.
# CARC execution must use an allocation; this helper intentionally refuses it.
set -euo pipefail
SSMO_SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$SSMO_SCRIPT_DIRECTORY/carc_env.sh"
SSMO_PROJECT_ROOT=${SSMO_PROJECT_ROOT:-$(dirname -- "$SSMO_SCRIPT_DIRECTORY")}
ssmo_env
if [[ $(id -un) = aadaniel || -n ${SLURM_JOB_ID:-} || "$SSMO_PROJECT_ROOT" = /home1/aadaniel/projects/SSNO ]]; then
    ssmo_error 'tune_local.sh is for cloud/local CPU use; do not tune on a CARC login node'
    exit 2
fi
ssmo_check_venv
SSMO_TUNING_RUN_ID=${1:-ssmo-config-tuning-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$SSMO_TUNING_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { ssmo_error 'invalid tuning run ID'; exit 2; }
cd -- "$SSMO_PROJECT_ROOT"
exec "$SSMO_PYTHON" -m singular_sensitivity.tuning \
    --config configs/carc_pilot.yaml \
    --run-dir "$SSMO_PROJECT_ROOT/runs/$SSMO_TUNING_RUN_ID" \
    --max-seconds 300
