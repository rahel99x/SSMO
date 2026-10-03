#!/bin/bash
# Called within the existing allocated report task; never on a login node.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/carc_env.sh"
ssmo_init_root
[[ $# = 1 ]] || { ssmo_error 'usage: compact_logs.sh CURRENT_RUN_DIRECTORY'; exit 2; }
[[ "$SSMO_PROJECT_ROOT" = /home1/aadaniel/projects/SSNO && "$(id -un)" = aadaniel && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_JOB_ACCOUNT:-}" = anakano_81 ]] || {
    ssmo_error 'log compaction requires an allocated CARC anakano_81 task in the approved SSNO root'
    exit 2
}
SSMO_COMPACTION_RUN=$(ssmo_path "$1")
[[ "$(dirname -- "$SSMO_COMPACTION_RUN")" = "$SSMO_PROJECT_ROOT/runs" && -d "$SSMO_COMPACTION_RUN/artifacts" ]] || { ssmo_error 'current run artifacts missing or outside runs'; exit 2; }
ssmo_env
ssmo_check_venv
"$SSMO_PYTHON" -m singular_sensitivity.log_compaction --artifacts "$SSMO_COMPACTION_RUN/artifacts"
