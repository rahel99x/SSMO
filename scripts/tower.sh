#!/bin/bash
# Explicit Tower imports/planning and launch; no jobs, installation or Tower edits.
set -euo pipefail
ssmo_tower_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ ${1:-} = --help || ${1:-} = -h || $# = 0 ]]; then
    cat <<'USAGE'
Usage: bash scripts/tower.sh COMMAND [OPTIONS]
  import   --run ID ... [--artifact-dir PATH ...] [--output-dir FRESH_PATH]
           [--accounting | --accounting-file PATH]
  planning --export-dir PATH ... [--attempt-dir PATH ...] [--reference PATH]
           [--output reports/NAME.json] [--replace]
  list     --export-dir PATH
  validate --attempt-dir PATH [--planning-file PATH]
  launch   --attempt-dir PATH [--planning-file PATH]
Imports read existing artifacts into a fresh export; original runs stay intact.
No scientific jobs run. --accounting makes one bounded sacct query, never polls.
Planning contains only explicit attempts, preserves failures/unknown resources,
and refuses overwrite unless --replace is explicit. Tower must already exist.
USAGE
    exit 0
fi
source "$ssmo_tower_root/scripts/carc_env.sh"
SSMO_PROJECT_ROOT=${SSMO_PROJECT_ROOT:-$ssmo_tower_root}
ssmo_env
[[ "$SSMO_PROJECT_ROOT" = "$ssmo_tower_root" ]] || { ssmo_error 'Tower commands must use this SSMO checkout'; exit 2; }
if [[ "$SSMO_PROJECT_ROOT" = /home1/aadaniel/projects/SSMO || $(id -un) = aadaniel || -n ${SLURM_JOB_ID:-} ]]; then
    ssmo_load_python
fi
[[ -x "$SSMO_PROJECT_ROOT/.venv/bin/python" ]] || { ssmo_error 'project venv missing; use the existing environment setup runbook'; exit 2; }
export PYTHONPATH="$ssmo_tower_root"
exec "$SSMO_PROJECT_ROOT/.venv/bin/python" -B -S -m singular_sensitivity.tower_export "$@"
