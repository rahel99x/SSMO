#!/bin/bash
# Allocation, account, absolute log paths and dependency flags come from submit.sh.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { printf 'Run this task with sbatch via scripts/submit.sh\n' >&2; exit 2; }
[[ $# = 9 ]] || { printf 'Invalid internal task arguments\n' >&2; exit 2; }
source_dir=$4
[[ "$source_dir" = /home1/aadaniel/projects/SSNO/runs/*/source ]] || { printf 'Invalid source snapshot path\n' >&2; exit 2; }
source "$source_dir/scripts/carc_env.sh"
ssmo_env
[[ $(id -un) = aadaniel ]] || { ssmo_error 'allocation identity must be aadaniel'; exit 2; }
[[ ${SLURM_JOB_ACCOUNT:-} = anakano_81 ]] || { ssmo_error 'allocation account mismatch or unavailable'; exit 2; }
# The Python module is loaded by run_stage.sh inside this same srun task.
srun --kill-on-bad-exit=1 --ntasks=1 bash "$source_dir/scripts/run_stage.sh" "$@"
