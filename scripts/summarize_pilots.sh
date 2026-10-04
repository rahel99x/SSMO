#!/bin/bash
# Read completed pilot artifacts and write a fresh project-contained review.
set -euo pipefail
ssmo_review_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ssmo_review_runs=(ssmo-pilot-001 ssmo-pilot-seed29 ssmo-pilot-seed43)
ssmo_review_custom_runs=0
ssmo_review_output="runs/ssmo-pilot-review-$(date -u +%Y%m%dT%H%M%SZ)-$$"

usage() {
    cat <<'USAGE'
Usage: bash scripts/summarize_pilots.sh [--run ID ...] [--output-dir PATH]
Defaults: ssmo-pilot-001, ssmo-pilot-seed29, ssmo-pilot-seed43.
Reads existing JSON summaries with the project venv; no Slurm job is submitted.
Writes summary.json, summary.txt and a checksum receipt to a fresh directory
inside SSMO. Explicit --run options replace the default run list.
USAGE
}

while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run|--output-dir)
            (( $# >= 2 )) || { printf 'SSMO: missing value for %s\n' "$1" >&2; exit 2; }
            if [[ "$1" = --run ]]; then
                if (( ! ssmo_review_custom_runs )); then ssmo_review_runs=(); ssmo_review_custom_runs=1; fi
                ssmo_review_runs+=("$2")
            else
                ssmo_review_output=$2
            fi
            shift 2 ;;
        *) printf 'SSMO: unknown summary option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

source "$ssmo_review_root/scripts/carc_env.sh"
SSMO_PROJECT_ROOT=${SSMO_PROJECT_ROOT:-$ssmo_review_root}
ssmo_init_root
[[ "$SSMO_PROJECT_ROOT" = "$ssmo_review_root" ]] || { ssmo_error 'summary command must use this SSMO checkout'; exit 2; }
ssmo_env
if [[ "$SSMO_PROJECT_ROOT" = /home1/aadaniel/projects/SSMO ]]; then
    ssmo_load_python
    ssmo_check_venv "${SSMO_PYTHON_MODULE:-python/3.12.8}"
else
    ssmo_check_venv
fi
ssmo_review_args=(--project-root "$SSMO_PROJECT_ROOT" --output-dir "$ssmo_review_output")
for ssmo_review_run in "${ssmo_review_runs[@]}"; do
    ssmo_review_args+=(--run "$ssmo_review_run")
done
exec "$SSMO_PYTHON" "$ssmo_review_root/scripts/summarize_pilots.py" "${ssmo_review_args[@]}"
