#!/bin/bash
# Attribute existing failed pilot queries on CPU without loading checkpoints.
set -euo pipefail
ssmo_diagnostic_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ssmo_diagnostic_review="runs/ssmo-pilot-review-20261004T000302Z-2641811"
ssmo_diagnostic_output="runs/ssmo-pilot-diagnostics-$(date -u +%Y%m%dT%H%M%SZ)-$$"

usage() {
    cat <<'USAGE'
Usage: bash scripts/diagnose_pilot_failures.sh [--review-dir PATH] [--output-dir PATH]
Reads the completed three-seed review and existing query logs with Python's
standard library. No GPU, checkpoint load, training or Slurm submission occurs.
Writes a signed component ledger, summary and checksums to a fresh directory
inside SSMO; source runs remain unchanged.
USAGE
}

while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --review-dir|--output-dir)
            (( $# >= 2 )) || { printf 'SSMO: missing value for %s\n' "$1" >&2; exit 2; }
            if [[ "$1" = --review-dir ]]; then ssmo_diagnostic_review=$2; else ssmo_diagnostic_output=$2; fi
            shift 2 ;;
        *) printf 'SSMO: unknown diagnostic option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

source "$ssmo_diagnostic_root/scripts/carc_env.sh"
SSMO_PROJECT_ROOT=${SSMO_PROJECT_ROOT:-$ssmo_diagnostic_root}
ssmo_init_root
[[ "$SSMO_PROJECT_ROOT" = "$ssmo_diagnostic_root" ]] || { ssmo_error 'diagnostic command must use this SSMO checkout'; exit 2; }
ssmo_env
if [[ "$SSMO_PROJECT_ROOT" = /home1/aadaniel/projects/SSMO ]]; then
    ssmo_load_python
    ssmo_check_venv "${SSMO_PYTHON_MODULE:-python/3.12.8}"
else
    ssmo_check_venv
fi
exec "$SSMO_PYTHON" -B -S "$ssmo_diagnostic_root/scripts/diagnose_pilot_failures.py" \
    --project-root "$SSMO_PROJECT_ROOT" --review-dir "$ssmo_diagnostic_review" --output-dir "$ssmo_diagnostic_output"
