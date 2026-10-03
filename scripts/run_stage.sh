#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/carc_env.sh"
ssmo_env
[[ $# = 9 ]] || { ssmo_error 'internal stage interface requires stage run config source profile method seed manifest checkpoint'; exit 2; }
stage=$1
run_dir=$(ssmo_path "$2")
config=$(ssmo_path "$3")
source_dir=$(ssmo_path "$4")
profile=$5
method=$6
seed=$7
manifest=$8
checkpoint=$9
[[ -f "$config" && -d "$source_dir" ]] || { ssmo_error 'frozen source/config missing'; exit 2; }
ssmo_load_python
export PYTHONPATH="$source_dir"
export SSMO_PROJECT_ROOT="$SSMO_ROOT"
unset SSMO_EXPECTED_GPU
ssmo_mkdir "$run_dir/artifacts"
# All jobs share one project venv. Locks protect active tasks. Queue checks and
# a per-run freeze comparison protect pending stages from backend changes.
ssmo_path "$SSMO_ROOT/local/venv.lock" >/dev/null
exec 9> "$SSMO_ROOT/local/venv.lock"
if [[ "$stage" = install ]]; then
    flock -n -x 9 || { ssmo_error 'venv is in use; retry install after those tasks finish'; exit 3; }
    queued_names=$(squeue -h -u aadaniel -o '%j')
    run_id=$(basename -- "$run_dir")
    other_jobs=$(printf '%s\n' "$queued_names" | awk -v own="SSMO-$run_id-" '/^SSMO-/ && index($0, own) != 1 {n++} END {print n+0}')
    (( other_jobs == 0 )) || { ssmo_error 'other SSMO jobs are queued/running; do not change their environment'; exit 3; }
    ssmo_check_tree "$SSMO_ROOT/.venv"
    if [[ ! -f "$SSMO_ROOT/.venv/pyvenv.cfg" ]]; then
        python3 -m venv --copies "$SSMO_ROOT/.venv"
    fi
    ssmo_check_venv
    "$SSMO_PYTHON" -m pip install --requirement "$source_dir/requirements/base.txt"
    if [[ "$profile" = cpu ]]; then
        "$SSMO_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu --requirement "$source_dir/requirements/cpu.txt"
    else
        "$SSMO_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu126 --requirement "$source_dir/requirements/carc-cu126.txt"
    fi
    "$SSMO_PYTHON" -m pip check
    "$SSMO_PYTHON" -m pip freeze --all > "$run_dir/artifacts/dependency-freeze.txt"
    ssmo_path "$SSMO_ROOT/.venv/ssmo-dependency-freeze.txt" >/dev/null
    cp -- "$run_dir/artifacts/dependency-freeze.txt" "$SSMO_ROOT/.venv/ssmo-dependency-freeze.txt"
    "$SSMO_PYTHON" -m pip list --format=json > "$run_dir/artifacts/installed-versions.json"
    exit 0
fi
flock -n -s 9 || { ssmo_error 'venv is being installed; retry after successful installation'; exit 3; }
ssmo_check_venv
expected_freeze="$run_dir/artifacts/dependency-freeze.txt"
if [[ ! -f "$expected_freeze" ]]; then
    # A manually linked first-install predecessor creates this marker later.
    [[ -f "$SSMO_ROOT/.venv/ssmo-dependency-freeze.txt" ]] || { ssmo_error 'environment freeze missing; run the install prerequisite'; exit 3; }
    cp -- "$SSMO_ROOT/.venv/ssmo-dependency-freeze.txt" "$expected_freeze"
fi
"$SSMO_PYTHON" -m pip freeze --all > "$run_dir/artifacts/environment-$stage-$method-seed$seed.txt"
cmp -s "$expected_freeze" "$run_dir/artifacts/environment-$stage-$method-seed$seed.txt" || { ssmo_error 'environment changed since this run was prepared; preserve outputs and reinstall/review a new run'; exit 3; }
sha256sum "$expected_freeze" > "$run_dir/artifacts/environment-$stage-$method-seed$seed.sha256"
case "$stage" in
    validate)
        ssmo_mkdir "$run_dir/artifacts/test-tmp"
        export PYTEST_ADDOPTS="--basetemp=$run_dir/artifacts/test-tmp"
        "$SSMO_PYTHON" -m singular_sensitivity.cli audit --config "$config" --run-dir "$run_dir/artifacts/validate"
        ;;
    data)
        "$SSMO_PYTHON" -m singular_sensitivity.cli generate --config "$config" --run-dir "$run_dir/artifacts/data"
        ;;
    gpu-smoke)
        export SSMO_EXPECTED_GPU="$profile"
        "$SSMO_PYTHON" -m singular_sensitivity.cli gpu-audit --config "$config" --expected-model "$profile" --run-dir "$run_dir/artifacts/gpu-smoke"
        ;;
    train)
        manifest=$(ssmo_path "$manifest")
        if [[ "$profile" != cpu ]]; then
            export SSMO_EXPECTED_GPU="$profile"
        fi
        args=(train --config "$config" --run-dir "$run_dir/artifacts/train-$method-seed$seed" --manifest "$manifest" --method "$method" --seed "$seed")
        if [[ "$profile" = cpu ]]; then args+=(--device cpu); fi
        if [[ "$checkpoint" != none ]]; then args+=(--resume "$(ssmo_path "$checkpoint")"); fi
        # exec delivers the advance signal directly to the application's handler.
        exec "$SSMO_PYTHON" -m singular_sensitivity.cli "${args[@]}"
        ;;
    evaluate)
        if [[ "$profile" != cpu ]]; then
            export SSMO_EXPECTED_GPU="$profile"
        fi
        args=(evaluate --config "$config" --run-dir "$run_dir/artifacts/evaluate-$method-seed$seed" --manifest "$(ssmo_path "$manifest")" --checkpoint "$(ssmo_path "$checkpoint")")
        if [[ "$profile" = cpu ]]; then args+=(--device cpu); fi
        "$SSMO_PYTHON" -m singular_sensitivity.cli "${args[@]}"
        ;;
    report)
        "$SSMO_PYTHON" -m singular_sensitivity.cli representation --config "$config" --run-dir "$run_dir/artifacts/representation"
        "$SSMO_PYTHON" -m singular_sensitivity.cli numerical --config "$config" --run-dir "$run_dir/artifacts/numerical"
        args=(inverse --config "$config" --run-dir "$run_dir/artifacts/inverse" --device cpu)
        if [[ "$checkpoint" != none && -f "$checkpoint" ]]; then args+=(--checkpoint "$(ssmo_path "$checkpoint")"); fi
        "$SSMO_PYTHON" -m singular_sensitivity.cli "${args[@]}"
        "$SSMO_PYTHON" -m singular_sensitivity.cli report --config "$config" --run-dir "$run_dir/artifacts"
        ;;
    *) ssmo_error "unknown stage: $stage"; exit 2 ;;
esac
