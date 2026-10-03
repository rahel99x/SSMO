#!/bin/bash
# Source this file, then call ssmo_env. No caller shell options are changed.

# Logging succeeds; each caller returns/exits its own meaningful status. A
# nonzero logger would trigger executable strict mode before the chosen exit.
ssmo_error() { printf 'SSMO: %s\n' "$*" >&2; }

ssmo_init_root() {
    # SSMO_ROOT is a legacy project-specific alias; never read PROJECT_ROOT.
    if [[ -n ${SSMO_PROJECT_ROOT:-} && -n ${SSMO_ROOT:-} && "$SSMO_PROJECT_ROOT" != "$SSMO_ROOT" ]]; then
        ssmo_error 'SSMO_PROJECT_ROOT and legacy SSMO_ROOT disagree'
        return 1
    fi
    SSMO_PROJECT_ROOT=${SSMO_PROJECT_ROOT:-${SSMO_ROOT:-/home1/aadaniel/projects/SSNO}}
    # Check before creating any caches: CARC writes have one approved root.
    if [[ $(id -un) = aadaniel || -n ${SLURM_JOB_ID:-} ]]; then
        [[ "$SSMO_PROJECT_ROOT" = /home1/aadaniel/projects/SSNO ]] || {
            ssmo_error 'CARC storage requires /home1/aadaniel/projects/SSNO'
            return 1
        }
    fi
    [[ "$SSMO_PROJECT_ROOT" = /* && "$SSMO_PROJECT_ROOT" != / ]] || { ssmo_error 'project root must be absolute'; return 1; }
    [[ -d "$SSMO_PROJECT_ROOT" ]] || { ssmo_error "project root does not exist: $SSMO_PROJECT_ROOT"; return 1; }
    local canonical
    canonical=$(realpath -e -- "$SSMO_PROJECT_ROOT") || return 1
    [[ "$canonical" = "$SSMO_PROJECT_ROOT" ]] || { ssmo_error 'project root must be canonical and cannot be a symlink'; return 1; }
    export SSMO_PROJECT_ROOT
}

# Resolve relative paths inside the project and reject every escaping component.
# This uses coreutils, so it works before Python/venv/cache setup.
ssmo_path() {
    local candidate=${1:?path required} canonical component resolved
    [[ "$candidate" = /* ]] || candidate="$SSMO_PROJECT_ROOT/$candidate"
    canonical=$(realpath -m -- "$candidate") || return 1
    [[ "$canonical" = "$SSMO_PROJECT_ROOT" || "$canonical" = "$SSMO_PROJECT_ROOT/"* ]] || { ssmo_error "path escapes project: $candidate"; return 1; }
    component="$candidate"
    while [[ "$component" != / && "$component" != . ]]; do
        if [[ -L "$component" ]]; then
            resolved=$(realpath -m -- "$component") || return 1
            [[ "$resolved" = "$SSMO_PROJECT_ROOT" || "$resolved" = "$SSMO_PROJECT_ROOT/"* ]] || { ssmo_error "symlink escapes project: $component"; return 1; }
        fi
        component=$(dirname -- "$component") || return 1
    done
    printf '%s\n' "$canonical"
}

ssmo_check_tree() {
    local location link
    location=$(ssmo_path "$1") || return 1
    [[ -e "$location" ]] || return 0
    (
        # Keep option changes in a subshell, and propagate traversal errors.
        set -o pipefail
        find "$location" -type l -print0 | while IFS= read -r -d '' link; do
            ssmo_path "$link" >/dev/null || return 1
        done
    )
}

ssmo_mkdir() {
    local location
    location=$(ssmo_path "$1") || return 1
    ssmo_check_tree "$location" || return 1
    mkdir -p -- "$location" || return 1
}

ssmo_env() {
    ssmo_init_root || return 1
    local location threads=${SLURM_CPUS_PER_TASK:-1}
    [[ "$threads" =~ ^[1-9][0-9]*$ ]] || { ssmo_error 'invalid allocation thread count'; return 1; }
    # The Python runtime also uses .cache; reject preexisting escaping links
    # before starting the interpreter, even though it validates paths itself.
    ssmo_check_tree "$SSMO_PROJECT_ROOT/.cache" || return 1
    for location in local/tmp local/cache/pip local/cache/xdg local/cache/torch local/cache/triton local/cache/inductor local/cache/extensions local/cache/matplotlib local/cache/cuda local/cache/compiler local/cache/bytecode local/cache/numba; do
        ssmo_mkdir "$location" || return 1
    done
    export TMPDIR="$SSMO_PROJECT_ROOT/local/tmp" TMP="$SSMO_PROJECT_ROOT/local/tmp" TEMP="$SSMO_PROJECT_ROOT/local/tmp"
    export PIP_CACHE_DIR="$SSMO_PROJECT_ROOT/local/cache/pip" XDG_CACHE_HOME="$SSMO_PROJECT_ROOT/local/cache/xdg"
    export XDG_CONFIG_HOME="$SSMO_PROJECT_ROOT/local/config" XDG_DATA_HOME="$SSMO_PROJECT_ROOT/local/data" XDG_STATE_HOME="$SSMO_PROJECT_ROOT/local/state"
    for location in "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME"; do ssmo_mkdir "$location" || return 1; done
    export TORCH_HOME="$SSMO_PROJECT_ROOT/local/cache/torch" TRITON_CACHE_DIR="$SSMO_PROJECT_ROOT/local/cache/triton"
    export TORCHINDUCTOR_CACHE_DIR="$SSMO_PROJECT_ROOT/local/cache/inductor"
    export TORCH_EXTENSIONS_DIR="$SSMO_PROJECT_ROOT/local/cache/extensions" MPLCONFIGDIR="$SSMO_PROJECT_ROOT/local/cache/matplotlib"
    export CUDA_CACHE_PATH="$SSMO_PROJECT_ROOT/local/cache/cuda" CCACHE_DIR="$SSMO_PROJECT_ROOT/local/cache/compiler"
    export CCACHE_TEMPDIR="$SSMO_PROJECT_ROOT/local/tmp" CMAKE_BUILD_PARALLEL_LEVEL="$threads" MAX_JOBS="$threads"
    export PYTHONPYCACHEPREFIX="$SSMO_PROJECT_ROOT/local/cache/bytecode" NUMBA_CACHE_DIR="$SSMO_PROJECT_ROOT/local/cache/numba"
    export PYTHONNOUSERSITE=1 PYTHONUSERBASE="$SSMO_PROJECT_ROOT/local/python-user"
    export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
    export NUMEXPR_NUM_THREADS="$threads" VECLIB_MAXIMUM_THREADS="$threads" BLIS_NUM_THREADS="$threads"
    export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_REQUIRE_VIRTUALENV=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export SSMO_PYTHON="$SSMO_PROJECT_ROOT/.venv/bin/python"
}

ssmo_load_python() {
    local selected=${SSMO_PYTHON_MODULE:-python/3.12.8}
    [[ "$selected" =~ ^python/[A-Za-z0-9._+-]+$ ]] || { ssmo_error 'invalid standalone Python module name'; return 1; }
    type module >/dev/null 2>&1 || { ssmo_error 'CARC module function unavailable in this shell'; return 1; }
    module load "$selected" || return 1
    command -v python3 >/dev/null || { ssmo_error 'python3 unavailable after module load'; return 1; }
}

ssmo_check_venv() {
    local canonical version selected=${1:-}
    canonical=$(ssmo_path .venv/bin/python) || return 1
    [[ -x "$canonical" && -f "$SSMO_PROJECT_ROOT/.venv/pyvenv.cfg" ]] || { ssmo_error 'project venv missing; run install stage'; return 1; }
    ssmo_check_tree "$SSMO_PROJECT_ROOT/.venv" || return 1
    if [[ -n "$selected" ]]; then
        version=$(awk -F' = ' '$1 == "version" {print $2}' "$SSMO_PROJECT_ROOT/.venv/pyvenv.cfg")
        [[ "$version" = "${selected#python/}" ]] || {
            ssmo_error 'venv Python version differs from requested module; preserve it inside SSNO and create a fresh environment after all tasks finish'
            return 1
        }
    fi
}

# Read-only readiness checks work on the login node without starting Python.
ssmo_environment_signature() {
    (
        set -o pipefail
        local selected=${1:-${SSMO_PYTHON_MODULE:-python/3.12.8}}
        local source_root=${2:-$SSMO_PROJECT_ROOT} backend=${3:-cuda} location wheel_file=carc-cu126.txt
        [[ "$selected" =~ ^python/[A-Za-z0-9._+-]+$ && ( "$backend" = cuda || "$backend" = cpu ) ]] || exit 1
        source_root=$(ssmo_path "$source_root") || exit 1
        [[ "$backend" != cpu ]] || wheel_file=cpu.txt
        {
            printf '%s\n%s\n' "$selected" "$backend"
            for location in base.txt "$wheel_file"; do
                location=$(ssmo_path "$source_root/requirements/$location") || exit 1
                [[ -f "$location" ]] || exit 1
                sha256sum "$location" | awk '{print $1}' || exit 1
            done
        } | sha256sum | awk '{print $1}'
    )
}

ssmo_environment_ready() {
    local selected=${1:-${SSMO_PYTHON_MODULE:-python/3.12.8}} location expected actual version
    for location in .venv/pyvenv.cfg .venv/bin/python .venv/ssmo-dependency-freeze.txt .venv/ssmo-environment-signature.txt; do
        location=$(ssmo_path "$location") || return 1
        [[ -f "$location" ]] || return 1
    done
    [[ -x "$SSMO_PROJECT_ROOT/.venv/bin/python" ]] || return 1
    ssmo_check_tree "$SSMO_PROJECT_ROOT/.venv" || return 1
    version=$(awk -F' = ' '$1 == "version" {print $2}' "$SSMO_PROJECT_ROOT/.venv/pyvenv.cfg")
    [[ "$version" = "${selected#python/}" ]] || return 1
    expected=$(ssmo_environment_signature "$selected") || return 1
    actual=$(cat "$SSMO_PROJECT_ROOT/.venv/ssmo-environment-signature.txt") || return 1
    [[ "$actual" = "$expected" ]] || return 1
    grep -Fxq 'torch==2.10.0+cu126' "$SSMO_PROJECT_ROOT/.venv/ssmo-dependency-freeze.txt" || return 1
}
