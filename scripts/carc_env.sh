#!/bin/bash
# Source this file, then call ssmo_env. No caller shell options are changed.

# Logging succeeds; each caller returns/exits its own meaningful status. A
# nonzero logger would trigger executable strict mode before the chosen exit.
ssmo_error() { printf 'SSMO: %s\n' "$*" >&2; }

ssmo_init_root() {
    SSMO_ROOT=${SSMO_ROOT:-/home1/aadaniel/projects/SSMO}
    [[ "$SSMO_ROOT" = /* && "$SSMO_ROOT" != / ]] || { ssmo_error 'project root must be absolute'; return 1; }
    [[ -d "$SSMO_ROOT" ]] || { ssmo_error "project root does not exist: $SSMO_ROOT"; return 1; }
    local canonical
    canonical=$(realpath -e -- "$SSMO_ROOT") || return 1
    [[ "$canonical" = "$SSMO_ROOT" ]] || { ssmo_error 'project root must be canonical and cannot be a symlink'; return 1; }
    export SSMO_ROOT
    export SSMO_PROJECT_ROOT="$SSMO_ROOT"
}

# Resolve relative paths inside the project and reject every escaping component.
# This uses coreutils, so it works before Python/venv/cache setup.
ssmo_path() {
    local candidate=${1:?path required} canonical component resolved
    [[ "$candidate" = /* ]] || candidate="$SSMO_ROOT/$candidate"
    canonical=$(realpath -m -- "$candidate") || return 1
    [[ "$canonical" = "$SSMO_ROOT" || "$canonical" = "$SSMO_ROOT/"* ]] || { ssmo_error "path escapes project: $candidate"; return 1; }
    component="$candidate"
    while [[ "$component" != / && "$component" != . ]]; do
        if [[ -L "$component" ]]; then
            resolved=$(realpath -m -- "$component") || return 1
            [[ "$resolved" = "$SSMO_ROOT" || "$resolved" = "$SSMO_ROOT/"* ]] || { ssmo_error "symlink escapes project: $component"; return 1; }
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
    ssmo_check_tree "$SSMO_ROOT/.cache" || return 1
    for location in local/tmp local/cache/pip local/cache/xdg local/cache/torch local/cache/triton local/cache/inductor local/cache/extensions local/cache/matplotlib local/cache/cuda local/cache/compiler local/cache/bytecode local/cache/numba; do
        ssmo_mkdir "$location" || return 1
    done
    export TMPDIR="$SSMO_ROOT/local/tmp" TMP="$SSMO_ROOT/local/tmp" TEMP="$SSMO_ROOT/local/tmp"
    export PIP_CACHE_DIR="$SSMO_ROOT/local/cache/pip" XDG_CACHE_HOME="$SSMO_ROOT/local/cache/xdg"
    export XDG_CONFIG_HOME="$SSMO_ROOT/local/config" XDG_DATA_HOME="$SSMO_ROOT/local/data" XDG_STATE_HOME="$SSMO_ROOT/local/state"
    for location in "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME"; do ssmo_mkdir "$location" || return 1; done
    export TORCH_HOME="$SSMO_ROOT/local/cache/torch" TRITON_CACHE_DIR="$SSMO_ROOT/local/cache/triton"
    export TORCHINDUCTOR_CACHE_DIR="$SSMO_ROOT/local/cache/inductor"
    export TORCH_EXTENSIONS_DIR="$SSMO_ROOT/local/cache/extensions" MPLCONFIGDIR="$SSMO_ROOT/local/cache/matplotlib"
    export CUDA_CACHE_PATH="$SSMO_ROOT/local/cache/cuda" CCACHE_DIR="$SSMO_ROOT/local/cache/compiler"
    export CCACHE_TEMPDIR="$SSMO_ROOT/local/tmp" CMAKE_BUILD_PARALLEL_LEVEL="$threads" MAX_JOBS="$threads"
    export PYTHONPYCACHEPREFIX="$SSMO_ROOT/local/cache/bytecode" NUMBA_CACHE_DIR="$SSMO_ROOT/local/cache/numba"
    export PYTHONNOUSERSITE=1 PYTHONUSERBASE="$SSMO_ROOT/local/python-user"
    export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
    export NUMEXPR_NUM_THREADS="$threads" VECLIB_MAXIMUM_THREADS="$threads" BLIS_NUM_THREADS="$threads"
    export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_REQUIRE_VIRTUALENV=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export SSMO_PYTHON="$SSMO_ROOT/.venv/bin/python"
}

ssmo_load_python() {
    local selected=${SSMO_PYTHON_MODULE:-python/3.11.9}
    [[ "$selected" =~ ^python/[A-Za-z0-9._+-]+$ ]] || { ssmo_error 'invalid standalone Python module name'; return 1; }
    type module >/dev/null 2>&1 || { ssmo_error 'CARC module function unavailable in this shell'; return 1; }
    module load "$selected" || return 1
    command -v python3 >/dev/null || { ssmo_error 'python3 unavailable after module load'; return 1; }
}

ssmo_check_venv() {
    local canonical
    canonical=$(ssmo_path .venv/bin/python) || return 1
    [[ -x "$canonical" && -f "$SSMO_ROOT/.venv/pyvenv.cfg" ]] || { ssmo_error 'project venv missing; run install stage'; return 1; }
    ssmo_check_tree "$SSMO_ROOT/.venv" || return 1
}
