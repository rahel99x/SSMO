#!/bin/bash
# Allocation, account, absolute log paths and dependency flags come from submit.sh.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { printf 'Run this task with sbatch via scripts/submit.sh\n' >&2; exit 2; }
[[ $# = 9 || $# = 10 ]] || { printf 'Invalid internal task arguments\n' >&2; exit 2; }
source_dir=$4
[[ "$source_dir" = /home1/aadaniel/projects/SSMO/runs/*/source ]] || { printf 'Invalid source snapshot path\n' >&2; exit 2; }
source "$source_dir/scripts/carc_env.sh"
ssmo_env
[[ $(id -un) = aadaniel ]] || { ssmo_error 'allocation identity must be aadaniel'; exit 2; }
[[ ${SLURM_JOB_ACCOUNT:-} = anakano_81 ]] || { ssmo_error 'allocation account mismatch or unavailable'; exit 2; }
run_dir=$(ssmo_path "$2")
attempt_dir=$(ssmo_path "${SSMO_TOWER_RUN_DIR:?reporting attempt missing}")
[[ "$attempt_dir" = "$run_dir/tower/"* && -d "$attempt_dir" && ! -L "$attempt_dir" ]] || { ssmo_error 'invalid reporting attempt'; exit 2; }
cd -- "$attempt_dir"
# The module bootstrap can publish install/preflight failures before a venv
# exists. Once verified, use the preferred project interpreter for reporting.
ssmo_load_python
report_python=$(command -v python3)
if ssmo_check_venv "${SSMO_PYTHON_MODULE:-python/3.12.8}" >/dev/null 2>&1; then
    report_python=$SSMO_PYTHON
fi
export PYTHONPATH="$source_dir"
export SSMO_TOWER_RUN_DIR="$attempt_dir"
scheduler_out=${SSMO_TOWER_SCHEDULER_OUT:?scheduler stdout missing}
scheduler_err=${SSMO_TOWER_SCHEDULER_ERR:?scheduler stderr missing}
scheduler_out=${scheduler_out//%j/$SLURM_JOB_ID}
scheduler_err=${scheduler_err//%j/$SLURM_JOB_ID}
# Tower's native GPU history reads this job-owned CSV from the actual WorkDir.
# Slurm's physical selector is required: task-visible cuda:0 does not identify
# a physical GPU, and an unqualified nvidia-smi would sample other users' GPUs.
gpu_trace_pid=
gpu_trace_selector=${SLURM_JOB_GPUS:-}
gpu_trace_nodes=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}
export SSMO_TOWER_GPU_TRACE_STATUS=skipped SSMO_TOWER_GPU_TRACE_REASON=cpu_request
unset SSMO_TOWER_GPU_TRACE_SELECTOR
if [[ ${SSMO_TOWER_GPU_COUNT:-} = 1 ]]; then
    if [[ ${SSMO_TOWER_GPU_TRACE:-1} = 0 ]]; then
        SSMO_TOWER_GPU_TRACE_REASON=disabled
    elif [[ ${SSMO_TOWER_GPU_TRACE:-1} != 1 ]]; then
        SSMO_TOWER_GPU_TRACE_REASON=invalid_trace_preference
    elif [[ "$gpu_trace_nodes" != 1 || ( -n ${SLURM_JOB_NUM_NODES:-} && -n ${SLURM_NNODES:-} && "$SLURM_JOB_NUM_NODES" != "$SLURM_NNODES" ) ]]; then
        SSMO_TOWER_GPU_TRACE_REASON=unverified_single_node
    elif [[ ! "$gpu_trace_selector" =~ ^[0-9]+$ && ! "$gpu_trace_selector" =~ ^GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$ ]]; then
        SSMO_TOWER_GPU_TRACE_REASON=missing_physical_selector
    elif ! command -v nvidia-smi >/dev/null; then
        SSMO_TOWER_GPU_TRACE_REASON=nvidia_smi_unavailable
    else
        SSMO_TOWER_GPU_TRACE_STATUS=eligible
        SSMO_TOWER_GPU_TRACE_REASON=single_allocated_physical_gpu
        export SSMO_TOWER_GPU_TRACE_SELECTOR="$gpu_trace_selector"
    fi
fi
"$report_python" -B -S -m singular_sensitivity.tower_reporting start \
    --attempt-dir "$attempt_dir" --pipeline-dir "$run_dir" \
    --stage "$1" --method "$6" --seed "$7" --profile "$5" \
    --config "$3" --source-dir "$source_dir" \
    --request-nodes 1 \
    --request-cpus "${SSMO_TOWER_CPUS:?CPU request missing}" \
    --request-mem-gb "${SSMO_TOWER_MEM_GB:?memory request missing}" \
    --request-time "${SSMO_TOWER_WALLTIME:?walltime request missing}" \
    --request-gpus "${SSMO_TOWER_GPU_COUNT:?GPU request missing}" \
    --request-gpu-type "${SSMO_TOWER_GPU_TYPE:-}" \
    --scheduler-out "$scheduler_out" --scheduler-err "$scheduler_err"
ssmo_tower_finish() {
    local task_status=$?
    trap - EXIT HUP INT TERM
    if [[ -n "$gpu_trace_pid" ]]; then
        # Only the still-running owned sampler may be terminated. The jobs
        # table prevents using a completed child's old PID for an unrelated job.
        if [[ "$(jobs -pr)" = "$gpu_trace_pid" ]]; then
            kill -TERM -- "$gpu_trace_pid" 2>/dev/null || true
        fi
        wait "$gpu_trace_pid" 2>/dev/null || true
    fi
    # Reporting failure must not turn a failed or paused scientific task into
    # success, or discard its original exit code and retained logs.
    if ! "$report_python" -B -S -m singular_sensitivity.tower_reporting finish \
        --attempt-dir "$attempt_dir" --exit-code "$task_status"; then
        ssmo_error 'reporting finalization failed; preserve this attempt and reconcile scheduler evidence'
    fi
    exit "$task_status"
}
trap ssmo_tower_finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$SSMO_TOWER_GPU_TRACE_STATUS" = eligible ]]; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits "--id=$gpu_trace_selector" --loop=60 \
        > "$attempt_dir/logs/gpu-util-$SLURM_JOB_ID.csv" \
        2> "$attempt_dir/logs/gpu-trace.stderr.log" &
    gpu_trace_pid=$!
else
    printf 'SSMO: native GPU trace skipped: %s\n' "$SSMO_TOWER_GPU_TRACE_REASON"
fi
# Keep training's exec and signal handlers inside the task. Separate exact app
# logs avoid tee pipelines and leave scheduler diagnostics in their own files.
srun --kill-on-bad-exit=1 --ntasks=1 \
    "--output=$attempt_dir/logs/application.stdout.log" \
    "--error=$attempt_dir/logs/application.stderr.log" \
    bash "$source_dir/scripts/run_stage.sh" "$@"
