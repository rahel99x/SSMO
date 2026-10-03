#!/bin/bash
# Small CARC frontend. Computational work remains inside allocated stages.
set -euo pipefail
SSMO_SCRIPT_DIR=$(realpath -e -- "$(dirname -- "${BASH_SOURCE[0]}")")
source "$SSMO_SCRIPT_DIR/carc_env.sh"
SSMO_PROJECT_ROOT=${SSMO_PROJECT_ROOT:-${SSMO_ROOT:-$(realpath -e -- "$SSMO_SCRIPT_DIR/..")}}
ssmo_init_root
SSMO_FRONTEND_ROOT=$SSMO_PROJECT_ROOT
if [[ -z ${SSMO_SITE_CONFIG:-} ]]; then
    SSMO_LOCAL_SITE_CONFIG=$(ssmo_path local/carc_site.env)
    if [[ -f "$SSMO_LOCAL_SITE_CONFIG" ]]; then SSMO_SITE_CONFIG=$SSMO_LOCAL_SITE_CONFIG; else SSMO_SITE_CONFIG=configs/carc_site.env; fi
fi
SSMO_SITE_CONFIG=$(ssmo_path "$SSMO_SITE_CONFIG")
[[ -f "$SSMO_SITE_CONFIG" ]] || { ssmo_error 'site settings missing: configs/carc_site.env'; exit 2; }
source "$SSMO_SITE_CONFIG"
[[ "$SSMO_PROJECT_ROOT" = "$SSMO_FRONTEND_ROOT" ]] || { ssmo_error 'site settings cannot change the project root'; exit 2; }
ssmo_init_root

usage() {
    cat <<'USAGE'
Usage: bash scripts/carc.sh COMMAND [OPTIONS]
  discover                  Capture policy, modules, GPU tags and account queue.
  setup                     Allocated venv install, tests and GPU audit.
  smoke | pilot             Serial bounded experiment and final report.
  status --run-id ID         Recorded jobs, scheduler/accounting and report path.
  resume --from-run ID       Resume the selected method from its last checkpoint.
  evaluate --from-run ID     Evaluate its selected best checkpoint.
  report --from-run ID       Collect old artifacts plus fresh CPU diagnostics.
Options:
  --run-id ID --gpu-profile a10040|a40|a30|l40|l40s --seed N
  --config PATH --python-module python/VERSION --after JOBID --max-project-jobs N
  --submit | --dry-run --skip-install | --reinstall
  --account-slots N --account-free-cpus N --account-free-mem-gb N
  --account-free-gpus N
Recovery: --from-run ID --method measure|state_only
  --checkpoint PATH applies to evaluate/report; --resume PATH applies to resume.
Inspection commands discover/status accept only --run-id ID (and --help).
Experiment default: print the complete plan; no writes or submission.
The explicit discover command writes only policy records inside the project.
Site settings: local/carc_site.env, falling back to configs/carc_site.env.
An explicit project-confined SSMO_SITE_CONFIG overrides both.
Live actions capture fresh discovery automatically, then use reviewed capacity.
No automatic job cancellation or run removal. Every result uses a fresh run ID.
USAGE
}
[[ $# -gt 0 ]] || { usage; exit 2; }
command=$1; shift
case "$command" in
    --help|-h|help) usage; exit 0 ;;
    discover|setup|smoke|pilot|status|resume|evaluate|report) ;;
    *) ssmo_error "unknown command: $command"; usage >&2; exit 2 ;;
esac
run_id= from_run= config= after= checkpoint= resume=
submit=0 skip_install=0 reinstall=0
profile=${SSMO_GPU_PROFILE:-a10040}
python_module=${SSMO_PYTHON_MODULE:-python/3.12.8}
max_jobs=${SSMO_MAX_PROJECT_JOBS:-10}
method=measure seed=17
profile_set=0 module_set=0 method_set=0 seed_set=0
slots=${SSMO_ACCOUNT_SLOTS:-}
free_cpus=${SSMO_ACCOUNT_FREE_CPUS:-}
free_mem=${SSMO_ACCOUNT_FREE_MEM_GB:-}
free_gpus=${SSMO_ACCOUNT_FREE_GPUS:-}
requested_options=()
while (( $# )); do
    requested_options+=("$1")
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --submit) submit=1; shift; continue ;;
        --dry-run) submit=0; shift; continue ;;
        --skip-install) skip_install=1; shift; continue ;;
        --reinstall) reinstall=1; shift; continue ;;
        --run-id|--from-run|--config|--gpu-profile|--python-module|--seed|--method|--after|--checkpoint|--resume|--max-project-jobs|--account-slots|--account-free-cpus|--account-free-mem-gb|--account-free-gpus)
            (( $# >= 2 )) || { ssmo_error "missing value for $1"; exit 2; }
            flag=$1 value=$2
            case "$flag" in
                --run-id) run_id=$value ;; --from-run) from_run=$value ;;
                --config) config=$value ;; --after) after=$value ;;
                --checkpoint) checkpoint=$value ;; --resume) resume=$value ;;
                --gpu-profile) profile=$value; profile_set=1 ;;
                --python-module) python_module=$value; module_set=1 ;;
                --seed) seed=$value; seed_set=1 ;;
                --method) method=$value; method_set=1 ;;
                --max-project-jobs) max_jobs=$value ;;
                --account-slots) slots=$value ;; --account-free-cpus) free_cpus=$value ;;
                --account-free-mem-gb) free_mem=$value ;; --account-free-gpus) free_gpus=$value ;;
            esac
            shift 2 ;;
        *) ssmo_error "unknown option: $1"; exit 2 ;;
    esac
done
# Reject options with no effect before discovery, caching or any submission.
# Pipeline method sets are fixed; --method selects a control during recovery.
for option in "${requested_options[@]}"; do
    if [[ "$command" = discover || "$command" = status ]]; then
        [[ "$option" = --run-id || "$option" = --skip-install || "$option" = --reinstall ]] || {
            ssmo_error "$command accepts only --run-id ID (and --help)"; exit 2;
        }
    fi
    case "$option" in
        --method|--from-run)
            case "$command" in resume|evaluate|report) ;; *) ssmo_error "$option applies only to resume, evaluate or report"; exit 2 ;; esac
            ;;
        --checkpoint)
            [[ "$command" = evaluate || "$command" = report ]] || { ssmo_error '--checkpoint applies only to evaluate or report'; exit 2; }
            ;;
        --resume)
            [[ "$command" = resume ]] || { ssmo_error '--resume applies only to resume'; exit 2; }
            ;;
    esac
done
valid_id() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]]; }
[[ -z "$run_id" ]] || valid_id "$run_id" || { ssmo_error 'invalid run ID'; exit 2; }
[[ -z "$from_run" ]] || valid_id "$from_run" || { ssmo_error 'invalid source run ID'; exit 2; }
[[ "$seed" =~ ^[0-9]+$ && "$max_jobs" =~ ^[1-9][0-9]*$ ]] || { ssmo_error 'invalid seed/job bound'; exit 2; }
[[ "$method" = measure || "$method" = state_only ]] || { ssmo_error 'invalid method'; exit 2; }
[[ "$python_module" =~ ^python/[A-Za-z0-9._+-]+$ ]] || { ssmo_error 'invalid Python module'; exit 2; }
[[ -z "$after" || "$after" =~ ^[0-9]+$ ]] || { ssmo_error '--after requires one real job ID'; exit 2; }
case "$profile" in a10040|a40|a30|l40|l40s) ;; *) ssmo_error 'unknown GPU profile'; exit 2 ;; esac
(( ! (skip_install && reinstall) )) || { ssmo_error '--skip-install and --reinstall conflict'; exit 2; }
if (( skip_install || reinstall )); then
    case "$command" in
        setup|smoke|pilot) ;;
        *) ssmo_error '--skip-install and --reinstall apply only to setup, smoke or pilot'; exit 2 ;;
    esac
fi
if [[ "$command" = status ]]; then
    [[ -n "$run_id" && -z "$from_run" ]] || { ssmo_error 'status requires --run-id ID'; exit 2; }
    run_dir=$(ssmo_path "runs/$run_id")
    jobs_file=$(ssmo_path "$run_dir/jobs.tsv")
    [[ -f "$jobs_file" ]] || { ssmo_error 'recorded jobs.tsv missing'; exit 2; }
    cat -- "$jobs_file"
    job_ids=$(awk -F '\t' 'NR > 1 && $4 ~ /^[0-9]+$/ {print $4}' "$jobs_file" | paste -sd, -)
    if [[ -n "$job_ids" ]]; then
        if command -v squeue >/dev/null; then squeue -j "$job_ids" -o '%.18i %.30j %.12T %.10M %.35R'; else printf 'squeue unavailable; inspect recorded logs on CARC.\n'; fi
        if command -v sacct >/dev/null; then sacct -j "$job_ids" --format=JobID,JobName,Account,State,Elapsed,AllocTRES,MaxRSS,ExitCode; else printf 'sacct unavailable; accounting needs CARC.\n'; fi
    fi
    printf 'Logs: %s/logs\nReport: %s/artifacts/report.json\n' "$run_dir" "$run_dir"
    exit 0
fi
[[ -n "$run_id" ]] || run_id="ssmo-$command-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
run_dir=$(ssmo_path "runs/$run_id")
[[ ! -e "$run_dir" ]] || { ssmo_error 'run ID already exists; choose a fresh ID'; exit 2; }
if [[ "$command" = discover ]]; then
    [[ -z "$from_run" ]] || { ssmo_error 'discover does not use --from-run'; exit 2; }
    bash "$SSMO_PROJECT_ROOT/scripts/discover_carc.sh" --run-id "$run_id"
    printf 'Review %s/discovery and set all four SSMO_ACCOUNT_* free-capacity values.\n' "$run_dir"
    exit 0
fi

# Read generated metadata as data. Never source it as shell code.
field() { awk -F '=' -v key="$1" '$1 == key {print substr($0, length(key)+2); exit}' "$2"; }
manifest= report_from=
case "$command" in
    resume|evaluate|report)
        [[ -n "$from_run" ]] || { ssmo_error "$command requires --from-run ID"; exit 2; }
        previous=$(ssmo_path "runs/$from_run")
        metadata=$(ssmo_path "$previous/submission-manifest.txt")
        frozen_config=$(ssmo_path "$previous/config.yaml")
        source_snapshot=$(ssmo_path "$previous/source")
        hashes=$(ssmo_path "$previous/source-sha256.txt")
        jobs=$(ssmo_path "$previous/jobs.tsv")
        [[ -f "$metadata" && -f "$frozen_config" && -f "$hashes" && -f "$jobs" ]] || { ssmo_error 'original frozen run metadata is incomplete'; exit 2; }
        ssmo_check_tree "$source_snapshot"
        (cd "$source_snapshot"; sha256sum --check --status "$hashes") || { ssmo_error 'original source snapshot checksum failed'; exit 2; }
        run_hashes=$(ssmo_path "$previous/run-sha256.txt")
        [[ -f "$run_hashes" ]] || { ssmo_error 'original frozen configuration checksum missing'; exit 2; }
        expected_config_hash=$(awk -v target="$frozen_config" '$2 == target {print $1}' "$run_hashes")
        [[ "$expected_config_hash" =~ ^[a-f0-9]{64}$ && "$(sha256sum "$frozen_config" | cut -d ' ' -f 1)" = "$expected_config_hash" ]] || { ssmo_error 'original frozen configuration checksum failed'; exit 2; }
        # Resume/evaluate must use the same scientific implementation and pinned
        # dependencies. Frontend/runbook/site-headroom edits need not invalidate it.
        for source_path in singular_sensitivity requirements pyproject.toml; do
            current=$(ssmo_path "$source_path")
            old=$(ssmo_path "$source_snapshot/$source_path")
            ssmo_check_tree "$current"
            if [[ -d "$current" && -d "$old" ]]; then
                diff -qr --exclude=__pycache__ --exclude='*.pyc' -- "$old" "$current" >/dev/null || { ssmo_error "science/dependencies changed: $source_path; restore the original checkout before recovery"; exit 2; }
            elif [[ -f "$current" && -f "$old" ]]; then
                cmp -s -- "$old" "$current" || { ssmo_error "science/dependencies changed: $source_path"; exit 2; }
            else
                ssmo_error "original source missing: $source_path"; exit 2
            fi
        done
        [[ -z "$config" || "$(ssmo_path "$config")" = "$frozen_config" ]] || { ssmo_error 'recovery uses the original frozen configuration'; exit 2; }
        config=$frozen_config
        old_profile=$(field profile "$metadata")
        old_module=$(field python_module "$metadata")
        (( ! profile_set )) || [[ "$profile" = "$old_profile" ]] || { ssmo_error 'recovery must preserve the recorded GPU profile'; exit 2; }
        (( ! module_set )) || [[ "$python_module" = "$old_module" ]] || { ssmo_error 'recovery must preserve the recorded Python module'; exit 2; }
        profile=$old_profile; python_module=$old_module
        # A pilot has two methods; --method selects the recorded training row.
        if (( ! method_set )); then
            old_method=$(field method "$metadata")
            [[ -z "$old_method" ]] || method=$old_method
        fi
        training_row=$(awk -F '\t' -v method="$method" -v seed="$seed" -v selected="$seed_set" 'NR > 1 && $1 == "train" && $2 == method && (!selected || $3 == seed) {print $3 "|" $4; exit}' "$jobs")
        if [[ -n "$training_row" ]]; then
            seed=${training_row%%|*}; training_job=${training_row#*|}
            [[ "$seed" =~ ^[0-9]+$ && "$training_job" =~ ^[0-9]+$ ]] || { ssmo_error 'invalid original training metadata'; exit 2; }
        elif [[ "$command" = report ]]; then
            # Standalone evaluation/report runs retain the training settings and
            # input checkpoint in metadata, although their jobs contain no train.
            old_method=$(field method "$metadata")
            old_seed=$(field seed "$metadata")
            [[ "$old_seed" =~ ^[0-9]+$ && ( "$old_method" = measure || "$old_method" = state_only ) ]] || { ssmo_error 'invalid recorded method/seed'; exit 2; }
            (( ! method_set )) || [[ "$method" = "$old_method" ]] || { ssmo_error 'report must preserve the recorded method'; exit 2; }
            (( ! seed_set )) || [[ "$seed" = "$old_seed" ]] || { ssmo_error 'report must preserve the recorded seed'; exit 2; }
            method=$old_method; seed=$old_seed; training_job=
        else
            ssmo_error 'selected method/seed has no recorded training job'; exit 2
        fi
        manifest=$(field manifest "$metadata")
        [[ -n "$manifest" ]] || manifest="$previous/artifacts/data/parents.json"
        manifest=$(ssmo_path "$manifest")
        if [[ "$command" = resume ]]; then
            [[ -n "$resume" ]] || resume="$previous/artifacts/train-$method-seed$seed/last.pt"
            resume=$(ssmo_path "$resume")
            [[ -f "$resume" && -f "$manifest" ]] || { ssmo_error 'resume needs an existing last checkpoint and parent manifest'; exit 2; }
            [[ -f "$(ssmo_path "$(dirname -- "$resume")/best.pt")" ]] || { ssmo_error 'keep best.pt beside the resume checkpoint'; exit 2; }
        else
            # A full pilot records its default measure checkpoint in metadata.
            # Derive the selected control's checkpoint from its training row;
            # standalone evaluation/report sources instead retain input paths.
            if [[ "$command" = report && -z "$checkpoint" && -z "$training_row" ]]; then checkpoint=$(field checkpoint "$metadata"); fi
            [[ -n "$checkpoint" ]] || checkpoint="$previous/artifacts/train-$method-seed$seed/best.pt"
            checkpoint=$(ssmo_path "$checkpoint")
            if [[ "$command" = evaluate && ( ! -f "$checkpoint" || ! -f "$manifest" ) ]]; then
                [[ -n "$after" ]] || after=$training_job
                printf 'Waiting for original training job %s via afterok.\n' "$after"
            fi
            if [[ "$command" = report ]]; then
                report_from=$(ssmo_path "$previous/artifacts")
                [[ -d "$report_from" ]] || { ssmo_error 'original artifact directory is missing'; exit 2; }
                if [[ -z "$after" ]]; then
                    after=$(awk -F '\t' 'NR > 1 && $4 ~ /^[0-9]+$/ {last=$4} END {print last}' "$jobs")
                    [[ "$after" =~ ^[0-9]+$ ]] || { ssmo_error 'report needs a recorded predecessor job'; exit 2; }
                fi
                printf 'Report waits for job %s via afterok; a failed predecessor leaves it pending. No jobs are cancelled.\n' "$after"
            fi
        fi
        ;;
    *)
        [[ -z "$from_run" && -z "$checkpoint" && -z "$resume" ]] || { ssmo_error 'recovery options require resume, evaluate or report'; exit 2; }
        [[ -n "$config" ]] || { if [[ "$command" = pilot ]]; then config=configs/carc_pilot.yaml; else config=configs/carc_smoke.yaml; fi; }
        ;;
esac

config=$(ssmo_path "$config")
[[ -f "$config" && "$config" != *_PLAN.yaml ]] || { ssmo_error 'implemented configuration file required'; exit 2; }
case "$profile" in a10040|a40|a30|l40|l40s) ;; *) ssmo_error 'invalid recorded GPU profile'; exit 2 ;; esac
[[ "$python_module" =~ ^python/[A-Za-z0-9._+-]+$ ]] || { ssmo_error 'invalid recorded Python module'; exit 2; }

args=(--config "$config" --run-id "$run_id" --gpu-profile "$profile" --python-module "$python_module" --seed "$seed" --method "$method" --max-project-jobs "$max_jobs")
case "$command" in
    setup|smoke|pilot)
        args+=(--pipeline "$command")
        if (( ! reinstall )) && ssmo_environment_ready "$python_module"; then skip_install=1; fi
        if (( skip_install )); then args+=(--skip-install); printf 'Reuse the verified project venv; allocated stages recheck its freeze.\n'; fi
        ;;
    resume) args+=(--stage train --manifest "$manifest" --resume "$resume") ;;
    evaluate) args+=(--stage evaluate --manifest "$manifest" --checkpoint "$checkpoint") ;;
    report) args+=(--stage report --checkpoint "$checkpoint" --report-from "$report_from") ;;
esac
[[ -z "$after" ]] || args+=(--after "$after")
if (( submit )); then
    [[ "$SSMO_PROJECT_ROOT" = /home1/aadaniel/projects/SSNO && "$(id -un)" = aadaniel ]] || { ssmo_error 'live actions require CARC aadaniel and the approved SSNO root'; exit 2; }
    for value in "$slots" "$free_cpus" "$free_mem" "$free_gpus"; do
        [[ "$value" =~ ^[0-9]+$ ]] || { ssmo_error 'set all four reviewed SSMO_ACCOUNT_* free-capacity values (or explicit capacity flags); run discover to review them'; exit 2; }
    done
    discovery="ssmo-policy-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
    bash "$SSMO_PROJECT_ROOT/scripts/discover_carc.sh" --run-id "$discovery"
    args+=(--discovery "$discovery" --account-slots "$slots" --account-free-cpus "$free_cpus" --account-free-mem-gb "$free_mem" --account-free-gpus "$free_gpus" --submit)
else
    printf 'Preview only. Add --submit after reviewing site policy and free capacity.\n'
fi
bash "$SSMO_PROJECT_ROOT/scripts/submit.sh" "${args[@]}"
printf 'Run: %s\nNext: bash scripts/carc.sh status --run-id %s\n' "$run_dir" "$run_id"
if [[ "$command" = report ]]; then
    printf 'Report collects %s plus new diagnostics; earlier artifacts are preserved.\n' "$report_from"
fi
