#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/carc_env.sh"
ssmo_init_root
config=configs/carc_smoke.yaml
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
pipeline=smoke
single_stage=
profile=a10040
submit=0
after=
resume=none
manifest=
checkpoint=
report_source=none
skip_install=0
method=measure
seed=17
discovery=
free_cpus=
free_mem=
free_gpus=
python_module=python/3.12.8
usage() {
    cat <<'USAGE'
Usage: scripts/submit.sh [--config PATH] [--run-id ID] [--submit]
  [--pipeline setup|smoke|pilot | --stage install|validate|data|gpu-smoke|train|evaluate|report]
  [--gpu-profile a10040|a40|a30|l40|l40s|cpu] [--method measure|state_only] [--seed N]
  [--after JOBID] [--resume CHECKPOINT] [--manifest PATH] [--checkpoint PATH]
  [--discovery ID] [--python-module python/VERSION]
  [--skip-install] [--report-from ARTIFACT_DIRECTORY]
Live submissions require reviewed current account free capacity:
  --account-free-cpus N --account-free-mem-gb N --account-free-gpus N
Dry-run is the default and does not create directories or submit jobs.
USAGE
}
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --submit) submit=1; shift; continue ;;
        --dry-run) submit=0; shift; continue ;;
        --skip-install) skip_install=1; shift; continue ;;
        --config|--run-id|--pipeline|--stage|--gpu-profile|--method|--seed|--after|--resume|--manifest|--checkpoint|--report-from|--discovery|--python-module|--account-free-cpus|--account-free-mem-gb|--account-free-gpus)
            (( $# >= 2 )) || { ssmo_error "missing value for $1"; exit 2; }
            flag=$1 value=$2
            case "$flag" in
                --config) config=$value ;; --run-id) run_id=$value ;; --pipeline) pipeline=$value ;;
                --stage) single_stage=$value ;; --gpu-profile) profile=$value ;; --method) method=$value ;;
                --seed) seed=$value ;; --after) after=$value ;; --resume) resume=$value ;;
                --manifest) manifest=$value ;; --checkpoint) checkpoint=$value ;;
                --report-from) report_source=$value ;;
                --discovery) discovery=$value ;; --python-module) python_module=$value ;;
                --account-free-cpus) free_cpus=$value ;; --account-free-mem-gb) free_mem=$value ;;
                --account-free-gpus) free_gpus=$value ;;
            esac
            shift 2 ;;
        *) ssmo_error "unknown option: $1"; exit 2 ;;
    esac
done
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || { ssmo_error 'invalid run ID'; exit 2; }
[[ "$pipeline" = setup || "$pipeline" = smoke || "$pipeline" = pilot ]] || { ssmo_error 'pipeline must be setup, smoke or pilot'; exit 2; }
[[ "$method" = measure || "$method" = state_only ]] || { ssmo_error 'invalid method'; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { ssmo_error 'invalid seed'; exit 2; }
[[ -z "$after" || "$after" =~ ^[0-9]+$ ]] || { ssmo_error 'after must be one real job ID'; exit 2; }
[[ "$python_module" =~ ^python/[A-Za-z0-9._+-]+$ ]] || { ssmo_error 'invalid Python module'; exit 2; }
case "$profile" in a10040|a40|a30|l40|l40s|cpu) ;; *) ssmo_error 'unknown GPU profile'; exit 2 ;; esac
config=$(ssmo_path "$config")
[[ -f "$config" && "$config" != *_PLAN.yaml ]] || { ssmo_error 'implemented configuration file required'; exit 2; }
run_dir=$(ssmo_path "runs/$run_id")
[[ ! -e "$run_dir" ]] || { ssmo_error 'run ID already exists; use a fresh ID even for resume'; exit 2; }
source_dir="$run_dir/source"
frozen_config="$run_dir/config.yaml"
[[ -n "$manifest" ]] || manifest="$run_dir/artifacts/data/parents.json"
manifest=$(ssmo_path "$manifest")
[[ -n "$checkpoint" ]] || checkpoint="$run_dir/artifacts/train-$method-seed$seed/best.pt"
checkpoint=$(ssmo_path "$checkpoint")
if [[ "$resume" != none ]]; then
    resume=$(ssmo_path "$resume")
    [[ -f "$resume" && "$single_stage" = train ]] || { ssmo_error 'resume requires --stage train and an existing checkpoint'; exit 2; }
fi
if [[ "$report_source" != none ]]; then
    report_source=$(ssmo_path "$report_source")
    [[ -d "$report_source" && "$single_stage" = report ]] || { ssmo_error '--report-from requires an existing artifact directory and --stage report'; exit 2; }
    ssmo_check_tree "$report_source"
fi
if (( skip_install )); then
    [[ -z "$single_stage" ]] || { ssmo_error '--skip-install applies only to pipelines'; exit 2; }
    ssmo_environment_ready "$python_module" || { ssmo_error 'verified CUDA environment unavailable or changed; run setup before skipping installation'; exit 2; }
fi
stages=()
methods=()
if [[ -n "$single_stage" ]]; then
    case "$single_stage" in install|validate|data|gpu-smoke|train|evaluate|report) ;; *) ssmo_error 'unknown stage'; exit 2 ;; esac
    stages+=("$single_stage"); methods+=("$method")
    if [[ "$single_stage" = train || "$single_stage" = evaluate ]]; then
        [[ -f "$manifest" || -n "$after" ]] || { ssmo_error 'standalone train/evaluate requires existing --manifest or an explicit predecessor'; exit 2; }
    fi
    if [[ "$single_stage" = evaluate ]]; then [[ -f "$checkpoint" || -n "$after" ]] || { ssmo_error 'evaluation checkpoint missing without an explicit predecessor'; exit 2; }; fi
else
    [[ "$profile" != cpu ]] || { ssmo_error 'full pipeline uses an allocated GPU; use CPU --stage for CPU-only stages'; exit 2; }
    if (( ! skip_install )); then stages+=(install); methods+=(measure); fi
    stages+=(validate); methods+=(measure)
    if [[ "$pipeline" = setup ]]; then
        stages+=(gpu-smoke); methods+=(measure)
    else
        stages+=(data gpu-smoke train evaluate)
        methods+=(measure measure measure measure)
        if [[ "$pipeline" = pilot ]]; then stages+=(train evaluate); methods+=(state_only state_only); fi
        stages+=(report); methods+=(measure)
    fi
fi
# Conservative budgets; application measurements must precede larger campaigns.
resources() {
    case "$1" in
        install) cpus=4; memory=8; wall=00:30:00 ;;
        validate) cpus=2; memory=4; wall=00:10:00 ;;
        data) cpus=1; memory=2; wall=00:10:00 ;;
        gpu-smoke) cpus=2; memory=8; wall=00:15:00 ;;
        train) cpus=2; memory=8; wall=00:30:00 ;;
        evaluate) cpus=2; memory=8; wall=00:10:00 ;;
        report) cpus=2; memory=4; wall=00:15:00 ;;
    esac
    partition=main; gpu_args=()
    if [[ "$1" = gpu-smoke || "$1" = train || "$1" = evaluate ]]; then
        [[ "$profile" != cpu ]] || { [[ "$1" != gpu-smoke ]] || return 2; return 0; }
        partition=gpu
        gres=$profile
        [[ "$profile" != a10040 ]] || gres=a100
        gpu_args=("--gpus-per-task=$gres:1")
        [[ "$profile" != a10040 ]] || gpu_args+=(--constraint=a100-40gb)
    fi
}

if (( submit )); then
    [[ "$SSMO_PROJECT_ROOT" = /home1/aadaniel/projects/SSMO ]] || { ssmo_error 'live submissions require the exact CARC project root'; exit 2; }
    [[ $(id -un) = aadaniel ]] || { ssmo_error 'live submissions require CARC user aadaniel'; exit 2; }
    command -v sbatch >/dev/null || { ssmo_error 'sbatch unavailable'; exit 2; }
    [[ "$discovery" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || { ssmo_error '--discovery ID is required for live submissions'; exit 2; }
    discovery_dir=$(ssmo_path "runs/$discovery/discovery")
    [[ -f "$discovery_dir/status.tsv" && -f "$discovery_dir/manifest.txt" ]] || { ssmo_error 'discovery record missing'; exit 2; }
    awk 'NR > 1 && $2 != 0 {bad=1} END {exit bad}' "$discovery_dir/status.tsv" || { ssmo_error 'discovery contains failed commands; resolve them before submission'; exit 2; }
    age=$(( $(date +%s) - $(stat -c %Y "$discovery_dir/manifest.txt") ))
    (( age >= 0 && age <= 3600 )) || { ssmo_error 'discovery must be rerun within one hour'; exit 2; }
    [[ $(cat "$discovery_dir/identity.txt") = aadaniel ]] || { ssmo_error 'discovery identity mismatch'; exit 2; }
    [[ -s "$discovery_dir/associations.txt" ]] || { ssmo_error 'account association missing'; exit 2; }
    grep -Eq 'anakano_81\|aadaniel\|' "$discovery_dir/associations.txt" || { ssmo_error 'account/user association not found'; exit 2; }
    grep -Fq -- "$python_module" "$discovery_dir/modules.txt" || { ssmo_error 'selected Python module was not observed'; exit 2; }
    for capacity in "$free_cpus" "$free_mem" "$free_gpus"; do
        [[ "$capacity" =~ ^[0-9]+$ ]] || { ssmo_error 'all three reviewed free-resource flags are required'; exit 2; }
    done
    peak_cpu=0; peak_mem=0; need_gpu=0
    for stage in "${stages[@]}"; do
        resources "$stage"
        (( cpus <= peak_cpu )) || peak_cpu=$cpus
        (( memory <= peak_mem )) || peak_mem=$memory
        [[ "$partition" != gpu ]] || need_gpu=1
        grep -Eq "^$partition(\\*?)\\|up\\|" "$discovery_dir/partitions.txt" || { ssmo_error "required partition $partition was not observed up"; exit 2; }
        awk -F'|' -v partition="$partition" '$1 == "anakano_81" && $2 == "aadaniel" && ($3 == "" || $3 == partition) {found=1} END {exit !found}' "$discovery_dir/associations.txt" || { ssmo_error "no observed account association authorizes partition $partition"; exit 2; }
        grep -Fq 'State=UP' "$discovery_dir/$partition.txt" || { ssmo_error "partition $partition is not confirmed State=UP"; exit 2; }
    done
    (( free_cpus >= peak_cpu && free_mem >= peak_mem && free_gpus >= need_gpu )) || { ssmo_error 'reviewed free account TRES capacity is insufficient'; exit 2; }
    if (( need_gpu )); then
        gres=$profile; [[ "$profile" != a10040 ]] || gres=a100
        grep -Eq "(^|[|,])gpu:$gres:[0-9]" "$discovery_dir/gpu_nodes.txt" || { ssmo_error "GPU GRES tag $gres not observed; revise profile after discovery"; exit 2; }
        if [[ "$profile" = a10040 ]]; then
            awk -F'|' '$2 ~ /gpu:a100:[0-9]/ && $3 ~ /(^|,)a100-40gb(,|$)/ {found=1} END {exit !found}' "$discovery_dir/gpu_nodes.txt" || { ssmo_error 'A100 GRES and 40GB feature were not observed together'; exit 2; }
        fi
    fi
    current_jobs=$(squeue -h -u aadaniel -o '%i|%j|%T|%a|%C|%m|%b')
    current_account=$(squeue -h -A anakano_81 -o '%i|%u|%j|%T|%C|%m|%b')
    # Pending jobs consume no current allocation; other projects may enqueue
    # while this review is fresh. Changes to allocated/non-PENDING records still
    # invalidate reviewed free TRES. Stable sorting ignores scheduler row order.
    current_allocations=$(printf '%s\n' "$current_account" | awk -F'|' 'NF && $4 != "PENDING"' | LC_ALL=C sort)
    observed_allocations=$(awk -F'|' 'NF && $4 != "PENDING"' "$discovery_dir/account_jobs.txt" | LC_ALL=C sort)
    [[ "$current_allocations" = "$observed_allocations" ]] || { ssmo_error 'shared account allocations changed; rediscover and review free resources'; exit 2; }
    existing=$(printf '%s\n' "$current_jobs" | awk -F'|' '$2 ~ /^SSMO-/ {n++} END {print n+0}')
    for stage in "${stages[@]}"; do
        if [[ "$stage" = install ]] && (( existing > 0 )); then
            ssmo_error 'installation must wait until other SSMO queued/running jobs finish'
            exit 2
        fi
    done
    ssmo_env
    ssmo_mkdir "$run_dir/logs"
    ssmo_mkdir "$run_dir/artifacts"
    ssmo_mkdir "$source_dir"
    # Include the known implementation source (tracked and untracked), never
    # arbitrary cache/data/output directories from the development checkout.
    source_paths=()
    for path in singular_sensitivity scripts slurm requirements tests configs docs pyproject.toml README.md AGENTS.md LICENSE .gitignore requirements.txt; do
        if [[ -e "$SSMO_PROJECT_ROOT/$path" ]]; then
            ssmo_path "$path" >/dev/null
            source_paths+=("$path")
        fi
    done
    tar -C "$SSMO_PROJECT_ROOT" --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' --exclude='.cache' --exclude='*.egg-info' --exclude='build' --exclude='dist' -cf "$run_dir/source.tar" "${source_paths[@]}"
    tar -C "$source_dir" -xf "$run_dir/source.tar"
    ssmo_check_tree "$source_dir"
    while IFS= read -r -d '' link; do
        target=$(realpath -m -- "$link")
        [[ "$target" = "$source_dir/"* ]] || { ssmo_error 'source snapshot symlink points to mutable or external files'; exit 2; }
    done < <(find "$source_dir" -type l -print0)
    cp -- "$config" "$frozen_config"
    (cd "$source_dir"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$run_dir/source-sha256.txt"
    (cd "$source_dir"; find . -type f -printf '%m %p\n' | sort) > "$run_dir/source-modes.txt"
    sha256sum "$run_dir/source.tar" "$frozen_config" > "$run_dir/run-sha256.txt"
    if [[ -e "$SSMO_PROJECT_ROOT/.git" ]]; then
        git -C "$SSMO_PROJECT_ROOT" rev-parse HEAD > "$run_dir/source-revision.txt" 2>/dev/null || printf 'unborn or unavailable\n' > "$run_dir/source-revision.txt"
    else
        printf 'Git metadata unavailable\n' > "$run_dir/source-revision.txt"
    fi
    if [[ ! -e "$SSMO_PROJECT_ROOT/.git" ]] || ! git -C "$SSMO_PROJECT_ROOT" status --short > "$run_dir/source-status.txt" 2> "$run_dir/git-status-error.txt"; then
        printf 'Git metadata unavailable; use recorded source hashes and modes.\n' > "$run_dir/source-status.txt"
    fi
    if [[ -f "$SSMO_PROJECT_ROOT/.venv/ssmo-dependency-freeze.txt" ]]; then
        ssmo_path "$SSMO_PROJECT_ROOT/.venv/ssmo-dependency-freeze.txt" >/dev/null
        cp -- "$SSMO_PROJECT_ROOT/.venv/ssmo-dependency-freeze.txt" "$run_dir/artifacts/dependency-freeze.txt"
    fi
    cp -a "$discovery_dir" "$run_dir/discovery"
    printf 'run_id=%s\nssmo_project_root=%s\nconfig=%s\nprofile=%s\npython_module=%s\npipeline=%s\nfree_cpus=%s\nfree_mem_gb=%s\nfree_gpus=%s\n' "$run_id" "$SSMO_PROJECT_ROOT" "$config" "$profile" "$python_module" "$pipeline" "$free_cpus" "$free_mem" "$free_gpus" > "$run_dir/submission-manifest.txt"
    printf 'method=%s\nseed=%s\nmanifest=%s\ncheckpoint=%s\nresume=%s\nreport_source=%s\nskip_install=%s\n' "$method" "$seed" "$manifest" "$checkpoint" "$resume" "$report_source" "$skip_install" >> "$run_dir/submission-manifest.txt"
    chmod -R a-w "$source_dir"
    chmod a-w "$frozen_config"
    printf 'stage\tmethod\tseed\tjob_id\tdependency\n' > "$run_dir/jobs.tsv"
fi

dependency=$after
for index in "${!stages[@]}"; do
    stage=${stages[$index]}; this_method=${methods[$index]}
    resources "$stage" || { ssmo_error 'GPU audit requires GPU profile'; exit 2; }
    this_checkpoint=$checkpoint
    [[ -n "$single_stage" ]] || this_checkpoint="$run_dir/artifacts/train-$this_method-seed$seed/best.pt"
    cmd=(sbatch --parsable --account=anakano_81 "--partition=$partition" --ntasks=1 "--cpus-per-task=$cpus" "--mem=${memory}G" "--time=$wall" "--job-name=SSMO-$run_id-$stage-$this_method" "--output=$run_dir/logs/$stage-$this_method-%j.out" "--error=$run_dir/logs/$stage-$this_method-%j.err" "--chdir=$source_dir" "--export=ALL,SSMO_PROJECT_ROOT=$SSMO_PROJECT_ROOT,SSMO_PYTHON_MODULE=$python_module")
    cmd+=("${gpu_args[@]}")
    [[ -z "$dependency" ]] || cmd+=("--dependency=afterok:$dependency")
    cmd+=("$source_dir/slurm/stage.sbatch" "$stage" "$run_dir" "$frozen_config" "$source_dir" "$profile" "$this_method" "$seed" "$manifest")
    if [[ "$stage" = train ]]; then cmd+=("$resume"); else cmd+=("$this_checkpoint"); fi
    cmd+=("$report_source")
    if (( submit )); then
        printf '%q ' "${cmd[@]}" > "$run_dir/logs/submission-$stage-$this_method.command"
        printf '\n' >> "$run_dir/logs/submission-$stage-$this_method.command"
        job_output=$("${cmd[@]}" 2> "$run_dir/logs/submission-$stage-$this_method.err") || {
            rc=$?
            printf '%s\n' "$job_output" > "$run_dir/logs/submission-$stage-$this_method.out"
            printf 'Submission failed for %s (exit %s); previously submitted jobs preserved.\n' "$stage" "$rc" >&2
            cat "$run_dir/logs/submission-$stage-$this_method.err" >&2
            exit "$rc"
        }
        printf '%s\n' "$job_output" > "$run_dir/logs/submission-$stage-$this_method.out"
        job_id=${job_output%%;*}
        [[ "$job_id" =~ ^[0-9]+$ ]] || { ssmo_error "sbatch returned invalid job ID for $stage"; exit 3; }
        printf '%s\t%s\t%s\t%s\t%s\n' "$stage" "$this_method" "$seed" "$job_id" "${dependency:-none}" >> "$run_dir/jobs.tsv"
        printf '%s: job %s\n' "$stage" "$job_id"
        dependency=$job_id
    else
        printf '%q ' "${cmd[@]}"; printf '\n'
        dependency="PREVIOUS_${index}_JOB_ID"
    fi
done
if (( ! submit )); then printf 'Preview only. Recheck discovery/account capacities before adding --submit.\n'; fi
