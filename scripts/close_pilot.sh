#!/bin/bash
# Preserve only the completed pilot's small review evidence; never modify runs.
set -euo pipefail
ssmo_close_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
replication=runs/ssmo-pilot-review-20261004T000302Z-2641811
diagnostics=runs/ssmo-pilot-diagnostics-20261004T003557Z-2892611
costs=runs/ssmo-pilot-cost-review-001
output="runs/ssmo-pilot-closeout-$(date -u +%Y%m%dT%H%M%SZ)-$$"

usage() {
    cat <<'USAGE'
Usage: bash scripts/close_pilot.sh [--replication-review-dir PATH]
       [--diagnostics-dir PATH] [--cost-review-dir PATH] [--output-dir PATH]
Copies existing small review files and the assessment into a fresh directory
inside SSMO, verifies original/copy hashes, and leaves all source runs intact.
No Python, installation, Slurm submission, cancellation or scientific rerun.
USAGE
}
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --replication-review-dir|--diagnostics-dir|--cost-review-dir|--output-dir)
            (( $# >= 2 )) && [[ -n "$2" ]] || { printf 'SSMO: missing value for %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --replication-review-dir) replication=$2 ;;
                --diagnostics-dir) diagnostics=$2 ;;
                --cost-review-dir) costs=$2 ;;
                --output-dir) output=$2 ;;
            esac
            shift 2 ;;
        *) printf 'SSMO: unknown closeout option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
source "$ssmo_close_root/scripts/carc_env.sh"
SSMO_PROJECT_ROOT=${SSMO_PROJECT_ROOT:-$ssmo_close_root}
ssmo_init_root
[[ "$SSMO_PROJECT_ROOT" = "$ssmo_close_root" ]] || { ssmo_error 'closeout must use this SSMO checkout'; exit 2; }

# Inspect the original spelling too: ssmo_path canonicalizes contained aliases.
plain_path() {
    local candidate=$1 component canonical
    [[ "$candidate" != *$'\n'* && "$candidate" != *$'\r'* && "$candidate" != *$'\t'* ]] || { ssmo_error 'control characters in path'; return 2; }
    [[ "$candidate" = /* ]] || candidate="$SSMO_PROJECT_ROOT/$candidate"
    canonical=$(ssmo_path "$candidate") || return 2
    component=$candidate
    while [[ "$component" != / ]]; do
        [[ ! -L "$component" ]] || { ssmo_error "symlink alias refused: $component"; return 2; }
        component=$(dirname -- "$component")
    done
    printf '%s\n' "$canonical"
}
replication=$(plain_path "$replication")
diagnostics=$(plain_path "$diagnostics")
costs=$(plain_path "$costs")
output=$(plain_path "$output")
[[ ! -e "$output" ]] || { ssmo_error "closeout already exists: $output"; exit 2; }
for directory in "$replication" "$diagnostics" "$costs"; do
    [[ -d "$directory" ]] || { ssmo_error "review directory missing: $directory"; exit 2; }
    [[ "$output" != "$directory" && "$output" != "$directory/"* && "$directory" != "$output/"* ]] || { ssmo_error 'closeout overlaps a source review directory'; exit 2; }
done
sources=() copies=()
for group in replication-review diagnostics cost-review; do
    case "$group" in
        replication-review) directory=$replication ;;
        diagnostics) directory=$diagnostics ;;
        cost-review) directory=$costs ;;
    esac
    names=(summary.json summary.txt provenance.json)
    [[ "$group" != diagnostics ]] || names+=(failures.jsonl)
    for name in "${names[@]}"; do
        source_file=$(plain_path "$directory/$name")
        [[ -f "$source_file" ]] || { ssmo_error "plain review file missing: $source_file"; exit 2; }
        sources+=("${source_file#"$SSMO_PROJECT_ROOT/"}")
        copies+=("$group/$name")
    done
done
source_file=$(plain_path docs/CARC_REPLICATION_ASSESSMENT.md)
[[ -f "$source_file" ]] || { ssmo_error 'assessment file missing'; exit 2; }
review_source_count=${#sources[@]}
sources+=("${source_file#"$SSMO_PROJECT_ROOT/"}") copies+=(assessment.md)
created=0
trap 'status=$?; if (( status != 0 && created )); then ssmo_error "partial closeout preserved: $output"; fi; exit "$status"' EXIT
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
created=1
(cd -- "$SSMO_PROJECT_ROOT" && sha256sum -- "${sources[@]:0:review_source_count}") > "$output/source-checksums.sha256"
assessment_checksum=$(cd -- "$SSMO_PROJECT_ROOT" && sha256sum -- "${sources[review_source_count]}")
mkdir -- "$output/replication-review" "$output/diagnostics" "$output/cost-review"
printf 'copy_path\tsource_path_relative_to_project\n' > "$output/source-index.tsv"
for index in "${!sources[@]}"; do
    cp -- "$SSMO_PROJECT_ROOT/${sources[index]}" "$output/${copies[index]}"
    printf '%s\t%s\n' "${copies[index]}" "${sources[index]}" >> "$output/source-index.tsv"
done
cat > "$output/README.md" <<'README'
# Completed pilot evidence

This directory preserves copies of the replication review, signed failure
diagnostic, cost review and assessment. The source index identifies originals.
Raw logs, source snapshots, manifests and checkpoints remain in their original
run directories. This is a small evidence bundle, not a backup of the full runs.

Checksums establish unchanged source bytes during copying and matching copies.
They do not revalidate prior receipts, protocols, scientific results or claims.
No jobs are submitted or cancelled and no policy, model or config is changed.
Preserve failed/partial bundles; choose a fresh output for another attempt.
The original-source manifest covers frozen reviews. The assessment original is
checked during copying, but excluded from that manifest because it may be edited
later. The preserved assessment copy is covered by SHA256SUMS.
README
printf '\nAssessment source SHA256 at copying: `%s`.\n' "${assessment_checksum%% *}" >> "$output/README.md"
(cd -- "$SSMO_PROJECT_ROOT" && sha256sum -c -- "$output/source-checksums.sha256")
for index in "${!sources[@]}"; do
    cmp -s -- "$SSMO_PROJECT_ROOT/${sources[index]}" "$output/${copies[index]}"
done
(cd -- "$SSMO_PROJECT_ROOT" && printf '%s\n' "$assessment_checksum" | sha256sum -c -- -)
(cd -- "$output" && printf '%s  assessment.md\n' "${assessment_checksum%% *}" | sha256sum -c -- -)
(cd -- "$output" && sha256sum -- "${copies[@]}" source-checksums.sha256 source-index.tsv README.md > SHA256SUMS && sha256sum -c -- SHA256SUMS)
printf '\nSSMO: closeout saved: %s\nVerify copies:\n  cd %q\n  sha256sum -c SHA256SUMS\nVerify originals:\n  cd %q\n  sha256sum -c %q\n' "$output" "$output" "$SSMO_PROJECT_ROOT" "$output/source-checksums.sha256"
