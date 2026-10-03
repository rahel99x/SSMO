#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/carc_env.sh"
ssmo_env
run_id="discovery-$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ $# -gt 0 ]]; then
    [[ $# = 2 && $1 = --run-id ]] || { ssmo_error 'usage: discover_carc.sh [--run-id ID]'; exit 2; }
    run_id=$2
fi
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || { ssmo_error 'invalid run ID'; exit 2; }
run_dir=$(ssmo_path "runs/$run_id")
[[ ! -e "$run_dir" ]] || { ssmo_error 'run ID already exists'; exit 2; }
ssmo_mkdir "$run_dir/discovery"
status="$run_dir/discovery/status.tsv"
printf 'command\texit_code\n' > "$status"
errors=0
capture() {
    local name=$1 rc=0
    shift
    "$@" > "$run_dir/discovery/$name.txt" 2>&1 || rc=$?
    printf '%s\t%s\n' "$name" "$rc" >> "$status"
    if (( rc != 0 )); then
        printf 'Discovery command %s failed (%s); see %s\n' "$name" "$rc" "$run_dir/discovery/$name.txt" >&2
        errors=$((errors + 1))
    fi
}
capture identity id -un
capture account myaccount
capture quota myquota
capture associations sacctmgr -nP show associations where user=aadaniel account=anakano_81 format=Account,User,Partition,QOS,MaxWall,GrpTRES
capture qos sacctmgr -nP show qos format=Name,MaxTRESPerUser,MaxJobsPU,MaxSubmitJobsPU,MaxWall
capture partitions sinfo -h -o '%P|%a|%G|%l|%c|%m'
capture gpu_nodes sinfo -p gpu -N -h -o '%N|%G|%f|%m|%c'
capture main scontrol show partition main
capture gpu scontrol show partition gpu
capture modules module avail python
capture jobs squeue -h -u aadaniel -o '%i|%j|%T|%a|%C|%m|%b'
capture account_jobs squeue -h -A anakano_81 -o '%i|%u|%j|%T|%C|%m|%b'
printf 'run_id=%s\nssmo_project_root=%s\nobserved_utc=%s\nfailed_commands=%s\n' "$run_id" "$SSMO_PROJECT_ROOT" "$(date -u +%FT%TZ)" "$errors" > "$run_dir/discovery/manifest.txt"
printf 'Discovery saved to %s\n' "$run_dir/discovery"
(( errors == 0 ))
