# SSMO on USC CARC

This workflow prepares the scalar one-dimensional Burgers sensitivity experiment described in [PROPOSAL.md](PROPOSAL.md). Start with analytic references and a small end-to-end smoke run. The implementation and local CPU checks are preparation evidence; they do not establish CARC access, a submitted job, successful CUDA execution, a speedup, or a trained research result.

The user's latest instruction sets the CARC storage root to **`/home1/aadaniel/projects/SSNO`**, superseding the earlier directory in the supplied [CARC_REQUIREMENTS.txt](CARC_REQUIREMENTS.txt). Keep the checkout, `.venv`, dependency downloads, temporary files, all caches, datasets, logs, reports and checkpoints under that root; do not create or modify files elsewhere, including `/tmp` and `/scratch1`. The repository and environment prefix remain `SSMO`; `SSNO` is the approved CARC directory name. Use username `aadaniel`, explicitly charge **every allocation** to `anakano_81`, and use standalone Python and `python3 -m venv`, never Conda. These requirements override the proposal's generic scratch/Conda examples and its illustrative large training allocations.

## 1. What the stages establish

| Stage | Partition | Tasks | CPUs/task | Host RAM | Walltime | Purpose |
|---|---|---:|---:|---:|---|---|
| `install` | `main` | 1 | 4 | 8G | 00:30:00 | Create the project-local venv; install pinned dependencies and record `pip check`, versions and freeze |
| `validate` | `main` | 1 | 2 | 4G | 00:10:00 | Mathematical identities, query semantics, event controls and tiny differentiation checks |
| `data` | `main` | 1 | 1 | 2G | 00:10:00 | Generate compact analytic parent records, split manifests and reference metadata |
| `gpu-smoke` | `gpu` | 1 | 2 | 8G | 00:15:00 | Verify the allocated GPU and actual CUDA forward/backward work in the same `srun` task |
| `train` | `gpu` | 1 | 2 | 8G | 00:30:00 | Train the smallest chart-consistent pilot within measured memory/runtime limits |
| `evaluate` | `gpu` | 1 | 2 | 8G | 00:10:00 | Evaluate the checkpoint on the same requested GPU profile and record weak-query/baseline diagnostics |
| `report` | `main` | 1 | 2 | 4G | 00:15:00 | Run the representation diagnostic and consolidate the implemented report artifacts |

These are conservative starting requests, not measured capacity claims. Host `--mem` does not request GPU VRAM. The GPU stages request one GPU per task. Do not increase CPU count, RAM, GPU count or walltime until a completed calibration run identifies the limiting resource and live account/partition limits permit the change.

The analytic data stage runs on a CPU: storing compact parameters and generating exact queries on demand avoids dense trajectories and large GPU reservations. Begin with `configs/carc_smoke.yaml`; use `configs/carc_pilot.yaml` only after mathematical and hardware checks pass. The smoke has 24 training, 8 validation, 8 test and 4 out-of-distribution parents, 20 maximum training steps, batch size 8 and a 120-second application budget. The pilot has 512/64/64/32 parents, 500 maximum steps, batch size 32 and a 1,200-second budget. Early stopping can shorten either run. They remain diagnostic experiments, not confirmatory campaigns.

The initial learned experiment is deliberately scoped. The smoke trains the measure method; the pilot also trains a `state_only` control. The report stage supplies the implemented representation diagnostic; it does not establish the proposal's full baseline, numerical-teacher, inverse/design or performance program. Keep those scientific gates pending until their separate experiments exist and pass.

## 2. Deploy the prepared checkout

Copy or clone the prepared implementation into `/home1/aadaniel/projects/SSNO` using an authorized login session. Do not launch the workflow against an empty Git repository. Inspect existing files first and preserve other work. Each Codex cloud task is already isolated; use the existing checkout and do not create a Git worktree unless explicitly requested.

With the existing `/home1/aadaniel/projects` parent directory and an absent or
empty `SSNO` destination, clone the published implementation directly there:

```bash
git clone --branch main https://github.com/rahel99x/SSMO.git /home1/aadaniel/projects/SSNO
```

For an existing clean clone at that path, use `git pull --ff-only` from inside
it. Preserve earlier run directories and their frozen source/configuration.

The Slurm install stage creates the compatible CARC venv in this project.
Earlier validation archives preserve historical code and are not deployment
sources for the current namespace and storage configuration.

On the CARC login node, use only lightweight inspection, editing, discovery, submission and monitoring:

```bash
cd /home1/aadaniel/projects/SSNO
export SSMO_PROJECT_ROOT="/home1/aadaniel/projects/SSNO"
source scripts/carc_env.sh
ssmo_env
```

All custom settings use the `SSMO_` prefix: `SSMO_PROJECT_ROOT`,
`SSMO_DISCOVERY_ID` and `SSMO_RUN_ID`. Generic
`PROJECT_ROOT` is ignored. Legacy `SSMO_ROOT` is accepted only when it agrees
with `SSMO_PROJECT_ROOT` if both are set. Standard variables required by Python,
libraries and Slurm retain their names and point into this project's storage.

The helper sets project-local temporary/cache paths and preserves the caller's shell options. Do not source executable task or submission scripts: those use strict mode and are intended to run as separate Bash processes. Paths and symlinks must resolve inside the project root. The workflow must reject an output path, cache path or symlink escaping that root; do not substitute `/tmp`, `/scratch1` or a shared environment.

Before dependency installation or Python execution, the helper redirects `TMPDIR`, `TMP`, `TEMP`, pip/XDG/PyTorch/compiler/Triton/extension/Matplotlib/CUDA caches and Python bytecode into the project. The user confirmed the CARC module as `python/3.12.8`; this is the default in the submission wrapper and module-loading helper. Load it inside allocated tasks as well as during venv creation, and invoke `.venv/bin/python` explicitly. Discovery must still record its availability, and the pinned PyTorch CUDA 12.6 wheel requires the allocated compatibility audit.

Use `--python-module python/3.12.8` when selecting it explicitly. The module is
loaded by the allocated stage; do not install dependencies on the login node.

If a CARC `.venv` already exists, inspect `.venv/pyvenv.cfg`: installation reuses
that interpreter. If it was created with a different Python module, first wait
for all queued/running SSMO tasks to finish, preserve the old venv under a fresh
backup name inside SSNO, and create the replacement through the Slurm install
stage. Do not replace an active environment.

The allocated install stage uses `python3 -m venv --copies` so the venv interpreter does not escape the root through a standard venv symlink. It installs pinned base requirements and `torch==2.10.0+cu126` from PyTorch's official `https://download.pytorch.org/whl/cu126` index, preserves TLS verification, runs `pip check`, and writes `dependency-freeze.txt` plus `installed-versions.json`. The selected CUDA wheel still needs the actual allocated kernel audit. The venv is shared only within this project: an exclusive installation lock and shared runtime locks protect active scientific tasks. Installation is also refused while another SSMO run has queued/running jobs, with a second queue check inside the install task. Pending tasks hold no lock; each task compares the full installed freeze against its run's recorded freeze before computation, records its fingerprint and stops on drift. If installation reports the venv is in use, allow the existing tasks to finish before retrying; do not overwrite the environment.

## 3. Discover policy before submission

Choose a fresh run identifier for every new attempt. Use only letters, numbers, underscores and hyphens, and retain failed run directories for diagnosis.

```bash
SSMO_DISCOVERY_ID="ssmo-policy-001"
bash scripts/discover_carc.sh --run-id "$SSMO_DISCOVERY_ID"
```

Discovery writes read-only observations under `runs/<policy-id>/discovery`. Use a separate fresh experiment ID: the submission wrapper rejects existing run directories, including discovery IDs. Check `status.tsv`, `manifest.txt` and the command output files for actual username, account associations/QoS limits, quota, `main` and `gpu` partitions, GPU GRES/features and standalone Python modules. A visible partition alone does not establish account authorization. Live submission requires every recorded discovery command to have succeeded, the record to be no more than one hour old and the account queue to match a fresh query. If the queue or policy changes, rediscover using a new policy ID and review capacity again.

The expected inspection commands are:

```bash
id -un
myaccount
myquota
sacctmgr -nP show associations where user=aadaniel account=anakano_81 \
  format=Account,User,Partition,QOS,MaxWall,GrpTRES
sacctmgr -nP show qos \
  format=Name,MaxTRESPerUser,MaxJobsPU,MaxSubmitJobsPU,MaxWall
sinfo -h -o '%P|%a|%G|%l|%c|%m'
sinfo -p gpu -N -h -o '%N|%G|%f|%m|%c'
scontrol show partition main
scontrol show partition gpu
module avail python
squeue -u aadaniel -o '%.18i %.30j %.12T %.10M %.35R'
```

If site policy prevents an inspection command, submission remains blocked until the required discovery evidence can be obtained through a supported correction. Do not fabricate quota, account or GPU authorization. Check existing queued/running jobs and shared account limits. The wrapper's default `--max-project-jobs 8` bounds **queued plus running** jobs whose names start with `SSMO-`; this is not an eight-GPU request. All jobs in a pipeline form one serial `afterok` chain, so only one newly submitted stage is eligible to run at once. The seven-stage smoke fits the default only if existing SSMO jobs leave sufficient slots. The nine-stage pilot needs `--max-project-jobs 10` and sufficient reviewed submission slots. Lower project/account caps when other work shares the account; never submit an unlimited campaign.

## 4. Select GPU requests from live evidence

Supported GPU profile names are `a10040`, `a40`, `a30`, `l40` and `l40s`. The `a10040` profile requests `--partition=gpu --gpus-per-task=a100:1 --constraint=a100-40gb`; live submission requires both the `a100` GRES and `a100-40gb` feature on a recorded node. The other profiles request `--gpus-per-task=a40:1`, `a30:1`, `l40:1` or `l40s:1` respectively and require that exact lowercase GRES tag in discovery. If CARC advertises a different tag, that profile is blocked until the wrapper is corrected against live evidence. The marketing model name is not evidence of the Slurm spelling. Do not guess constraints or silently substitute hardware.

`--gpu-profile cpu` is available for individual CPU diagnostic stages; a complete smoke/pilot pipeline requires an allocated GPU. Training and evaluation default to the selected GPU profile so comparisons record the same hardware.

Record the requested profile and actual device identity separately. Within the GPU allocation, record GPU name, device count, visible-device mapping, reported total VRAM, driver, CUDA runtime, PyTorch version, and MIG/partitioning observations. Preserve Slurm's `CUDA_VISIBLE_DEVICES`; with one visible GPU, use `cuda:0`. A successful `nvidia-smi` command or driver query is not a CUDA kernel test.

Use the same task for hardware validation and the computation. Stop on an unexpected model, multiple visible GPUs or unsupported partitioning. Keep comparison tables separate by actual GPU model; heterogeneous throughput measurements cannot substantiate one universal speed claim. Begin with the tiny FP64 audit and eager FP32 training. Leave at least 20% of reported VRAM available, measure allocated/reserved peaks and whole-process use where available, and reduce the batch/query blocks before enlarging reservations. Do not enable BF16, compilation, activation checkpointing or custom kernels without measured need and accuracy revalidation.

## 5. Preview, then submit explicitly

The wrapper defaults to a dry run. Read the printed allocation commands, absolute config/log/output paths and dependency chain before using explicit submission:

```bash
SSMO_RUN_ID="ssmo-smoke-001"
bash scripts/submit.sh --config configs/carc_smoke.yaml --run-id "$SSMO_RUN_ID" \
  --pipeline smoke --gpu-profile a10040 --seed 17

# Example minima for this serial smoke ONLY if discovery review confirms
# at least these free account capacities; otherwise do not use these values.
bash scripts/submit.sh --config configs/carc_smoke.yaml --run-id "$SSMO_RUN_ID" \
  --pipeline smoke --gpu-profile a10040 --seed 17 \
  --discovery "$SSMO_DISCOVERY_ID" --account-slots 7 --account-free-cpus 4 \
  --account-free-mem-gb 8 --account-free-gpus 1 --submit
```

Preparing these scripts does not authorize live submission from this chat. No CARC jobs were run during local preparation. A printed command is a preview, not a job; only the real ID returned by `sbatch --parsable` records a submission.

The four capacity flags are mandatory for live submission and must reflect manually reviewed free account capacity, including other users/projects: free queued/running submission slots, CPUs, host memory in GiB and GPUs. The serial smoke needs seven submission slots but at most four CPUs, 8G host memory and one GPU at a time. The pilot needs nine slots with the same maximum per-stage resources. The wrapper validates numerical bounds and rechecks the queue; it cannot infer all policy semantics from these supplied numbers. Review associations/QoS/partition limits and quota before setting them. `--python-module python/VERSION` can select a verified alternative standalone module only if discovery observed it.

The submission wrapper creates absolute log directories before `sbatch`, passes `--account=anakano_81` explicitly, uses `main` for CPU stages and `gpu` for GPU stages, and stores real job IDs in `runs/<run-id>/jobs.tsv`. It freezes source under `runs/<run-id>/source` and the selected configuration at `runs/<run-id>/config.yaml`, makes both read-only, and records hashes/Git state. Do not edit the snapshot while jobs are pending or running. No shell-variable expansion is expected in `#SBATCH` directives; variable options belong on the `sbatch` command line. Dry runs do not create directories or snapshots.

The smoke chain is `install → validate → data → gpu-smoke → train(measure) → evaluate(measure) → report`. The pilot inserts `train(state_only) → evaluate(state_only)` before the report. Every link uses `afterok`; installation and validation must succeed before expanding the experiment. This conservative implementation serializes the pipeline. Independently useful CPU work can be submitted with `--stage` under the same discovery/account safeguards when there is reviewed capacity. Do not treat a failed or paused predecessor as satisfied.

A single-stage preview uses the same safeguards:

```bash
bash scripts/submit.sh --config configs/carc_smoke.yaml --run-id ssmo-data-001 \
  --stage data
```

Use `--after JOB_ID` for a stage that requires a newly submitted predecessor. Supply `--manifest PATH` only for a validated compact parent manifest inside the project root. Standalone training/evaluation require an existing manifest; standalone evaluation also requires `--checkpoint PATH`. A path that a predecessor will produce may be absent at submission only when `--after` explicitly names that real predecessor job. The task validates the artifact when it runs. With `--seed 17`, the smoke outputs are:

| Artifact | Path below `runs/<run-id>` |
|---|---|
| Parent manifest | `artifacts/data/parents.json` |
| Measure checkpoints | `artifacts/train-measure-seed17/best.pt`, `last.pt` |
| Measure evaluation | `artifacts/evaluate-measure-seed17` |
| Pilot control checkpoints/evaluation | corresponding `train-state_only-seed17` / `evaluate-state_only-seed17` directories |
| Representation diagnostic | `artifacts/representation` |
| Numerical reference audit | `artifacts/numerical` |
| CPU inverse diagnostic | `artifacts/inverse` |
| Consolidated stage artifacts | `artifacts/report.json` |
| Stage stdout/stderr | `logs/<stage>-<method>-<job-id>.out` / `.err` |

The wrapper trains one explicitly supplied seed per invocation. The pilot config lists seeds `[17, 29, 43]`; these are a replication plan, not three automatic runs. Run additional seeds only after reviewing pilot variability, reuse the immutable parent manifest with standalone stages, and recheck current capacity before each submission. Example pilot preview:

```bash
bash scripts/submit.sh --config configs/carc_pilot.yaml --run-id ssmo-pilot-001 \
  --pipeline pilot --gpu-profile a10040 --seed 17 --max-project-jobs 10
```

Explicit pilot submission additionally needs a fresh `--discovery` record and all four reviewed capacity flags; at least nine available submission slots are required.

## 6. Required scientific gates

1. **Before learning:** pass FP64 translated-step signs, atomic mass without an extra cell-width factor, single-shock parameter derivatives, zero-atom rarefaction controls, direction zero/scaling/additivity, nonlinear payoff-jump formulas, and post-collision event-time derivatives. Label near-event, one-sided and unresolved cases rather than inventing a two-sided derivative.
2. **Before a pilot:** verify weak-query pairings across raster/finite-difference ladders, higher-order/mixed gradient checks on a tiny model, and forward/tangent consistency. Inputs must not read future teacher shock/event labels unless the experiment is explicitly teacher-conditioned.
3. **Before expanding data:** keep all directions, times, resolutions, finite-difference pairs and queries from a physical parent in one split. Seal test parents and choose hyperparameters on validation only. Store reference tiers and solver/formula versions; viscous labels describe a regularized problem, not the inviscid reference.
4. **Before performance claims:** compare with exact/classical front sensitivity and coefficient regression, ordinary state-operator autodiff evaluated weakly, direct grid sensitivity, chart autodiff without special derivative training, and a declared regularization study. Match forward information, tolerance and tuning budget. Include state generation, shock localization, teacher generation and query work in total cost.
5. **Before more complex PDEs/events:** demonstrate a useful benefit on held-out parents and new smooth queries/resolutions, without changing smoothing widths to conceal errors. The proposal's 20% complete-cost reduction is a proposed success threshold, not a current measured outcome. Numerical-teacher, inverse/design and higher-dimensional extensions require their separate reference/convergence gates.

Scalar Burgers analytic controls are correctness evidence. A neural surrogate is not expected to outperform the closed-form formula on its own toy family; a scientific computational advantage needs a nontrivial matched workload. Stop or narrow the scope on persistent direction-linearity errors, inconsistent state/tangent pairs, reference uncertainty larger than the claimed benefit, or unreliable transverse-event handling.

## 7. Data and logs worth preserving

Retain immutable parent/split manifests, compact parameters, reference/formula versions, directions, smooth-query definitions, observation times, event distances and derivative statuses. Avoid storing dense parameter-by-grid Jacobians, batch-by-grid-by-query products, and redundant raster copies. Stream directions and query blocks; cache expensive numerical teachers only with their convergence and uncertainty metadata.

Store raw signed query values and differences in addition to averages: a wrong sign or rare event failure can disappear in a mean. Preserve state/support/atom/diffuse errors, nonlinear-objective derivative errors, finite-difference ladders, total variation, consistency/linearity diagnostics and held-out query results. Report quantiles and worst observed event strata, with physical parents as the independent units.

Every run needs source/config hashes, recorded Git state, installation/software freeze, stage job IDs, stage status/failure reason, total walltime, host RSS and GPU allocated/reserved peaks where measured. `MaxRSS` is host memory; it is not VRAM. Separate data/reference generation, cold startup, encoder, direction and query timings, and full-task walltime. Do not label dry-run budgets as measurements.

Keep thread/worker counts within `SLURM_CPUS_PER_TASK`. Start with no extra data-loader workers and streamed analytic data. Add bounded parallel CPU workers only after profiling. Do not use the login node for tests, dependency installation, data generation or benchmarks.

## 8. Monitor and recover

Use the IDs recorded for this run:

```bash
squeue -u aadaniel -o '%.18i %.30j %.12T %.10M %.35R'
sacct -j JOB_IDS \
  --format=JobID,JobName,Account,State,Elapsed,AllocTRES,MaxRSS,ExitCode
```

Check parent-job state/exit code and relevant `srun` task logs. Preserve failed outputs. For a failed dependency, diagnose the earliest failing prerequisite; rerun only after correcting its cause. A timeout warrants reviewing measured progress/memory and checkpoint behavior, not automatically quadrupling the allocation.

After a failed or paused predecessor, its `afterok` descendants may remain pending with an unsatisfied dependency. They still count toward the project cap and prevent a new installation while that other run remains queued. Inspect each recorded ID and its state; when abandoning that chain, cancel only its individually reviewed blocked descendants. Preserve the run directory and rebuild dependencies from the new successful job IDs.

Training checkpoints must preserve model/optimizer state, random states, sampler position, configuration and split identity, parameter/direction convention and event/chart metadata. The smoke checkpoints every five steps; the pilot every 50 steps. Write a temporary checkpoint beside the target, flush/close it, then atomically replace it while retaining a previous known-good checkpoint. Test loading in a fresh process. Load only trusted project checkpoints.

Resume under a new run ID and preview the resumed training command before explicit submission:

```bash
bash scripts/submit.sh --config runs/ssmo-smoke-001/config.yaml --run-id ssmo-resume-001 \
  --stage train --gpu-profile a10040 --method measure --seed 17 \
  --manifest /home1/aadaniel/projects/SSNO/runs/ssmo-smoke-001/artifacts/data/parents.json \
  --resume /home1/aadaniel/projects/SSNO/runs/ssmo-smoke-001/artifacts/train-measure-seed17/last.pt
```

Replace the example paths with actual trusted artifacts. This is a preview; live resume needs the same discovery/capacity flags and `--submit`. Reuse the previous frozen configuration and verify the checkout's source still matches the checkpoint's recorded source assumptions before taking the new snapshot. Do not edit a prior snapshot. A changed implementation requires a separately validated migration rather than silently continuing the old run. Add `--after NEW_PREREQUISITE_ID` only when the resumed training actually needs a newly submitted prerequisite.

After real resumed training returns its new job ID, submit evaluation under another fresh run ID with `--stage evaluate --method measure --seed 17`, the original `--manifest`, the resumed `--checkpoint .../ssmo-resume-001/artifacts/train-measure-seed17/best.pt`, and `--after NEW_TRAIN_JOB_ID`. Rebuild later dependencies with those new real IDs; the old failed job cannot satisfy `afterok`. Do not pass a dry-run placeholder as a real dependency.

The application has safe-boundary checkpoint handlers, but this wrapper deliberately requests no advance Slurm signals because delivery has not been verified on CARC. Periodic checkpoints remain necessary for abrupt failure. Add an advance-signal request only after a short allocated test establishes delivery and checkpoint behavior.

Cancel only individually reviewed job IDs belonging to this SSMO run. Never run `scancel -u aadaniel` or cancel another project's jobs. Concurrency caps must also account for other allocations charged to the shared account.

## 9. Report execution honestly

Keep these categories distinct: prepared scripts, dry-run preview, local CPU validation, CARC CPU validation, driver observations, actual allocated CUDA execution, trained-model results and measured full-cost comparisons. Report which categories actually completed and attach their run artifacts. This chat has no established CARC access, so the prepared workflow is intended for execution in the user's authorized CARC session. Returned discovery/job logs can then be interpreted to resolve site-specific policy, dependency or hardware failures.
