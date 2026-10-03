# Automated SSMO workflow on USC CARC

Use **`/home1/aadaniel/projects/SSNO`** for every CARC file: checkout, `.venv`,
downloads, caches, temporary files, data, logs, reports and checkpoints. Do not
create or modify files elsewhere, including `/tmp` and `/scratch1`. This latest
user instruction overrides older directories in the supplied requirements.
The GitHub repository and environment prefix remain **SSMO**.

Use username `aadaniel`, account `anakano_81`, standalone module
**`python/3.12.8`**, and Python `venv`; never Conda. Login nodes are for lightweight
inspection, editing, discovery, submission and monitoring. The Bash frontend
runs installation and computational stages through Slurm and `srun`.

## 1. Get the current code

With `/home1/aadaniel/projects` already present and `SSNO` absent or empty:

```bash
git clone --branch main https://github.com/rahel99x/SSMO.git /home1/aadaniel/projects/SSNO
cd /home1/aadaniel/projects/SSNO
```

For an existing clean clone at that path, use `git pull --ff-only` inside it.
Preserve prior outputs and frozen snapshots. Earlier source archives are
historical evidence, not deployment sources for this configuration.

```bash
export SSMO_PROJECT_ROOT="/home1/aadaniel/projects/SSNO"
unset SSMO_ROOT
bash scripts/carc.sh --help
```

`SSMO_ROOT` is only a legacy alias; conflicting SSMO root settings fail. Generic
`PROJECT_ROOT` is ignored. The frontend locates the checkout and prepares each
allocated task's environment automatically. Do not source executable scripts.

## 2. Review site policy and set one local settings file

```bash
bash scripts/carc.sh discover --run-id ssmo-policy-001
mkdir -p local
cp -n configs/carc_site.env local/carc_site.env
```

Inspect `runs/ssmo-policy-001/discovery/`: command status, account associations,
QoS, quota, partitions, GPU GRES/features, Python modules and shared-account
queue. Every discovery command must succeed. An observed GPU or partition alone
does not establish account authorization or available capacity.

Edit **`local/carc_site.env`**, which is ignored by Git and used automatically.
Keep the module as `python/3.12.8`; choose an observed GPU profile. Fill all four
`SSMO_ACCOUNT_*` values with **currently reviewed free account capacity**:

| Setting | Meaning |
|---|---|
| `SSMO_ACCOUNT_SLOTS` | Free queued/running submission slots |
| `SSMO_ACCOUNT_FREE_CPUS` | Free allocatable CPUs |
| `SSMO_ACCOUNT_FREE_MEM_GB` | Free host memory in GiB |
| `SSMO_ACCOUNT_FREE_GPUS` | Free GPUs |
| `SSMO_MAX_PROJECT_JOBS` | Cap on this project's queued plus running jobs; default 10 |

The template leaves capacity values empty deliberately. These are shared-account
headroom, not total node capacity or account maxima. Review them before every
live action and update them when other work changes. The frontend cannot infer
all CARC policy semantics; it never invents these values.

The settings file uses defaults such as `: "${SSMO_ACCOUNT_SLOTS:=}"`; put the
reviewed number after `:=`, or use ordinary `SSMO_ACCOUNT_SLOTS=...` assignments.
Environment or explicit command flags can override the template defaults. An
explicit project-confined `SSMO_SITE_CONFIG` selects another settings file.

| GPU profile | Typed GPU request | Additional requirement |
|---|---|---|
| `a10040` | `a100:1` | `a100-40gb` feature on the same recorded node |
| `a40` | `a40:1` | Exact observed GRES tag |
| `a30` | `a30:1` | Exact observed GRES tag |
| `l40` | `l40:1` | Exact observed GRES tag |
| `l40s` | `l40s:1` | Exact observed GRES tag |

Do not choose a different GPU merely from its marketing name. Keep results
separate by actual model. Slurm's `CUDA_VISIBLE_DEVICES` is preserved and one
visible GPU is addressed as `cuda:0`.

## 3. Preview and run setup

```bash
bash scripts/carc.sh setup --run-id ssmo-setup-001
bash scripts/carc.sh setup --run-id ssmo-setup-001 --submit
```

Without `--submit`, experiment commands only print allocation commands: no
policy records, caches, snapshots or jobs are created. The explicit `discover`
command is the exception and records observations inside SSNO.

With `--submit`, the frontend captures fresh discovery automatically, checks
reviewed capacity, then submits a serial `afterok` chain. Setup runs installation,
CPU validation and allocated GPU kernel/memory checks. Defaults need three
submission slots, at most four CPUs, 8 GiB host memory and one GPU. Each allocation
explicitly charges `anakano_81`.

Installation uses `.venv` with copied interpreters, pinned base dependencies and
`torch==2.10.0+cu126` from the official verified PyTorch index. It records versions,
`pip check`, a full dependency freeze and a requirement/module fingerprint.
An existing venv with a different Python version is refused. After all queued
and running SSMO tasks finish, preserve it under a fresh backup name **inside
SSNO** before creating a replacement; never replace an active environment.

If the current venv's Python version, CUDA wheel, recorded freeze and pinned
requirements match, the frontend reuses it and omits installation. Allocated
stages still compare the actual full freeze before work. `--reinstall` explicitly
requests installation again; `--skip-install` requires verified readiness.
Installation locks and queue checks protect the shared project venv.

```bash
bash scripts/carc.sh status --run-id ssmo-setup-001
```

Wait for successful setup before expanding. The GPU audit executes real forward,
JVP and mixed-derivative backward kernels; driver observations alone do not pass.
Unexpected hardware, multiple visible GPUs and unsupported MIG allocations stop
the workflow. Measured reserved VRAM must stay within 80% of reported capacity.

## 4. Run the small smoke

```bash
bash scripts/carc.sh smoke --run-id ssmo-smoke-001
bash scripts/carc.sh smoke --run-id ssmo-smoke-001 --submit
bash scripts/carc.sh status --run-id ssmo-smoke-001
```

A smoke can also bootstrap directly without a separate setup invocation. Fresh
setup has seven jobs; with a verified venv there are six. It runs validation,
compact CPU data generation, GPU audit, measure training, evaluation and CPU
report diagnostics. Jobs are serial; this is never a multi-GPU campaign.

The smoke remains a 20-update implementation check with batch eight and a
120-second application budget. It uses 24 training, eight validation, eight test
and four range-holdout parents. A short smoke is not a research accuracy claim.

| Stage | Partition | CPUs/task | Host memory | Walltime |
|---|---|---:|---:|---|
| Install | `main` | 4 | 8G | 30 minutes |
| Validate | `main` | 2 | 4G | 10 minutes |
| Data | `main` | 1 | 2G | 10 minutes |
| GPU audit | `gpu` | 2 | 8G | 15 minutes |
| Train | `gpu` | 2 | 8G | 30 minutes |
| Evaluate | `gpu` | 2 | 8G | 10 minutes |
| Report | `main` | 2 | 4G | 15 minutes |

Every stage uses one task; GPU stages request one GPU. These retained resource
requests are conservative starting limits, not measured hardware optima. Do not
raise CPU, RAM, GPU count or walltime without calibration. Host RAM is not GPU
VRAM. Thread counts are capped by `SLURM_CPUS_PER_TASK`.

## 5. Run the validation-selected pilot

After inspecting smoke correctness and allocated hardware results:

```bash
bash scripts/carc.sh pilot --run-id ssmo-pilot-001 --seed 17
bash scripts/carc.sh pilot --run-id ssmo-pilot-001 --seed 17 --submit
bash scripts/carc.sh status --run-id ssmo-pilot-001
```

The pilot uses **width 32, depth two, learning rate 0.003, batch 32, three
directions, eager FP32 and one CPU thread**. It retains 512/64/64/32
train/validation/test/range-holdout parents, 500 maximum updates, validation every
25 updates and checkpointing every 50. Patience is six, `min_delta=1e-8`, and the
application budget is 300 seconds per training invocation.

The measure and `state_only` control use the same architecture and budgets.
The pilot has nine jobs with installation or eight with verified reuse. Use a
project cap allowing that chain and reviewed free submission slots for every job;
peak per-stage requests remain four CPUs, 8 GiB host memory and one GPU.

These settings were selected from a bounded CPU validation-only search and
confirmed on two initialization seeds. They are not a global or GPU-specific
optimum. See [CONFIG_TUNING.md](CONFIG_TUNING.md) for raw selection criteria,
measured costs, alternatives and limitations. Sealed test results never choose
hyperparameters. Each invocation runs one seed; `[17,29,43]` is a replication
plan, not an automatic sweep. Recheck variability and capacity before replication.

## 6. Inspect the important outputs

`status` reads actual IDs from `runs/<run-id>/jobs.tsv`, calls `squeue` and `sacct`,
and prints log/report locations. Inspect the earliest failing stage, its exit
code and `srun` logs. `COMPLETED` establishes execution, not scientific accuracy.
`MaxRSS` is host memory; GPU peaks and actual device identity are in application
artifacts.

| Artifact | Path below `runs/<run-id>` |
|---|---|
| Source/config/software provenance | `source/`, `config.yaml`, hashes, `artifacts/dependency-freeze.txt` |
| Scheduler IDs and recovery settings | `jobs.tsv`, `submission-manifest.txt` |
| Raw stdout/stderr and submission errors | `logs/` |
| Immutable physical-parent manifest | `artifacts/data/parents.json` |
| Measure training and selected/latest checkpoints | `artifacts/train-measure-seed17/` |
| State-only pilot control | `artifacts/train-state_only-seed17/` |
| Raw evaluation errors and statuses | `artifacts/evaluate-<method>-seed17/` |
| Representation/numerical/inverse diagnostics | corresponding directories under `artifacts/` |
| Consolidated summary | `artifacts/report.json` |

The allocated report stage losslessly gzips only known completed raw evaluation
and representation JSONL files in its own run, using fast level-1 compression.
`artifacts/log-compaction.json` maps the original summary paths to `.jsonl.gz`
files and records verified uncompressed SHA256 hashes and byte counts. Training
logs, checkpoints, active Slurm outputs and `--from-run` source artifacts stay
unchanged. Read compressed records with `gzip -cd PATH.jsonl.gz`; avoid expanding
copies unless needed. A failed verification preserves that file's original.

Keep compact manifests, raw signed query errors, event/unresolved statuses,
validation selection, checkpoint/RNG/sampler state, source/config hashes,
software freezes, full costs and memory. Avoid dense trajectories and complete
parameter-by-grid Jacobians. Keep all variants of a physical parent in one
split and preserve failures. Never change a pending/running source snapshot.

## 7. Resume, evaluate or collect a new report

Each recovery command automatically recovers the original frozen config,
manifest, method, seed, module and GPU profile, and checks source/config hashes.
It refuses a changed scientific implementation or dependencies. Preserve both
`last.pt` and its sibling `best.pt`. Use a fresh destination ID each time:

```bash
bash scripts/carc.sh resume --from-run ssmo-pilot-001 --run-id ssmo-resume-001
bash scripts/carc.sh resume --from-run ssmo-pilot-001 --run-id ssmo-resume-001 --submit

bash scripts/carc.sh evaluate --from-run ssmo-resume-001 --run-id ssmo-eval-001
bash scripts/carc.sh evaluate --from-run ssmo-resume-001 --run-id ssmo-eval-001 --submit

bash scripts/carc.sh report --from-run ssmo-eval-001 --run-id ssmo-report-001
bash scripts/carc.sh report --from-run ssmo-eval-001 --run-id ssmo-report-001 --submit
```

Use `--method state_only` to select that control from a pilot. Evaluation can
wait for the recorded training job when its output is not yet present; reports
wait on the source run's final recorded job unless an explicit real `--after`
ID is supplied. Every live action captures new discovery and rechecks capacity.

A new report preserves earlier outputs and includes its explicitly selected
source run's artifacts plus fresh CPU diagnostics. It does not automatically
merge every unrelated or earlier recovery directory; retain those directories
as separate evidence. No automatic job cancellation, run removal, unbounded retries or
Slurm signal requests occur.

Failed `afterok` descendants can remain pending, count toward project caps and
block installation. Inspect individually recorded IDs and cancel only reviewed
blocked descendants of the abandoned chain. Rebuild dependencies using new
successful IDs. Never cancel all jobs belonging to the account or user.
Periodic checkpoints remain necessary because advance signal delivery has not
been verified on CARC.

The low-level `scripts/submit.sh` remains available for individual stages and
explicit advanced flags. Use `bash scripts/carc.sh --help` for the normal path.
Before performance or research claims, pass held-out raw-query accuracy and
reference-uncertainty gates, measure complete allocated costs, keep GPU models
and seeds separate, and follow [SCIENCE_SCOPE.md](SCIENCE_SCOPE.md).
