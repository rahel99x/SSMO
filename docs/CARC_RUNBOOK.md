# Automated SSMO workflow on USC CARC

Use **`/home1/aadaniel/projects/SSMO`** for every CARC file: checkout, `.venv`,
downloads, caches, temporary files, data, logs, reports and checkpoints. Do not
create or modify files elsewhere, including `/tmp` and `/scratch1`. The project
name, CARC directory, GitHub repository and environment prefix are **SSMO**.
The latest user correction supersedes the earlier SSNO directory spelling.

Use username `aadaniel`, account `anakano_81`, standalone module
**`python/3.12.8`**, and Python `venv`; never Conda. Login nodes are for lightweight
inspection, editing, discovery, submission and monitoring. The Bash frontend
runs installation and computational stages through Slurm and `srun`.

The user's three-seed pilot, signed diagnostic and cost review are complete.
The current next step is [Tower export](#view-completed-evidence-in-tower);
retain the evidence and frozen settings. [Closeout](#close-the-completed-single-front-pilot)
remains available if its small evidence bundle has not yet been made. Earlier sections document how those
stages were run. Further GPU runs for this pilot are not justified by its
reported error/cost results.

## View completed evidence in Tower

These steps read completed artifacts with the existing Python venv and standard
library. They submit no jobs, load no models and leave Tower and original runs
unchanged. Use the Tower installation already available in your shell.

```bash
cd /home1/aadaniel/projects/SSMO
git pull --ff-only
export SSMO_PROJECT_ROOT="$PWD"
unset SSMO_ROOT
module load python/3.12.8
bash scripts/tower.sh --help
command -v tower

SSMO_TOWER_EXPORT="runs/ssmo-tower-completed-001"
bash scripts/tower.sh import \
  --run ssmo-pilot-001 \
  --run ssmo-pilot-seed29 \
  --run ssmo-pilot-seed43 \
  --artifact-dir runs/ssmo-pilot-review-20261004T000302Z-2641811 \
  --artifact-dir runs/ssmo-pilot-diagnostics-20261004T003557Z-2892611 \
  --artifact-dir runs/ssmo-pilot-cost-review-001 \
  --output-dir "$SSMO_TOWER_EXPORT" \
  --accounting
```

Expected: a fresh export with one attempt per recorded job plus three review
attempts, an `index.json`, a small input-checksum `sources.json`, and captured
accounting stdout/stderr. `--accounting` makes one `sacct` request with a
30-second timeout; it is optional. Omit it if historical accounting is
unavailable. Without accounting, original application evidence determines
known outcomes and missing scheduler/resource facts remain unknown. Existing
destinations are refused; retain partial failures and choose another suffix.
No raw JSONL is copied or expanded, and original checkpoints/source stay intact.

Build planning history from only this export and choose its first evaluation
attempt explicitly from the index:

```bash
bash scripts/tower.sh planning \
  --export-dir "$SSMO_TOWER_EXPORT" \
  --output reports/planning.json
bash scripts/tower.sh list --export-dir "$SSMO_TOWER_EXPORT"

SSMO_TOWER_ATTEMPT="$SSMO_PROJECT_ROOT/$(
  .venv/bin/python -B -S - "$SSMO_TOWER_EXPORT/index.json" <<'PY'
import json
import sys
from pathlib import Path
index = json.loads(Path(sys.argv[1]).read_text())
print(next(row["path"] for row in index["attempts"] if row["stage"] == "evaluate"))
PY
)"
bash scripts/tower.sh validate --attempt-dir "$SSMO_TOWER_ATTEMPT"
bash scripts/tower.sh launch \
  --attempt-dir "$SSMO_TOWER_ATTEMPT" \
  --planning-file reports/planning.json
```

Expected: native Tower contract validation succeeds, and the Experiment view
shows the selected attempt's imported observations and scientific results.
The native Log view uses exact grouped scheduler, application, training and
science paths when its selected job matches the inventory's actual `job_id`.
Imported attempts do not reconstruct separate application captures; original
terminal output remains in the indexed scheduler logs. Historical observations are labeled `imported_summary` at
export time: they are not reconstructed live curves or historical ETAs. The
three-seed review preserves per-seed accuracy and paired cost ratios. A file
contract pass is operational evidence; the pilot's complete weak gate still
fails. Three seeds do not justify calibrated resource predictions or scaling
claims. If `reports/planning.json` already exists, choose another contained
filename and pass that filename to `launch`; deliberate replacement requires
`planning --replace`.

Share the import/planning counts, the native validation result and any missing
log warning. A schema check in the cloud does not establish this native CARC
check. If `tower` is absent from PATH, activate your existing installation and
repeat only `validate`/`launch`; no SSMO dependency installation is needed.

For a future authorized new pipeline, reporting is automatic. Its concrete
attempts live in `runs/<run-id>/tower/`; pass the chosen absolute directory to
`validate`/`launch`. Each sbatch job uses that writable directory as its WorkDir,
with an explicit one-node/one-task allocation and read-only source snapshot.
Scheduler stdout/stderr stay in `runs/<run-id>/logs/`; actual task stdout/stderr
are separate files inside the attempt. Training metrics report measured losses,
validation, timing and memory; evaluation/representation report physical-parent
progress. Optional native GPU CSV traces sample one verified allocated physical
GPU at 60 seconds. Set `SSMO_TOWER_GPU_TRACE=0` before a future submission to
disable them. Unsupported selectors or unknown node counts skip sampling with
a recorded reason; no extra scheduler steps are started to poll utilization.
An exit-75 pause stays interrupted, and forced kills can leave incomplete
reports until actual scheduler evidence is imported into a fresh export.

Do not resume/evaluate old pre-integration source snapshots using a changed
checkout: existing source-recovery guards intentionally refuse that mismatch.
The import path above reads their completed evidence without changing frozen
scientific bytes. Read [TOWER_FORMATS.md](TOWER_FORMATS.md) for paths, limits,
measurement scopes, contract/schema provenance and the absence of unmeasured
workflow/scaling recipes.

## 1. Get the current code

With `/home1/aadaniel/projects` already present and `SSMO` absent or empty:

```bash
git clone --branch main https://github.com/rahel99x/SSMO.git /home1/aadaniel/projects/SSMO
cd /home1/aadaniel/projects/SSMO
```

For an existing clean clone at that path, use `git pull --ff-only` inside it.
Preserve prior outputs and frozen snapshots. Earlier source archives are
historical evidence, not deployment sources for this configuration.

```bash
export SSMO_PROJECT_ROOT="/home1/aadaniel/projects/SSMO"
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
Keep the module as `python/3.12.8`; choose an observed GPU profile. Fill the three
`SSMO_ACCOUNT_FREE_*` values with **currently reviewed free resource capacity**:

| Setting | Meaning |
|---|---|
| `SSMO_ACCOUNT_FREE_CPUS` | Free allocatable CPUs |
| `SSMO_ACCOUNT_FREE_MEM_GB` | Free host memory in GiB |
| `SSMO_ACCOUNT_FREE_GPUS` | Free GPUs |

The template leaves capacity values empty deliberately. These are shared-account
headroom, not total node capacity or account maxima. Review them before every
live action and update them when other work changes. The frontend cannot infer
all CARC policy semantics; it never invents these values.

The settings file uses defaults such as `: "${SSMO_ACCOUNT_FREE_CPUS:=}"`; put the
reviewed number after `:=`, or use ordinary `SSMO_ACCOUNT_FREE_CPUS=...` assignments.
Environment or explicit command flags can override the template defaults. An
explicit project-confined `SSMO_SITE_CONFIG` selects another settings file.

The supplied `ssmo-policy-001` snapshot, observed on 2026-10-03, confirms
`aadaniel`'s `anakano_81` association, `python/3.12.8`, and A100, A40 and L40S
nodes in the `gpu` partition. All four account queue entries were PENDING;
none showed an active allocation. For that reviewed snapshot, a conservative
resource envelope for one serial SSMO pipeline is:

```bash
SSMO_PYTHON_MODULE=python/3.12.8
SSMO_GPU_PROFILE=a10040
SSMO_ACCOUNT_FREE_CPUS=4
SSMO_ACCOUNT_FREE_MEM_GB=8
SSMO_ACCOUNT_FREE_GPUS=1
```

Use those assignments in `local/carc_site.env` after confirming the observations
remain current. These numbers cover the pipeline's peak requests; they do not
claim total free cluster capacity or reserve resources. Slurm determines when
hardware becomes available. The observed `normal` QoS permits 2,000 CPUs and
36 GPUs per user; the `gpu` partition QoS includes per-user limits of 400 CPUs,
36 GPUs in total, 12 A100, 12 A40 and six L40S. The serial pipeline requests at most four CPUs,
8 GiB and one GPU. A30 and plain L40 remain supported profiles, but neither was
observed in this snapshot; choose them only after fresh discovery confirms them.

There is no project job-count cap or manual submission-slot setting. Other
projects can enqueue concurrently. Pending-job additions, removals and queue
ordering do not invalidate resource review; changes to non-pending allocations
require refreshed CPU/memory/GPU headroom. CARC's scheduler enforces site limits.

For an existing local settings file, remove the retired count settings:

```bash
sed -i '/SSMO_MAX_PROJECT_JOBS/d; /SSMO_ACCOUNT_SLOTS/d' local/carc_site.env
```

Stale environment values for those retired settings are ignored by the frontend.

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
command is the exception and records observations inside SSMO.

With `--submit`, the frontend captures fresh discovery automatically, checks
reviewed capacity, then submits a serial `afterok` chain. Setup runs installation,
CPU validation and allocated GPU kernel/memory checks. Setup submits three jobs
initially and needs at most four CPUs, 8 GiB host memory and one GPU. Each allocation
explicitly charges `anakano_81`.

Installation uses `.venv` with copied interpreters, pinned base dependencies and
`torch==2.10.0+cu126` from the official verified PyTorch index. It records versions,
`pip check`, a full dependency freeze and a requirement/module fingerprint.
An existing venv with a different Python version is refused. After all queued
and running SSMO tasks finish, preserve it under a fresh backup name **inside
SSMO** before creating a replacement; never replace an active environment.

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

A smoke can also bootstrap directly without a separate setup invocation. A fresh
smoke has seven jobs; with a verified venv there are six. It runs validation,
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

Every stage uses one task; GPU stages request one GPU. The longest GPU allocation
is **30 minutes**; GPU audit and evaluation have **15- and 10-minute** limits.
Smoke training has a 120-second application budget and pilot training 300 seconds.
Pending queue time consumes no allocated GPU time. Each pipeline is serial;
there are no automatic retries or repeated resume loops. These retained resource
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
The pilot has nine jobs with installation or eight with verified reuse. Review
current free CPU/memory/GPU resources before submitting;
peak per-stage requests remain four CPUs, 8 GiB host memory and one GPU.

These settings were selected from a bounded CPU validation-only search and
confirmed on two initialization seeds. They are not a global or GPU-specific
optimum. See [CONFIG_TUNING.md](CONFIG_TUNING.md) for raw selection criteria,
measured costs, alternatives and limitations. Sealed test results never choose
hyperparameters. Each invocation runs one seed; `[17,29,43]` is a replication
plan, not an automatic sweep. Recheck variability and capacity before replication.

### Review the completed three-seed pilot

After `ssmo-pilot-001` (seed 17), `ssmo-pilot-seed29` and `ssmo-pilot-seed43`
finish, collect their existing results with one Bash command:

```bash
cd /home1/aadaniel/projects/SSMO
git pull --ff-only
export SSMO_PROJECT_ROOT="$PWD"
unset SSMO_ROOT
bash scripts/summarize_pilots.sh
```

This is lightweight JSON inspection using the standalone Python module and
project `.venv`. It submits no Slurm jobs and imports no GPU libraries. The
printed summary is also saved as `summary.txt`, alongside `summary.json` and
an input-checksum receipt in `provenance.json`, under a fresh
`runs/ssmo-pilot-review-.../` directory. Existing runs and checkpoints are kept.

For other run IDs or a chosen fresh destination:

```bash
bash scripts/summarize_pilots.sh \
  --run ssmo-pilot-001 --run ssmo-pilot-seed29 --run ssmo-pilot-seed43 \
  --output-dir runs/ssmo-pilot-review-001
```

The helper derives each seed from recorded training jobs and checks compatible
scientific configuration, parent manifests and evaluation protocols. Missing
results, inconsistent comparisons or an existing destination are errors, not
silently skipped runs. Each seed's measure/state-only counts, fixed audit,
stopping reason, application time and error diagnostics stay separate.
Classical-control rows repeated across the two evaluations are counted once.
The same physical parents across three seeds are not three times as many
independent data samples. A passing weak-query gate alone does not establish a
research advantage or complete nonlinear/event accuracy.

Review all seeds before changing the model. Diagnose support, atom-weight,
nonlinear-gradient and direction-consistency errors using the saved records.
Keep the preset and tolerance frozen; a changed scientific protocol requires
validation and a fresh final holdout. Prepare exact-reference multiple-front
tests before progressing to learned collision logic or larger PDE regimes.

The user's completed frozen review reports measure passes of 61/62, 55/62 and
61/62 at seeds 17, 29 and 43; all three fail the fixed shock audit at tolerance
0.01. Classical front regression passes 62/62 in each seed. The review is saved
under `runs/ssmo-pilot-review-20261004T000302Z-2641811/`. Preserve it and the three
source runs; these replications are complete and need no rerun. The subsequent
signed diagnostic and cost review below are also complete. Follow the closeout
decision in
[CARC_REPLICATION_ASSESSMENT.md](CARC_REPLICATION_ASSESSMENT.md) before further
learned experiments. The assessment does not change any submission settings.

### Inspect frozen failed parents on CPU

The failed-parent lists identify `audit-000000` in every seed and six additional
OOD parents in seed 29. Inspect their already recorded query predictions with:

```bash
cd /home1/aadaniel/projects/SSMO
git pull --ff-only
export SSMO_PROJECT_ROOT="$PWD"
unset SSMO_ROOT
bash scripts/diagnose_pilot_failures.sh \
  --review-dir runs/ssmo-pilot-review-20261004T000302Z-2641811
```

This is lightweight artifact inspection with Python 3.12.8 and the project
venv. It imports no NumPy/Torch, loads no checkpoints, executes no model and
submits no Slurm jobs. It streams the existing plain/gzip query records, verifies
review and input receipts, and writes a fresh contained diagnostics directory.
The source runs, selected checkpoints and declared 0.01 tolerance remain fixed.

The diagnostic compares measure and state-only on the same measure-failure
parents, retaining every recorded time/direction/query for those parents. The
expected selection is one parent for seed 17, seven for seed 29 and one for
seed 43, reported separately. The printed worst-row details include signed
diffuse-coefficient, diffuse-boundary, atom-weight and atom-position terms.
These terms telescope in a fixed order anchored at the reference support;
they can cancel and are not a unique causal decomposition. Nonlinear errors
are reconstructed separately using base traces and the payoff jump, with six
signed terms retained in the JSON outputs.

Each method's worst query may differ. Use `failures.jsonl` for comparisons at
the same parent/time/direction/query. Other state-only failures, classical
control rows and unseen parents are outside this targeted diagnostic. Zero
directions are validated and retained but excluded from the original nonzero
gate maximum.

The numerical reconstruction check uses `1e-10 * max(1, abs(values))`; it checks
agreement with the stored scalar calculations and does not change the physical
weak-error tolerance. Missing/mismatched records, changed formulas, invalid or
unresolved selected charts and corrupted gzip receipts are errors. Preserve a
failed diagnostics directory if a later source-change check fails; use a fresh
destination for any corrected rerun.

Read `summary.txt` in the printed destination and retain its `summary.json`,
`failures.jsonl` and `provenance.json`. Share the printed summary before choosing
another learned experiment. This is exploratory inspection of exposed failures;
changed settings still require independent validation and a fresh final holdout.

### Review existing endpoint costs and trusted inverse results

The user completed the failure diagnostic and preserved
`runs/ssmo-pilot-diagnostics-20261004T003557Z-2892611/`. Its ledger shows repeated
audit support-position errors and seed-29 OOD atom-weight errors. Keep that
result; no repeat of the diagnostic or pilot stages is needed.

Collect the next evidence from the existing evaluation summaries:

```bash
cd /home1/aadaniel/projects/SSMO
git pull --ff-only
export SSMO_PROJECT_ROOT="$PWD"
unset SSMO_ROOT
bash scripts/summarize_pilots.sh \
  --output-dir runs/ssmo-pilot-cost-review-001
```

Use another fresh output name if that destination already exists. This reads
small saved JSON files using the existing Python 3.12.8 venv, submits no jobs
and runs no model or optimizer. The original review, diagnostic and source runs
remain intact. The same `summary.json`, `summary.txt` and `provenance.json`
outputs now include cost and inverse sections.

Expected: separate summaries for seeds 17, 29 and 43, with complete endpoint
timings matched by physical parent, number of directions and number of queries.
Warm repeated-call medians and recorded first-call timings are kept separate;
the first call is not a process/model cold start. Learned endpoints include
the forward chart, directional assembly, requested queries and host transfer.
Classical/exact controls run on CPU; the learned charts run on the recorded GPU.
Positive learned/control ratios above one mean the recorded learned endpoint
was slower on that paired workload. These measurements do not establish an
accuracy-qualified speedup while the complete weak gate fails, and overlapping
component measurements must not be added together.

Control `fit_seconds` includes label generation and all classical/grid fits;
it is not isolated classical-front training time. The fixed inverse comparison
reports trusted initial/final objectives, accepted steps, status, reference
evaluations, exact-gradient fallbacks and application time. Exact diagnostic
gradients and trusted objective acceptance are used even with learned proposals;
the timings are audited task costs, not autonomous-surrogate speedups. A single
fixed initialization and tracking target do not establish parameter recovery.
Missing costs are marked unavailable rather than replaced with zero.

Share the printed cost/inverse sections. Keep tolerance and settings frozen;
review this final existing evidence before any new scientific protocol.

### Close the completed single-front pilot

The user has now completed `runs/ssmo-pilot-cost-review-001/`. On the recorded
3-direction/6-query workload, measure endpoints are 16.21–16.86 times slower
than exact CPU fronts and 11.12–11.37 times slower than classical front
regression. The inverse audit reaches similar trusted objectives but uses
20.45–21.15 times the exact task time, including exact diagnostic gradients.
Classical passes every supported parent; every measure seed fails the full
weak gate. Read [CARC_REPLICATION_ASSESSMENT.md](CARC_REPLICATION_ASSESSMENT.md).

Close this learned single-front pilot and retain correctness/representation
evidence. Preserve the frozen settings, all source runs and selected/latest
checkpoints. No new jobs or repeated diagnostic/timing stages are needed.

```bash
cd /home1/aadaniel/projects/SSMO
git pull --ff-only
export SSMO_PROJECT_ROOT="$PWD"
unset SSMO_ROOT
bash scripts/close_pilot.sh \
  --output-dir runs/ssmo-pilot-closeout-001
```

The defaults point to the completed replication review
`runs/ssmo-pilot-review-20261004T000302Z-2641811/`, diagnostic
`runs/ssmo-pilot-diagnostics-20261004T003557Z-2892611/` and cost review
`runs/ssmo-pilot-cost-review-001/`. Override the corresponding options only
when retaining another already completed review. Use a fresh output name if
the destination exists.

Expected: a new contained directory with small review copies, `assessment.md`,
`source-checksums.sha256`, `SHA256SUMS` and an evidence index. The Bash-only
command checks copied hashes and unchanged originals; it runs no Python,
training, optimizer or Slurm job. It does not copy full raw logs/checkpoints,
delete anything, change submission limits or cancel queued jobs. It preserves
partial output if a later check fails. These checks identify the files at
closeout time; they do not independently revalidate past provenance receipts,
reproduce the measurements or provide a full backup of the source runs.

Verify the retained copies and the original small review files:

```bash
(
  cd /home1/aadaniel/projects/SSMO/runs/ssmo-pilot-closeout-001
  sha256sum --check SHA256SUMS
)
sha256sum --check runs/ssmo-pilot-closeout-001/source-checksums.sha256
```

Expected: every entry prints `OK`. Keep the originals where they are. Share the
closeout confirmation only if a check fails; the scientific review is complete.
A new learned study first needs a specific downstream accuracy/cost requirement
and a regime where strong classical controls leave a material unmet need, with
independent validation and a fresh final holdout declared before fitting.
Until then, use the implemented exact reference for this analytic family and
retain the classical front regression as its learning comparison.

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
| Per-job Tower inventory, log index, metrics, final summary and compact analytics | `tower/<attempt>/` |
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

Failed `afterok` descendants can remain pending and protect the shared venv
against reinstallation. Inspect individually recorded IDs and cancel only reviewed
blocked descendants of the abandoned chain. Rebuild dependencies using new
successful IDs. Never cancel all jobs belonging to the account or user.
Periodic checkpoints remain necessary because advance signal delivery has not
been verified on CARC.

If validation stops with `flock: 9: Bad file descriptor`, the earlier Bash
implementation opened the lock write-only before requesting a shared lock.
Network filesystem read locks can reject that descriptor. The corrected scripts
open it read/write without truncating it and distinguish actual lock contention
from descriptor/filesystem errors. Keep `local/venv.lock`: removing the file can
allow two processes to lock different inodes and bypass protection.

For the reported validation failure `12617610` in `ssmo-smoke-001`, preserve the
original run and use a fresh source snapshot after pulling the correction:

```bash
cd /home1/aadaniel/projects/SSMO
git pull --ff-only
export SSMO_PROJECT_ROOT="$PWD"
unset SSMO_ROOT

mapfile -t ssmo_blocked_jobs < <(
  awk -F '\t' -v failed=12617610 \
    'NR>1 && ($5 == failed || blocked[$5]) {blocked[$4]=1; print $4}' \
    runs/ssmo-smoke-001/jobs.tsv
)
if (( ${#ssmo_blocked_jobs[@]} )); then
  scancel --state=PENDING "${ssmo_blocked_jobs[@]}"
fi

bash scripts/carc.sh smoke --run-id ssmo-smoke-002 \
  --account-free-cpus 4 --account-free-mem-gb 8 --account-free-gpus 1 --submit
bash scripts/carc.sh status --run-id ssmo-smoke-002
```

The cancellation targets only pending descendants recorded in this failed
pipeline. Its installation completed successfully because validation started
through `afterok`; the frontend reuses the verified environment. Refresh the
reviewed resource values if current allocations changed. This failure preceded
data generation and training, so restart the smoke rather than resume from a
checkpoint. A Git pull does not modify the failed run's frozen source.

The low-level `scripts/submit.sh` remains available for individual stages and
explicit advanced flags. Use `bash scripts/carc.sh --help` for the normal path.
Before performance or research claims, pass held-out raw-query accuracy and
reference-uncertainty gates, measure complete allocated costs, keep GPU models
and seeds separate, and follow [SCIENCE_SCOPE.md](SCIENCE_SCOPE.md).
