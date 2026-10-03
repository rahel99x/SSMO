# Current execution evidence

Validation ran in the cloud workspace on **CPU**, with Python **3.12.14**,
PyTorch **2.10.0+cpu**, NumPy **2.2.6**, SciPy **1.15.3** and PyYAML **6.0.2**,
using the project-local `.venv`. `pip check` passed. The standalone local setup
script was exercised against the installed environment; it uses verified
official package indexes, copied venv interpreters and confined caches.

## Initial implementation outcomes

| Check | Result |
|---|---|
| Full test audit | **55 executed, 55 passed, 0 failed/errors/skipped** |
| Exact formulas | Shock/translated-step signs, diffuse/atomic units, rarefaction no-atom control, payoff jumps, differentiated collisions, directional linearity and query normalization passed |
| Tiny FP64 mixed derivatives | Network-weight gradients of weak JVP losses match centered finite differences |
| Checkpoint recovery | Real CPU `SIGUSR1` pause/resume matches uninterrupted model, optimizer and RNG/sampler state; stale crash-log tail is archived; prior best checkpoint survives a fresh recovery directory |
| Read-only source | Model/training tests passed from an unchanged read-only source snapshot |
| Slurm workflow | **16 mocked tests** include exact accounts, IDs/dependencies, dry-run behavior, source snapshots, confined paths, resource caps, copied deployments, queued work and dependency-freeze drift |
| Data | Compact manifest has **47 parents**: 24 train, 8 validation, 8 test, 4 parameter-range holdouts, 3 fixed analytic audits; no dense trajectories |
| Learned smoke | Measure and state-only methods each completed **20 updates**, seed 17, one globally selected validation checkpoint |
| Held-out evaluation | **8 physical parents**, **2,245 raw query/status rows**, shock/constant/fan/collision audit coverage, **6 LP diagnostics** with passing residual checks |
| Event statuses | Pre/post and near-collision times evaluated; exact collision retained as one **unresolved** record |
| Numerical teacher audit | **114 Godunov solves**, 366 raw records, declared forward/FD tolerances passed |
| Actual CARC/GPU execution | **Unrun**: no CARC connection, Slurm runtime or GPU exists in this workspace |

Artifacts are under `runs/local-validation-001/`, including `audit-final/audit.json`,
`data/parents.json`, per-method training logs/checkpoints, `evaluate/evaluation.json`,
raw query errors, representation ladders, `numerical/numeric_teacher_audit.json`
and the trusted inverse report. Failed earlier audits and their diagnostics are
preserved; the copied-deployment Git provenance defect was corrected before the
final passing audit. Run directories are intentionally ignored by Git.

## What these results support

The first-order numerical solver conserves mass with maximum observed residual
**6.44e-15** on the selected controls. Forward query errors decrease over
128/256/512 cells. Finest-grid worst forward errors are approximately **6.42e-6**
for the shock and **1.15e-3** for the fan. Worst finest-grid FD errors over the
selected perturbations are **1.36e-3** and **1.58e-2** respectively. Coarse-grid
fan derivatives can worsen as the FD step shrinks. No convergence plateau,
certified uncertainty bound or inviscid numerical-label promotion is claimed.

The chart implementation preserves internal calculus: observed FP32 additivity
errors are at most about **1.03e-8**, and its own weak finite-variation errors
are at most about **7.37e-6** at the declared FD step. These consistency checks
do not establish PDE accuracy. The unconstrained direct-grid head has an
observed state/tangent consistency error around **0.0347**, which is reported
rather than silently treated as an integrable derivative.

The **20-update learned smoke fails the 0.01 worst-query accuracy gate on most
selected parents**. The measure chart's observed worst error is about **0.0725**;
the state-only chart's is about **0.0854**. The classical degree-two front
regression is essentially exact on this analytic family (worst approximately
**2.55e-15**). No research advantage, matched-budget speedup or confirmatory
sample-size conclusion follows from this smoke.

A bounded learned-gradient inverse run decreased the independently evaluated
trusted tracking objective from **1.7187 to 0.05072** in eight accepted steps,
with zero exact-gradient fallbacks in this run. The line search still used
trusted objective acceptance and counted trusted reference evaluations. This is
a scoped optimization-path check, not parameter identifiability or inverse-task
efficacy evidence.

The per-seed CPU training invocation observed about **370 MB peak host RSS**;
the tiny learning loops completed in under a second after initialization on
this machine. Those are current-machine smoke observations, not CARC walltime
recommendations or GPU throughput measurements. CARC calibration must record
the actual Python module, GPU model/capacity, driver, CUDA kernels, synchronized
complete costs and memory before any expansion. Keep GPU models and model-seed
variation separate, and advance only through the gates in SCIENCE_SCOPE.md.

## Configuration and containment follow-up

After consolidating the configuration under `SSMO_PROJECT_ROOT` and changing
the CARC root to `/home1/aadaniel/projects/SSNO`, the CPU audit executed **62
tests: 62 passed, zero failures, errors or skips**. Regression checks cover
unrelated `PROJECT_ROOT` settings, conflicting legacy roots, and refusing CARC
storage outside the approved directory before creating files. Mocked Slurm
fixtures now use project-confined storage instead of inheriting `TMPDIR`.
The dependency check and submission preview also passed; this remains local
validation, with no live CARC submission.

## Automated workflow and preset selection

The next revision adds the Bash frontend for discovery, allocated setup,
smoke/pilot submission, status and recovery, plus verified reuse of the pinned
Python 3.12.8 CUDA venv. Site headroom remains explicitly reviewed; neither this
cloud instance nor mocked scheduler tests can establish shared-account capacity.
Completed raw evaluation logs are losslessly compressed inside the allocated
report stage; source runs, training logs and checkpoints are preserved.

The final CPU audit executed **107 tests: 107 passed, zero failures, errors,
skips or expected failures** from an unchanged read-only source copy under
`runs/ssmo-automation-audit-20261003T070415Z-140623/`. `pip check` passed.
This includes 24 mocked low-level Slurm tests, 18 frontend tests, eight report
checks, ten lossless-compaction checks and four validation-selection checks.
Real Bash previews for setup/smoke/pilot produced 3/7/9 correctly accounted
jobs with module `python/3.12.8` and L40S requests; no run directories or jobs
were created by those previews. Verified venv reuse removes one installation
job from each chain.

The first frozen-copy audit exposed a test-fixture permission error: copied
requirements were still read-only during a deliberate dependency-drift test.
Only that disposable fixture's mode was corrected; the preserved failed audit
and subsequent clean audit both left their source copies unchanged.

A bounded CPU validation-only search selected width 32, depth two, learning rate
0.003 and batch 32. Two initialization seeds confirmed the selection: 1,315
parameters, 24.14-second mean complete CPU invocation and approximately 355 MiB
maximum RSS. The search, including an excluded failed draft, took 287.89 seconds.
The selected preset reduces the training application cap to 300 seconds while
retaining conservative Slurm resource headroom. See [CONFIG_TUNING.md](CONFIG_TUNING.md)
for the criterion, complete trial table and measured limitations.

One independent assessment then used the frozen preset without retuning.
All 31 IID test and 30 range-holdout supported parents passed the 0.01 weak-query
gate; the fixed audit shock failed at 0.01181979. Thus 61/62 supported parents
passed, but the complete gate remains unmet. The largest nonlinear-gradient error
was 0.04996314; the classical control remained more accurate. Exact collision
records remain unresolved. No global or GPU-specific optimum, research advantage
or actual CARC execution is established.

The selected search records are preserved under `runs/ssmo-config-tuning-002/`
and the independent assessment under `runs/ssmo-heldout-assessment-002/`.
Verified level-one gzip reduced the latter directory from approximately 72.54 MB
to 7.05 MB while retaining every raw record, original summary and checksum
receipt. Generated runs remain ignored by Git; aggregate evidence is committed.

## SSMO directory correction

The user's latest correction sets the project name and approved CARC root to
**SSMO**, `/home1/aadaniel/projects/SSMO`. Active Bash/Python storage guards,
Slurm source paths, site settings and deployment guidance now use that path.
The earlier SSNO audit above is historical evidence, not current deployment
guidance. Explicit regression checks reject the former spelling before creating
caches or files; strict containment remains in force.

The corrected revision passed **108/108 CPU tests**, with zero failures, errors
or skips, from an unchanged read-only source copy under
`runs/ssmo-root-correction-audit-20261003T091909Z-157716/`. `pip check` passed.
Slurm checks remain mocked; no CARC jobs were submitted for this correction.

## Concurrent projects and bounded GPU duration

Following the user's queue-policy update, local project job-count caps and the
manual submission-slot field are removed. The three reviewed free resource
values (CPU, host memory, GPU) remain. Account freshness compares sorted
non-PENDING allocation rows; pending jobs from other projects can be submitted
or removed concurrently. Running/allocation changes still require resource
review. Queued/running SSMO tasks continue protecting their shared venv against
reinstallation.

GPU stage walltimes remain capped at 30 minutes for training, 15 minutes for the
kernel audit and ten minutes for evaluation. One task receives one GPU, and each
pipeline is serial. Smoke and pilot training application budgets remain 120 and
300 seconds. Queue waiting time consumes no allocated GPU time; automatic
retries and unbounded recovery loops are absent.

This revision passed **113/113 CPU tests**, with zero failures, errors, skips
or expected failures, from an unchanged read-only source copy under
`runs/ssmo-duration-audit-20261003T094808Z-184270/`. Regression checks cover
concurrent pending jobs across five other projects, removal of both retired
count settings, allocation changes and the 30-minute maximum GPU walltime for
every supported GPU profile. Slurm checks remain mocked; no CARC jobs were
submitted during this validation.

The user's supplied successful `ssmo-policy-001` discovery confirms the
`anakano_81` association, Python 3.12.8, A100/A40/L40S hardware and an account
queue containing only PENDING jobs at that observation. The runbook records a
conservative four-CPU, 8-GiB, one-GPU envelope for that snapshot; checked-in free
capacity fields stay empty because a historical observation cannot establish
future headroom. A30 and plain L40 were not present in this supplied discovery.

## Shared venv lock on network filesystems

The supplied CARC log records validation job `12617610` failing before tests
with `flock: 9: Bad file descriptor`. It was scheduled after successful install
job `12617609`. The task opened its lock descriptor write-only and requested a
shared lock; network filesystem implementations that map shared flock to a
POSIX read lock can reject this descriptor. The scripts now open the existing
lockfile read/write without truncation. Installation remains exclusive and
computation shared. Actual contention retains exit 3; other flock errors retain
their distinct status and filesystem diagnostic.

Three new regression checks exercise a real POSIX read lock, genuine
shared/exclusive contention, lock lifetime through a mocked separate Slurm task,
and a closed-descriptor error. Restoring the old write-only open in a disposable
fixture reproduced EBADF and failed the regression. The corrected revision
passed **116/116 CPU tests**, with zero failures, errors or skips, using an
unchanged read-only source copy under
`runs/ssmo-lock-audit-20261003T100613Z-214094/`. Its confirmation CLI audit exit
code was recorded as zero and source hashes were rechecked. Mocked Slurm and
POSIX lock checks do not establish successful execution on the CARC filesystem;
the runbook provides a fresh smoke submission that preserves the failed run.

## User-reported CARC smoke and pilot

The user's subsequent accounting rows report all six `ssmo-smoke-002` stages
completed with exit code `0:0`, including validation after the shared-lock fix.
Their one-GPU audit/training/evaluation allocations total 45 seconds. All eight
`ssmo-pilot-001` stages also completed with exit code `0:0`; its five one-GPU
allocations total 221 seconds. These are user-supplied CARC observations, not
cloud-executed jobs or independent access to the remote filesystem.

Extracted pilot summaries show measure training passing 61/62 supported parents,
state-only passing 32/62, and classical front regression passing 62/62. Measure
training's fixed shock error `0.01181978014` exceeds the unchanged `0.01` tolerance.
The complete scientific accuracy gate remains unmet despite successful execution.
See [CARC_PILOT_ASSESSMENT.md](CARC_PILOT_ASSESSMENT.md) for cohort counts, costs,
early stopping, reported memory and evidence limits. No preset, tolerance or
scientific implementation was changed in response to these sealed results.
