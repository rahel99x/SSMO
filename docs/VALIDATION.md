# Current execution evidence

Validation ran in the cloud workspace on **CPU**, with Python **3.12.14**,
PyTorch **2.10.0+cpu**, NumPy **2.2.6**, SciPy **1.15.3** and PyYAML **6.0.2**,
using the project-local `.venv`. `pip check` passed. The standalone local setup
script was exercised against the installed environment; it uses verified
official package indexes, copied venv interpreters and confined caches.

## Observed outcomes

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
