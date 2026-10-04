# Singular-Sensitivity Measure Operators (SSMO)

This repository implements the first executable stages of the supplied
[proposal](docs/PROPOSAL.md). It studies the scalar Burgers entropy solution
`u_t + (u²/2)_x = 0` on `[-2,2]`, with constant exterior states and observation
times before boundary interaction. A directional sensitivity is a **signed**
measure: a diffuse density plus atoms `(u_left-u_right) Ds[v]` at continuous
shock positions. Atomic masses are evaluated directly against smooth queries.

The implemented learned experiment receives initial-data coefficients and time.
It predicts a one-front state chart; chart JVPs and explicit moving-boundary
assembly supply its derivative. It uses a small eager FP32 MLP, with exact FP64
references and tests. Learned multi-front architectures, numerical-label
training, systems, higher dimensions, and efficacy claims are gated extensions.

## What runs

| Stage | Purpose | Default budget |
|---|---|---|
| Mathematical audit | Signs, query normalization, nonlinear payoff jumps, collision derivatives, mixed weight/parameter gradients, checkpoint recovery | CPU, small FP64 cases |
| Compact data manifest | Disjoint physical parents; all times, directions, queries and grids share their parent's split | Smoke: 24/8/8 train/validation/test, 4 range holdouts |
| Representation study | Measure pairings versus grid FD and fixed physical smoothing ladders | CPU, short bounded ladders |
| Numerical teacher audit | Conservative Godunov refinement and coupled FD error records | CPU, 128/256/512 cells |
| Learned chart smoke | Exercise actual gradient-bearing training, validation selection, logs and checkpoints | 20 updates, one seed, width 32 |
| Pilot | Match state-only and measure-training controls, separately per seed/hardware | Validation-selected width 32, depth 2, 500 updates, 300-second application cap |
| Evaluation/inverse | Raw held-out query errors, parent-level statistics, classical/grid controls, BL LP diagnostics and trusted objective checks | Bounded parent and LP counts |

Neither a successful smoke nor an analytic toy comparison establishes a
learning advantage. The classical polynomial front control contains the exact
single-shock dynamics and should be extremely accurate. Costs include reference
generation, fitting/training, state/support reconstruction, directions and queries.
Different fitting and supervision costs are disclosed. GPU models remain separate
measurement strata.

## Local CPU workflow

Python 3.11 or 3.12 and a project-local `.venv` are required. Use
`bash scripts/setup_local.sh` for the pinned CPU environment. It confines caches
and temporary files to this project and refuses heavy installation on a CARC
login node. No Conda environment is used.

Use `SSMO_PROJECT_ROOT` for this checkout and `SSMO_RUN_DIR` for a study's
artifacts. The generic `PROJECT_ROOT` variable is ignored. The earlier
`SSMO_ROOT` name remains a compatibility alias; if both SSMO root variables
are set, they must agree. Library-required cache and temporary variable names
remain standard, with their paths confined to the SSMO directory.

```bash
cd /absolute/path/to/SSMO
bash scripts/setup_local.sh
export SSMO_PROJECT_ROOT="$PWD"
source scripts/carc_env.sh
ssmo_env

# Select a fresh, project-local directory for each study.
SSMO_RUN_DIR="$SSMO_PROJECT_ROOT/runs/local-smoke-001"
mkdir -p "$SSMO_RUN_DIR"
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli plan --config configs/smoke.yaml
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli audit --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR/audit"
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli generate --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR/data"
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli representation --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR/representation"
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli numerical --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR/numerical"
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli train --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR/measure" --manifest "$SSMO_RUN_DIR/data/parents.json" --method measure --seed 17
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli train --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR/state-only" --manifest "$SSMO_RUN_DIR/data/parents.json" --method state_only --seed 17
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli evaluate --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR/evaluate-measure" --manifest "$SSMO_RUN_DIR/data/parents.json" --checkpoint "$SSMO_RUN_DIR/measure/best.pt"
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli evaluate --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR/evaluate-state-only" --manifest "$SSMO_RUN_DIR/data/parents.json" --checkpoint "$SSMO_RUN_DIR/state-only/best.pt"
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli inverse --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR/inverse" --checkpoint "$SSMO_RUN_DIR/measure/best.pt"
"$SSMO_PROJECT_ROOT/.venv/bin/python" -m singular_sensitivity.cli report --config configs/smoke.yaml --run-dir "$SSMO_RUN_DIR"
```

For CARC use the [inline runbook](docs/CARC_RUNBOOK.md), which contains discovery,
resource budgets, dry-run previews, explicit submissions and recovery commands.
The Bash frontend automates these steps:

```bash
bash scripts/carc.sh discover --run-id ssmo-policy-001
# Review policy/capacity and fill local/carc_site.env from configs/carc_site.env.
bash scripts/carc.sh setup --run-id ssmo-setup-001 --submit
bash scripts/carc.sh status --run-id ssmo-setup-001
# After setup succeeds:
bash scripts/carc.sh smoke --run-id ssmo-smoke-001 --submit
bash scripts/carc.sh status --run-id ssmo-smoke-001
# After checking the smoke:
bash scripts/carc.sh pilot --run-id ssmo-pilot-001 --submit
```

Omit `--submit` to preview. Live actions capture fresh discovery automatically,
reuse a verified venv, and retain explicit shared-account capacity checks.
No local pending-job or submission-count limit is imposed. Other projects can
enqueue concurrently; allocated resource changes require refreshed review.
GPU jobs retain a maximum 30-minute Slurm limit and bounded application work.
The allocated report stage losslessly compresses completed raw JSONL with
verified hashes and a path mapping; training logs and checkpoints remain intact.
The [configuration evidence](docs/CONFIG_TUNING.md) describes the bounded CPU
validation search; these are measured starter settings, not GPU-specific optima.

After the frozen seed-17, seed-29 and seed-43 pilots finish, run
`bash scripts/summarize_pilots.sh` to compare their saved accuracy, stopping and
cost records. It uses only the standard library and the project venv, writes a
fresh review inside SSMO and keeps each initialization seed separate. See the
[runbook](docs/CARC_RUNBOOK.md) for custom run IDs and review criteria.

The [reported three-seed assessment](docs/CARC_REPLICATION_ASSESSMENT.md) finds
measure training passes more weak-query parents than state-only in every seed,
but every seed fails the
fixed shock audit and the classical control passes all supported parents.
The next step is CPU inspection of saved failures with the preset frozen.
Use `bash scripts/diagnose_pilot_failures.sh --review-dir runs/<review-id>` to
reconstruct their signed diffuse/atom errors from existing records. It streams
plain or verified gzip logs, compares both learned methods on those parents,
and writes a fresh contained review without model execution or GPU jobs.
The updated `summarize_pilots.sh` also pairs complete learned/classical endpoint
timings and extracts the already recorded trusted inverse-task results. See the
runbook for a fresh cost-review destination and the limits of those comparisons.
The user-reported three-seed cost review closes this single-front learned pilot:
the full weak gate fails and measured exact/classical endpoints are faster.
Use `scripts/close_pilot.sh` to retain small evidence copies and checksums, with
all original runs preserved; broader learned studies remain gated.

Every allocation charges `anakano_81`; project storage is
`/home1/aadaniel/projects/SSMO`. CPU stages use `main`; GPU stages use `gpu`.
A100 40 GB, A40, A30, L40 and L40S are separate profiles. Only currently observed
GRES/feature labels may be submitted. Runtime audits execute CUDA kernels and
mixed-derivative backward work in the allocated task and enforce the 80% measured
reserved-VRAM budget. Host `--mem` is recorded separately.

## Scientific and artifact contract

Physical parents are the independent split/statistical units. Training reads
only train and validation examples. Validation queries differ from the training
bank and the test bank; one globally selected validation checkpoint answers all
test objectives. Compact manifests hold initial-data parameters, not dense
trajectories, future event labels or teacher supports. Exact and numerical labels
remain separate from model inputs.

Exact shock, rarefaction, constant and two-shock collision references are
implemented. At an exact collision the derivative is conservatively **unresolved**
until its one-sided weak limits have been established; calling a pairing raises.
Near-event results retain their status and event distance. No clipping converts
grazing events into regular gradients. The event/reset calculus is an explicitly
scoped finite-dimensional calculation.

Linear queries use diffuse integrals and atom evaluations. A squared tracking
objective uses the jump in `0.5*(u-target)²`, rather than an arbitrary trace
multiplied by an atom. Query normalization bounds amplitude and Lipschitz constant
in the nondimensional coordinate `(x+2)/4`. LP diagnostics describe the discrete
signed measure and carry solver residuals and diffuse discretization bounds;
they are not rigorous PDE certificates.

Important outputs are immutable parent manifests; raw per-query reference,
prediction and error records; FD/refinement ladders; event statuses; parent-level
quantiles and maxima; training JSONL; software/config/source hashes; actual
hardware and memory observations; and selected/last/previous checkpoints.
Resume restores optimizer and RNG/sampler state, preserves the direction
convention, and rejects changed configuration or parent manifests. A paused
training stage exits nonzero so `afterok` cannot start its dependents. No Slurm
advance signal is requested until delivery has been tested in an allocation.

See [scope and gates](docs/SCIENCE_SCOPE.md) and the current
[validation record](docs/VALIDATION.md). This cloud workspace supplies CPU
validation and prepared Slurm scripts; it does not itself establish CARC access,
account authorization, GPU operation, or publication-ready research results.
