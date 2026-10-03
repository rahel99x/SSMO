# CARC seed-17 pilot assessment

This assessment records the user's supplied Slurm accounting rows and extracted
training/evaluation summaries for `ssmo-pilot-001`. It is reported CARC evidence,
not a run executed or independently inspected from the cloud workspace. The
source summaries, checkpoints and raw records remain in the user's approved
CARC root, `/home1/aadaniel/projects/SSMO`. No settings are selected or changed
from the exposed held-out results.

## Execution and costs

All eight recorded pilot stages completed with exit code `0:0`. The project venv
was reused. The five one-GPU stages have these reported allocation durations:

| Stage | Job ID | Allocated seconds |
|---|---:|---:|
| GPU audit | 12623556 | 15 |
| Measure training | 12623557 | 51 |
| Measure evaluation | 12623558 | 68 |
| State-only training | 12623559 | 20 |
| State-only evaluation | 12623560 | 67 |
| **Total** | | **221** |

This is **3 minutes 41 seconds**, or approximately **0.0614 allocated GPU-hours**.
It excludes queue waiting, CPU stages and the earlier installation/smokes. GPU
allocation duration is not a measurement of continuously active GPU kernels or
total billed service units. The longest individual GPU stage took 68 seconds.
The hardware audit JSON, full accounting and source/config hashes were not
included in the supplied extract; no cross-GPU performance comparison is made.

| Training measure | Measure | State-only |
|---|---:|---:|
| Application elapsed seconds | 40.38033 | 10.54919 |
| Updates completed | 450 | 175 |
| Selected checkpoint update | 300 | 25 |
| Selected validation score | 7.89480e-7 | 9.42137e-6 |
| Peak PyTorch reserved GPU memory | 66 MiB | 66 MiB |
| Stopping reason | `validation_patience` | `validation_patience` |

Both jobs stopped through the predeclared validation patience, within their
300-second application budgets. Measure training used approximately 3.83 times
the state-only application time under these stopping rules. These durations
have different completed update counts and do not establish a matched-work GPU
speedup. The validation score is state MSE plus weak-sensitivity MSE; it does not
certify a worst-query error bound. Allocator reservation excludes CUDA context
and other allocations and is distinct from host RSS.

## Declared weak-query gate

The unchanged absolute tolerance is **0.01**. Counts below refer to physical
parents supported by the single-front learned charts:

| Cohort | Measure passed | State-only passed | Classical front regression passed |
|---|---:|---:|---:|
| IID constant | 4/4 | 4/4 | 4/4 |
| IID shock | 27/27 | 25/27 | 27/27 |
| Range-holdout shock | 30/30 | 2/30 | 30/30 |
| Fixed analytic shock audit | 0/1 | 1/1 | 1/1 |
| **Total** | **61/62** | **32/62** | **62/62** |

Repeated classical-control rows across the two evaluations refer to the same
parents; they are counted once. All listed groups report zero invalid charts.

| Cohort | Worst measure weak-query error | Worst state-only weak-query error |
|---|---:|---:|
| IID constant | 0.00072835 | 0.00323471 |
| IID shock | 0.00675464 | 0.02158942 |
| Range-holdout shock | 0.00915018 | 0.03728317 |
| Fixed analytic shock audit | **0.01181978** | 0.00811900 |

Measure training improves the held-out sensitivity results relative to the
state-only control in this seed, particularly on the range holdout. Its fixed
shock error exceeds the tolerance by approximately **18.2%**, so the complete
weak-query gate **has not passed**. The classical control's largest listed weak
error is approximately `2.83e-15`; no advantage over that control is established
on this exact single-front family.

Each evaluation reports all three fixed audit parents included and one unresolved
record. Audit inclusion does not imply learned accuracy on all event families.
The extract does not identify the unresolved row, so its event identity is not
asserted here. Rarefaction/collision controls remain reference audits, not learned
transfer evidence. Nonlinear-gradient, direction-consistency and inference-cost
summaries were not supplied and cannot be assessed from these weak-query counts.

## Progression

The measure checkpoint update and weak-error pattern closely reproduce the
earlier CPU seed-17 assessment in [CONFIG_TUNING.md](CONFIG_TUNING.md). This is a
hardware repeat on the same protocol, not an additional independent initialization
seed or dataset. One seed and a finite query/parent bank do not establish research
advantage, complete range coverage or the full bounded-Lipschitz norm.

Preserve both methods' selected/latest checkpoints, raw failures, unresolved
statuses, timing records and frozen provenance. Keep the preset, tolerance and
sealed evaluation protocol fixed. Do not expand the GPU campaign or increase
walltime to compensate for this gate failure. The predeclared seeds 29 and 43,
if later needed for an initialization-robustness assessment, must use unchanged
settings and bounded jobs, retain every outcome and be analyzed separately.
They cannot establish a nontrivial advantage over the exact classical control
on this family. Follow [SCIENCE_SCOPE.md](SCIENCE_SCOPE.md) before broader regimes.
