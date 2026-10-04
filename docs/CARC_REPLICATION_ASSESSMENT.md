# Frozen three-seed CARC pilot assessment

The user supplied the successful output of `bash scripts/summarize_pilots.sh`
for `ssmo-pilot-001`, `ssmo-pilot-seed29` and `ssmo-pilot-seed43`. This records
reported CARC evidence, not jobs executed or remote files independently read
from the cloud workspace. The helper reported a common NVIDIA A100-PCIE-40GB
profile and passed CUDA kernel audits for all three seeds.

The retained review is
`/home1/aadaniel/projects/SSMO/runs/ssmo-pilot-review-20261004T000302Z-2641811/`.
Keep its `summary.json`, `summary.txt` and `provenance.json` alongside all three
source runs. The supplied terminal extract has SHA-256
`8b77af30ab32e36611871f2d21793ca76120920790bd5c5a5a3ba37ffd782176`.
That checksum identifies the extract; it does not independently verify the
remote input files or their receipts. The earlier seed-17 accounting remains
in [CARC_PILOT_ASSESSMENT.md](CARC_PILOT_ASSESSMENT.md).

## Results at the unchanged tolerance

The absolute weak-query tolerance remains **0.01**, with no invalid chart
allowed. These are three initializations evaluated on the **same 62 physical
parents**, not 186 independent samples. Repeated classical rows within a seed
are counted once.

| Seed | Measure passed | State-only passed | Classical passed | Measure OOD shocks passed | State-only OOD shocks passed | Measure fixed shock audit passed |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 61/62 | 32/62 | 62/62 | 30/30 | 2/30 | 0/1 |
| 29 | 55/62 | 41/62 | 62/62 | 24/30 | 10/30 | 0/1 |
| 43 | 61/62 | 32/62 | 62/62 | 30/30 | 3/30 | 0/1 |

Every measure seed passes all four IID constants and 27 IID shocks. State-only
passes all constants and 25, 27 and 25 IID shocks respectively. All listed
methods have zero invalid-chart rows. The state-only fixed shock passes only
at seed 17; it fails at seeds 29 and 43.

| Seed | Measure fixed shock weak error | Measure OOD worst weak error | Measure OOD worst nonlinear-gradient error | State-only OOD worst nonlinear-gradient error |
|---:|---:|---:|---:|---:|
| 17 | 0.011819780 | 0.0091501821 | 0.049963344 | 0.43946814 |
| 29 | 0.011571504 | 0.012928587 | 0.061942806 | 0.12648106 |
| 43 | 0.011132227 | 0.0081342293 | 0.031753995 | 0.34534683 |

Measure training passes more supported parents and more OOD shocks than
state-only in each initialization, with variability on the range holdout.
**No measure seed passes
the complete declared weak-query gate.** The fixed audit misses tolerance by
11.3–18.2%; seed 29 also has six OOD failures. The classical front control passes
every supported parent in every seed. A research advantage over that control
has not been established on this exact single-front family. Do not choose seed
17 or 43 after inspecting these results.

Nonlinear-gradient errors are separate from the linear weak-query gate. The
small linear errors do not establish inverse-task efficacy or an acceptable
nonlinear-objective tolerance.

The user's subsequent failure-list extraction identifies `audit-000000` as
the only failed measure parent at seeds 17 and 43. Seed 29 also fails these OOD
parents, with zero invalid chart rows in every listed failure:

| Seed-29 parent | Worst absolute weak-query error |
|---|---:|
| `audit-000000` | 0.011571503628520472 |
| `ood-000004` | 0.010294983430147314 |
| `ood-000005` | 0.012928586769360484 |
| `ood-000007` | 0.012089161157333939 |
| `ood-000010` | 0.01101926340263823 |
| `ood-000014` | 0.012911335448274525 |
| `ood-000026` | 0.012052648899172264 |

## Diagnostics and costs

The measure fixed-shock support errors are 0.053922701, 0.051620054 and
0.050229180 for seeds 17, 29 and 43. Corresponding atom-weight errors are
0.0009517337, 0.011604809 and 0.0048599271. The recurring position error is a
diagnostic hypothesis, not a demonstrated cause of the weak-query failures:
signed diffuse and atomic errors can cancel, and maxima need not describe the
same query.

The fixed audit uses `(uL,uR,a)=(2.0,-0.5,0.1)` at time 0.6. Its left state is
outside both the training interval `[0.1,1.2]` and the declared OOD interval
`[1.3,1.8]`. This identifies an extrapolation case; it does not prove that
extrapolation alone explains the failures or justify removing the audit.

Measure direction additivity errors are at most `3.49e-8`; zero-direction and
scaling errors are zero in the printed summaries. State-chart finite-variation
discrepancies range from `8.47e-6` to `1.23e-5` for measure and from `1.40e-5` to
`1.77e-5` for state-only. These are numerical consistency observations, not
certificates or proof that learned states/supports are accurate.

All three fixed audit parents were included in every evaluation. Each reports
one `unresolved:collision` record. Preserve that event status. Inclusion of
rarefaction/collision reference audits does not establish learned transfer.

| Seed | Method | Training seconds | Updates | Selected update | Stopping reason |
|---:|---|---:|---:|---:|---|
| 17 | Measure | 40.380 | 450 | 300 | Validation patience |
| 17 | State-only | 10.549 | 175 | 25 | Validation patience |
| 29 | Measure | 42.321 | 475 | 325 | Validation patience |
| 29 | State-only | 29.680 | 500 | 500 | Step budget |
| 43 | Measure | 44.372 | 500 | 450 | Step budget |
| 43 | State-only | 10.601 | 175 | 25 | Validation patience |

All reported application times remain below the frozen 300-second training
budget. Peak PyTorch reserved memory is 66 MiB for every training; this excludes
CUDA context and other allocations and is not host RSS. Update counts differ,
so these times are not matched-work speed comparisons. Seed-29/43 Slurm
allocation durations and inference timing records were not included in this
extract; their total billed costs or GPU-hours cannot be calculated from it.

## Next step

The three-seed pilot is complete. Keep the preset, tolerance and sealed protocol
frozen. More seeds or longer A100 allocations are not justified by these
results. Follow [SCIENCE_SCOPE.md](SCIENCE_SCOPE.md) in this order:

1. Inspect the existing per-parent/query records on CPU. Identify the failing
   parent, time, direction and query; retain every seed and all failures. Use
   `artifacts/log-compaction.json` to resolve compressed raw files. Reconstruct
   exact fronts from the recorded initial parameters, then decompose the signed
   weak error into diffuse coefficient, diffuse-boundary displacement,
   atom-weight and atom-position contributions, with a reconstruction residual.
   Compare nonlinear payoff-jump errors separately. This is exploratory failure
   attribution, without training, checkpoint reselection or tolerance changes.
2. Require an explicit downstream accuracy/cost target and strong classical
   controls before another learned study. Any revised model/data protocol must
   select settings on independent validation and reserve a fresh final holdout
   before fitting. Exposed audit/test cases cannot select the revised settings
   or serve as its confirmatory holdout.
3. Prepare a small CPU exact-reference study of two noninteracting shocks, with
   an explicit margin from collision/boundary events. Check FP64 weak queries,
   payoff jumps, direction linearity and state/tangent finite variation against
   classical fronts. This uses the existing pre-collision reference calculus;
   it does not claim a learned multiple-front implementation. Learned collision
   logic, inverse efficacy and broader PDE/GPU campaigns remain gated.

No scientific settings, model code or job limits were changed from this review.
