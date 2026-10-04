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

## Reported signed failure attribution

The user subsequently supplied the successful output of
`scripts/diagnose_pilot_failures.sh`. Its retained directory is
`/home1/aadaniel/projects/SSMO/runs/ssmo-pilot-diagnostics-20261004T003557Z-2892611/`.
The terminal extract has SHA-256
`d2d57daa33f1a51bc99c09be0b0f9f3a9220aa8fe3bcf6f3f0261188f150cf1c`.
This remains user-reported CARC evidence; the cloud workspace has not read the
remote raw records or receipts directly. Existing source runs and settings were
preserved, and all selected rows were reconstructed for both methods.

The repeated audit's worst measure query is the same bump query (index 2), at
time 0.6 and direction 0, in every seed:

| Seed | Exact support | Predicted support | Signed weak error | Atom-position term |
|---:|---:|---:|---:|---:|
| 17 | 0.55 | 0.4960772991 | -0.01181978014 | -0.011006509 |
| 29 | 0.55 | 0.4983799458 | -0.01157150363 | -0.010449785 |
| 43 | 0.55 | 0.4997708201 | -0.01113222719 | -0.010263674 |

Atom-position displacement is the largest term in this reference-anchored
telescoping ledger. Its magnitude is roughly 90–93% of the net signed weak
error. This is evidence about the recorded query calculation, not a unique
causal explanation of how the network learned its chart.

Seed 29's six OOD worst queries instead have atom-weight terms as the largest
contributions. Every listed row is the same bump query at time 0.65; the selected
directions differ by parent:

| Parent | Signed weak error | Atom-weight term | Atom-position term |
|---|---:|---:|---:|
| `ood-000004` | +0.01029498343 | +0.012054024 | -0.0021850358 |
| `ood-000005` | -0.01292858677 | -0.010088645 | -0.0026615789 |
| `ood-000007` | +0.01208916116 | +0.0099622217 | +0.0019777251 |
| `ood-000010` | -0.01101926340 | -0.0076921551 | -0.0033532507 |
| `ood-000014` | +0.01291133545 | +0.011896619 | +0.00097309497 |
| `ood-000026` | +0.01205264890 | +0.0098238976 | +0.0020314518 |

Position contributions reinforce or partly cancel the mass error. A universal
localization explanation would therefore be unsupported. The six measure OOD
weak maxima improve on state-only on these parents, but the nonlinear maxima
do not uniformly improve: state-only has smaller maxima on `ood-000004` and
`ood-000010`. Each method's printed worst query may differ; matched comparisons
must use the retained `failures.jsonl` rows.

The largest printed reconstruction residuals over both methods are approximately
`6.6e-17` for weak errors and `2.1e-15` for nonlinear errors. This confirms
agreement with the stored calculations; it does not make the learned predictions
accurate. The measure fixed-audit nonlinear errors are 0.01196970452,
0.02324352706 and 0.01489061758. The linear tolerance 0.01 does not define a new
nonlinear-objective gate. Zero-direction rows are retained; this targeted review
does not cover every state-only failure, classical row or unseen parent.

The next evidence to inspect is already in each frozen `evaluation.json`:
paired complete endpoint timings, aggregate control-fitting costs and the
trusted inverse-task results. The report stages already exercised exact
pre-collision reference controls, so repeating that identical CPU job is not
needed. Keep the existing diagnostics and inspect costs before another study.

## Progression

The three-seed pilot is complete. Keep the preset, tolerance and sealed protocol
frozen. More seeds or longer A100 allocations are not justified by these
results. Follow [SCIENCE_SCOPE.md](SCIENCE_SCOPE.md) in this order:

1. Preserve the completed signed diagnostic above and inspect existing complete
   endpoint/inverse costs with the updated `scripts/summarize_pilots.sh`. Compare
   the same parent/direction/query workload within each seed; keep first-call
   timings separate and never sum overlapping component measurements. Control
   fitting includes exact-label generation and all three classical/grid fits,
   not an isolated front-only fit. The inverse audit computes trusted gradients
   every iteration, even when using learned proposals; its timings do not
   establish autonomous-surrogate speedup or parameter recovery.
2. Require an explicit downstream accuracy/cost target and strong classical
   controls before another learned study. Any revised model/data protocol must
   select settings on independent validation and reserve a fresh final holdout
   before fitting. Exposed audit/test cases cannot select the revised settings
   or serve as its confirmatory holdout.
3. Broader two-front arrangement studies need a new declared protocol and an
   explicit margin from collision/boundary events. Existing report audits already
   exercise pre-collision FP64 reference calculus; a repeat is not learned
   multiple-front evidence. If the complete error/cost and downstream review
   supplies no material benefit, narrow this experiment to correctness and
   representation diagnostics. Learned collision logic, inverse efficacy and
   broader PDE/GPU campaigns remain gated.

No scientific settings, model code or job limits were changed from this review.
