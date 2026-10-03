# Bounded validation-selected starter configuration

`configs/carc_pilot.yaml` now uses width 32, depth 2, learning rate 0.003,
batch size 32, three directions, eager FP32 and one CPU thread. It retains
500 maximum updates, 512 training parents, 64 validation parents, 64 sealed test
parents and 32 range holdouts. Validation occurs every 25 updates, checkpoints
every 50, with patience six and an improvement threshold of `1e-8`. The
application budget is reduced from 1,200 to 300 seconds per invocation.

This is a conservative starter selected by the bounded experiment below.
It is not a global optimum or a measured A100/A40/A30/L40/L40S optimum. CUDA
execution, host and GPU memory, throughput and held-out accuracy require the
allocated CARC smoke and pilot. The existing resource headroom is retained;
small CPU RSS does not justify assuming the same CUDA host footprint.

## Selection protocol

All candidates used the same physical-parent manifest, data seed 1729, times
`[0.15,0.4,0.65]`, three directions, loss weights, maximum 500 updates and
fixed independent validation bank already implemented in `training.py`.
The score is **saved-best validation state MSE plus weak-sensitivity MSE**;
it is not a worst-query error or a claim that the final accuracy gate passes.
No sealed test/OOD evaluation, final query bank, inverse outcome or benchmark
result was used to choose settings.

The checkpoint threshold is `1e-8` for every accepted tuning trial. Existing
training semantics remain unchanged: improvements below `min_delta` do not
replace the saved checkpoint. The smaller threshold avoids treating changes
near the observed `1e-6` score scale as negligible. It is still a saved-best
criterion, rather than the minimum of every infinitesimal score fluctuation.

Six configurations spanning widths 32/64, depths 2/3, learning rates
0.001/0.003 and batch sizes 16/32 were trained with initialization seed 17.
This is a sparse search, not a full factorial campaign. Three Pareto finalists
were confirmed with seed 29: best first-seed accuracy, fastest first-seed
contender within 5% of it, and the fastest first-seed Pareto alternative.
Among complete two-seed finalists, select the fastest mean complete invocation
whose worst-seed score lies within 5% of the best worst-seed score.

Each trial ran in a fresh one-thread CPU process. Complete invocation time
includes imports, manifest loading, reference construction, training, validation
and checkpoint writes. Shared manifest generation and orchestration are included
in the complete search time. Per-trial JSONL retains analytic teacher costs,
model/JVP work, backward/optimizer work, raw validation checkpoints and peak RSS.
Failures and incomplete replicates cannot win.

## Observed results

Environment: Python 3.12.14, PyTorch 2.10.0+cpu, NumPy 2.2.6, SciPy 1.15.3,
PyYAML 6.0.2, AMD EPYC 9V74 cloud host. Walltimes describe this shared CPU
environment; CARC uses the user's `python/3.12.8` module. They are not CARC
throughput measurements.

| Width / depth | LR | Batch | Seed | Saved-best score | Best / completed updates | Complete seconds | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 / 3, previous baseline | 0.001 | 32 | 17 | 1.19645e-6 | 450 / 500 | 27.09 | 8,835 |
| 32 / 2 | 0.001 | 16 | 17 | 8.87767e-7 | 500 / 500 | 17.71 | 1,315 |
| 32 / 2 | 0.003 | 16 | 17 | 9.23537e-7 | 175 / 325 | 13.08 | 1,315 |
| **32 / 2, selected** | **0.003** | **32** | **17** | **7.89477e-7** | **300 / 450** | **23.73** | **1,315** |
| 32 / 3 | 0.003 | 16 | 17 | 1.75684e-6 | 175 / 325 | 12.65 | 2,371 |
| 64 / 2 | 0.003 | 32 | 17 | 7.83649e-7 | 500 / 500 | 27.20 | 4,675 |
| 64 / 2 | 0.003 | 32 | 29 | 7.29554e-7 | 475 / 500 | 26.89 | 4,675 |
| 32 / 3 | 0.003 | 16 | 29 | 1.17572e-6 | 475 / 500 | 18.36 | 2,371 |
| **32 / 2, selected** | **0.003** | **32** | **29** | **6.98880e-7** | **325 / 475** | **24.54** | **1,315** |

Every listed trial completed successfully: either its maximum updates or its
predeclared validation patience stopped training. None hit its 40-second tuning
budget. The selected configuration's worst-seed score is `7.89477e-7`, within
0.75% of the 64/depth-2 alternative. Its mean invocation is 24.14 seconds versus
27.04 seconds for that alternative, with 72% fewer parameters. Its measured
maximum RSS is 371,920,896 bytes (approximately 355 MiB). Teacher construction
still dominates much of the complete cost; it is measured rather than hidden.

Against the previous baseline on the matched seed-17 trial, the selected
configuration has about 85% fewer parameters, 34% lower validation score and
12% shorter observed complete time. That baseline was not confirmed with a
second seed, so this particular comparison is explicitly seed-specific.
Batch 16 is a faster, slightly less accurate alternative; it is retained in the
table rather than presented as dominated at every possible accuracy tolerance.

The accepted nine-run experiment took 191.34 seconds, including the additional
near-best confirmation. An earlier 96.54-second draft search used the old
`1e-6` threshold and then failed while writing provenance because of an API
argument error. That draft is excluded from selection and remains preserved.
The error was fixed before the clean rerun. Combined experiment invocation
time was 287.89 seconds, below five minutes; no CARC jobs were submitted.

Compact raw artifacts remain ignored beneath
`runs/ssmo-config-tuning-002/`. The original eight-run `tuning_summary.json`
is preserved alongside `tuning_confirmation.json`, which includes the extra
near-best finalist and the final selection. The discarded draft is preserved
under `runs/ssmo-config-tuning-001/`. Neither checkpoints nor generated records
are committed. This document records the aggregate evidence needed to review
the preset.

All accepted workers recorded identical SHA-256 hashes for the scientific
sources despite concurrent shell/runbook work:

| Source | SHA-256 |
|---|---|
| `training.py` | `562fb56aea1de3e72fa3936b7882abe74c93a2b2ee48126dde8e79ab5814e20c` |
| `models.py` | `6b087967c81269e8a1375611e544774e5126dfc32d12b83edd132c3c9b4767fc` |
| `data.py` | `7f3f1ade2248cf235726e3674f83e0e221a2082a17516adea6281c3f3826ef85` |
| `reference.py` | `88f4dde8b01858f048c1f078a7bb41aa9456be9b3dd2cd4e48353bd191e0feba` |
| `queries.py` | `56412ae8baa94b4619658b2665024653931666b2acdfdf1063993aefc19d9de9` |

## Independent assessment after selection

After freezing the preset, one independent CPU evaluation assessed the selected
seed-17 checkpoint at update 300. It retained that trial's exact configuration,
including its original 64-parent evaluation cap, four LP parents, tolerance
`0.01`, resolution/FD ladders and sealed held-out query bank. The invocation
completed in 43.44 seconds within its external 90-second bound, with peak RSS
407,212,032 bytes (approximately 388 MiB). No settings were retuned from these
results. This assessment is separate from the 287.89-second selection search.

| Learned-chart cohort | Physical parents | Weak gate passed | Worst absolute weak-query error |
|---|---:|---:|---:|
| IID test, constant | 4 | 4 | 0.00072835 |
| IID test, shock | 27 | 27 | 0.00675466 |
| Range holdout, shock | 30 | 30 | 0.00915017 |
| Fixed analytic audit, shock | 1 | 0 | 0.01181979 |

Thus 61 of the 62 physical parents supported by the learned chart pass the
predeclared weak-query gate, with no invalid predicted charts. The fixed audit
shock fails; the complete gate has **not** passed. The largest observed learned
nonlinear-gradient error is 0.04996314, which is separate from the declared
linear weak-query gate. The classical polynomial front control passes all 62
supported parents with worst weak error `1.55e-15`, so this toy assessment does
not establish an advantage over the strong classical control.

All three fixed analytic family/event parents were included. Rarefaction and
collision calculations are reference audits, not learned transfer results.
Raw records preserve 12 near-event query rows and one unresolved exact collision
at time 0.6; no ordinary derivative is assigned there. The full assessment retains
22,039 query/status rows and eight bounded-Lipschitz diagnostic records under
`runs/ssmo-heldout-assessment-002/`. These generated artifacts remain outside Git.
Generated raw JSONL and captured stdout were losslessly compressed with gzip
level 1: 71.48 MB became 4.92 MB, and the complete assessment directory is
approximately 7.05 MB including retained configuration, provenance, summaries
and the original summary backup. Streaming SHA-256 and byte-count round trips
were verified before removing originals. `compression_receipt.json` maps each
original filename to its compressed counterpart and hashes; the active summary
points to the compressed artifacts. Compaction did not alter the selected
configuration, checkpoint or scientific calculations. The CARC report stage
now provides analogous verified compression for its own completed raw logs;
it preserves older source runs and records its mapping in `log-compaction.json`.
This single seed and finite query/parent subset do not establish complete range
coverage, learned event sensitivity, CARC performance or research efficacy.

## Optional local reproduction

The selected preset is already committed; do not repeat a search merely to
install it. To reproduce on a cloud/local CPU checkout with its project venv:

```bash
bash scripts/tune_local.sh ssmo-config-tuning-new
```

This refuses CARC, confines caches, temporary data, logs and checkpoints to the
project, requires a fresh run ID, uses eager FP32 and one thread, and limits the
search to 300 seconds. It saves all attempted trial statuses and confirms up to
three finalists. A slower host may exhaust the budget; unconfirmed candidates
are excluded and a failed search preserves its evidence. Timing variations can
change Pareto shortlists or a near-tie selection. Do not automatically overwrite
the committed CARC preset from these local results.

CARC expansion remains gated on actual GPU runtime/memory observations and
held-out raw query errors, with three separately analyzed initialization seeds
before confirmatory research claims. No compiler, reduced precision, larger
architecture or broader PDE task is enabled by this tuning exercise.
