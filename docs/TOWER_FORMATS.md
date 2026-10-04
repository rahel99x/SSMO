# SSMO reporting for Tower 2.3.1

SSMO produces Tower's version-1 interchange files alongside its existing
scientific artifacts. The integration changes SSMO's reporting and leaves the
Tower application untouched. Python 3.12.8, the project venv, CARC storage
containment, frozen source snapshots and scientific protocols retain their
existing meaning. See [the CARC runbook](CARC_RUNBOOK.md) for commands.

The six schemas in [`.tower/schemas`](../.tower/schemas) are exact copies from
the user-supplied Slurm-Tower 2.3.1 bundle at source commit
`b7e28338683526453ec16ba3fb864884b5944e46`. Their original relative references
are preserved. [UPSTREAM.json](../.tower/UPSTREAM.json) records their checksums,
source paths and version; [LICENSE.upstream](../.tower/LICENSE.upstream) retains
the MIT notice. Configuration, output contracts and reporting code belong to
SSMO. Runtime reporting requires only Python's standard library, not Tower or
a JSON Schema package.

## Attempt identity and paths

A Slurm pipeline contains distinct scheduling attempts: validation, data,
GPU audit, training, evaluation and reporting. Each stage has its own Tower
inventory, metric stream, terminal summary and log index. A pipeline's
shared physical parents do not become additional independent samples when
several stages or initialization seeds inspect them.

| File | Meaning and consumer |
| --- | --- |
| `run.json` | `tower.run/v1` project inventory, lifecycle and declared paths |
| `logs.json` | `tower.logs/v1` exact grouped log locations for that attempt |
| `metrics.jsonl` | Native numeric observations and optional progress |
| `summary.json` | `tower.summary/v1` flat attempt/request/usage/result record |
| `outputs/analytics.json` | Compact SSMO scientific results; large originals remain in existing artifacts |
| `logs/gpu-util-<job-id>.csv` | Optional measured GPU trace in the scheduler's actual WorkDir |
| `reports/planning.json` | Explicitly selected summary records for native planning analysis |

`run_id` and summary `id` identify one attempt and remain stable on later
exports. `job_id` is the actual scheduler identity; a batch or extern step is
not another independent experiment. A retry or resume uses a fresh attempt,
retaining old logs and checkpoints. `experiment_id` and workload `name`
describe logical work. Do not put timestamps, scheduler IDs or run-directory
names in the stable comparison name.

The coordinator creates writable attempt directories before submission.
Their scheduler WorkDir must be writable because Tower looks for GPU traces
at `<WorkDir>/logs/gpu-util-<job-id>.csv`. The frozen source directory stays
read-only. Original source/config/data provenance still identifies the bytes
used for the scientific calculation.

Tower does not discover an active run from `run.json` or follow its paths.
[`.tower/config.json`](../.tower/config.json) explicitly binds `logs.json` and
`metrics.jsonl`. Select the concrete attempt with an absolute `--workdir`.
The configuration's contract and planning filenames resolve from the shell's
current directory, so launch from the project root. Inventory paths resolve
from their attempt directory. Log-index entry paths resolve from the directory
containing `logs.json`; explicit relative sibling paths are allowed there.
Contract outputs remain strictly inside the selected attempt, with no `..`,
absolute path, glob or symlink traversal.

## Scientific and operational measurements

Process success and scientific adequacy are separate fields. A completed
evaluation can contain failed weak-error gates. Preserve its actual process
state and put pass/fail counts, tolerances, worst errors and scientific scope
under `results`. Do not interpret an artifact-contract pass as an accuracy or
research-advantage certificate.

Training observations use existing measured losses, validation values and
timings. Evaluation observations summarize parent-level outcomes, unresolved
events, direction checks and complete endpoint costs without turning each
parent/query into a separate named metric series. Reporting exports retain
representation, numerical, inverse, diagnostic and pilot-review findings in
compact results and exact indexed source files. Descriptive status strings
and provenance are metadata, not numeric metric observations.

Keep the meaning of each cost:

- Complete endpoint costs include the declared transfer, forward chart,
  directional assembly and weak-query workload. CPU exact/classical controls
  and GPU learned endpoints retain their separate devices.
- A recorded first endpoint call is distinct from a warm median and from
  process/model startup. Pair ratios on the same physical workload; values
  above one mean the learned endpoint took longer.
- Component timings overlap and must not be summed. Teacher costs already
  occur within training time; baseline fit cost includes label generation and
  all fitted controls.
- Inverse timings include exact diagnostic gradients and trusted acceptance
  checks. They are not autonomous learned-optimizer speedups.

Summary resources and measured usage have different fields:

| Field | Measurement definition |
| --- | --- |
| `cpus`, `nodes`, `gpus` | Known allocation totals, not worker/rank counts or host capacity |
| `mem_bytes` | Total requested memory with Slurm per-node/per-CPU semantics resolved |
| `time_seconds` | Requested walltime limit |
| `runtime_seconds` | Actual execution duration, excluding queue wait |
| `cpu_seconds` | Measured CPU time with source and process/allocation scope declared |
| `memory_bytes`, `memory_scope` | An observed memory value and its actual scope |
| `script_sha256` | Digest of the actual declared entry-point bytes |
| `parameters` | Small stable scientific/code/input/software/hardware identity |
| `results`, `metadata` | Scientific findings, measurement provenance and limitations |

Ordinary Slurm `MaxRSS` has scope `max_task_rss`; it is not an allocation-wide
simultaneous peak. Python process RSS and Torch allocated/reserved GPU memory
retain their source and scope. Do not derive host memory from Torch GPU values,
sum asynchronous peaks or interpret reserved Torch bytes as all GPU usage.
Unknown resource counts, queue events and measurements stay absent or `null`
in summaries. In a metric row they are omitted: zero is a real observation.

The optional GPU trace records the allocated device's actual `nvidia-smi`
timestamp, index, utilization percentage and memory MiB at a 60-second cadence.
Keep Slurm's device visibility unchanged and avoid sampling other node GPUs.
A short job may produce very few samples; these do not establish sustained
utilization. CPU-only attempts have no GPU trace. Tower's CARC profile disables
additional GPU sampling, weather and budget polling; its slower scheduler
intervals are declared in the project configuration.

## Live observations, imported evidence and failures

Live metric rows use finite nonnegative epoch timestamps; durations use a
monotonic clock. Steps and timestamps increase within a phase. Progress has
`0 <= completed <= total`, positive `total` and an explicit unit when useful.
Reset progress on a new phase. A pause or early stop does not complete the
original step budget merely because the process exits.

Legacy training logs contain elapsed durations, not verified per-step epoch
timestamps. Preserve these originals. Imported observations must identify
their historical source and actual export time; do not fabricate original
live timestamps, pre-start predictions or historical progress ETAs. Missing
accounting facts remain unknown. Fresh reports supplement frozen old runs
without rewriting their source snapshots, scientific JSON or raw logs.
Legacy imports leave their new application capture files empty; actual original
terminal output remains in the indexed scheduler files. The Log view must
select the matching scheduler job when the index declares a `job_id`.

Preserve `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED` and other known
outcomes. A signal alone does not establish its scheduler cause. An interrupted
process can leave a `RUNNING` inventory and no final summary; reconciliation
requires actual scheduler evidence. Preserve paused checkpoints and their
application exit codes. Reconciliation and repeated exports do not create
additional experimental repeats.

Lossless gzip archives and their checksums remain authoritative raw evidence.
Tower's text browser does not decode gzip. Index their compaction receipts and
small readable analytics files, retaining archive locations in provenance;
avoid expanding large logs merely for display. Genuine Tower passports are
captured through Tower's own API/command. SSMO checksum records are not
passports, and existing passports remain immutable.

## Bounded interchange and analysis

JSON files use UTF-8, unique keys and finite values. Publish inventories,
log indexes, summaries and aggregate reports atomically using a temporary
file in the same directory. One coordinator appends complete flushed metric
lines; independent attempts never share a stream.

| Data | Producer/native limit |
| --- | --- |
| Run inventory | 64 KiB in SSMO's contract |
| Log index | 256 KiB; at most 256 exact entries with unique IDs |
| Summary and compact analytics | 256 KiB each in SSMO's contract |
| Metric line | At most 65,536 bytes; at most 64 stable metric names |
| Metric stream | 4 MiB in SSMO's contract; native reader retains bounded recent samples |
| Planning bundle | 1 MiB; at most 10,000 history rows, depth 32 and 100,000 JSON values |
| Existing scientific input JSON | 32 MiB and two million values, depth 32; validated once and compacted before Tower publication |
| Scientific cohort parameters | Depth four, at most 128 total JSON values and 64 members per object/list |
| Output contract | 256 KiB; exact unique paths and ordered size/row bounds |

The common [output contract](../.tower/contracts/outputs.v1.json) requires
inventory, log index and a nonempty metric stream. Terminal summary and
compact scientific results are optional because an active or failed attempt
may not have them. Read state and available results to judge completion;
file presence checks alone cannot establish it.

Planning exports explicitly select attempts, preserve failures and keep
method/config/input/hardware cohorts distinct. Tower withholds predictions
without sufficient compatible measured history; three initialization seeds
do not supply a calibrated resource interval. There is no SSMO scaling recipe
because these pilots do not vary controlled worker counts. Exported
dependency details describe actual `jobs.tsv` edges; they do not constitute
a measured-duration workflow recipe. There is no predefined SSMO workflow
recipe. If a future declared study needs one, keep unknown duration unknown
rather than replacing it with requested walltime. Tower's workflow analysis
is conditional on immediate capacity and successful predecessors, not a
scheduler guarantee or submission engine.

The copied Draft 2020-12 schemas provide portable structural validation.
Tower's native readers additionally enforce identity, finite-number,
duplicate-key, path, progress, comparability and I/O constraints. A schema
pass cannot establish file availability or scientific validity. Check native
readers using the user's existing Tower installation; do not install or
modify Tower as part of SSMO reporting.

`tower.sh validate/launch` require an executable on PATH. Shell aliases and
functions are not inherited by the Python launcher. When `tower` is an alias,
invoke it directly in the interactive shell with absolute SSMO contract,
configuration, planning and attempt paths as shown in the CARC runbook.
Experiment graphs numeric metrics; use indexed JSON artifacts/logs for nested
scientific results. An incomplete import has no `index.json`; retain it and
retry in a fresh export before planning or selecting an attempt.
