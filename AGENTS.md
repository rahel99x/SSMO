# Working on SSMO

Use the existing checkout. Each cloud task is already isolated; do not create a
Git worktree unless the user explicitly requests it. Read README.md and
docs/SCIENCE_SCOPE.md for the implemented scope; the supplied proposal describes
gated future work as well as current requirements.

Use standalone Python 3.11/3.12 and the project-local `.venv`, never Conda.
`bash scripts/setup_local.sh` prepares the cloud/CPU environment. Source
`scripts/carc_env.sh` only after setting `SSMO_PROJECT_ROOT` to the actual project root;
call `ssmo_env` before Python. Invoke `.venv/bin/python` explicitly. All artifact,
temporary and cache paths must stay inside the project; preserve failed results.

CPU checks in this cloud workspace use `python -m unittest discover -s tests -v`
or the CLI audit, with the venv interpreter and confined caches. Full tests run
in seconds; tests with mocked Slurm do not establish CARC execution. Actual
CARC tests, installation, generation, training and benchmarks require Slurm
allocations and `srun`, explicit `--account=anakano_81`, and root
`/home1/aadaniel/projects/SSMO`. Read docs/CARC_RUNBOOK.md before deployment.
Preparing scripts does not authorize live submissions. `submit.sh` is dry-run
by default; preserve Slurm's `CUDA_VISIBLE_DEVICES` and use `cuda:0`.
Use `scripts/carc.sh` for normal CARC discovery/setup/smoke/pilot/status/recovery.
It reads project-local site settings, captures fresh policy for live actions,
and reuses only a verified Python 3.12.8 CUDA environment. Never guess free
shared-account capacities. Read docs/CONFIG_TUNING.md before modifying presets;
selection uses independent validation, with no retuning on sealed test results.
Do not reintroduce project pending-job caps or manual submission-slot settings.
Other projects may enqueue concurrently. Preserve the 30-minute maximum GPU
allocation and three reviewed free CPU/memory/GPU resource values; account queue
freshness excludes PENDING rows. Keep queued-SSMO protection against venv changes.
Open `local/venv.lock` read/write (`9<>`) for both shared task locks and exclusive
installation locks; shared locks on network filesystems can reject write-only
descriptors. Keep the lockfile inode and distinguish contention from lock errors.
Use `scripts/summarize_pilots.sh` to review completed pilot replications. It reads
existing artifacts with the venv and standard library; never pool repeated
physical parents across initialization seeds or choose the best seed afterward.
The user requests next runbook steps in every response: include runnable CARC
commands, the expected result and the next evidence to inspect. Use completed
artifacts when available rather than asking for a repeat of finished jobs.
Use `scripts/diagnose_pilot_failures.sh --review-dir runs/<review-id>` for CPU
inspection of frozen failed-parent query records. It validates receipts and
reconstructs signed linear/nonlinear errors without loading model checkpoints.
Its fixed-order terms may cancel; they are exploratory bookkeeping, not causal
proof or permission to tune on the exposed audit/test cases.
The user-reported failure diagnostic and endpoint/inverse cost review are
complete; read docs/CARC_REPLICATION_ASSESSMENT.md. The measure chart fails the
full gate and is slower than exact/classical controls on the measured workloads.
Close/narrow this single-front learned pilot to correctness/representation
evidence; use `scripts/close_pilot.sh` for small copies/checksums, retaining
all original runs. No additional seeds, tuning or longer GPU runs are justified.
Any new learned study needs a declared downstream accuracy/cost target,
independent validation and a fresh final holdout. Pair costs on the same physical
workload, separate first calls from warm medians, and never sum overlapping
components. Baseline fit time
includes label generation/all controls. Inverse timing includes exact diagnostic
gradients and trusted acceptance, so it is not autonomous learned speedup.

Read docs/TOWER_FORMATS.md before changing reporting. The user's Tower 2.3.1
application must remain unchanged; adapt SSMO only. Each actual Slurm stage gets
one fresh writable Tower WorkDir with exact job identity, grouped log index,
finite bounded numeric metrics, terminal summary and compact scientific results.
Keep frozen source read-only and preserve original task exit/signal/checkpoint
behavior. Existing scientific summaries/raw logs remain authoritative. Training
reports reuse existing observations at validation/checkpoint boundaries or about
every two seconds; GPU traces sample only one verified allocated physical device
on one node every 60 seconds, with an explicit skip reason otherwise.
Use scripts/tower.sh to import explicit completed runs/reviews into fresh
contained exports and build explicitly selected planning history. It uses the
venv and standard library; no Tower installation, checkpoint load, scientific
execution, raw decompression, automatic job polling or submissions. Optional
--accounting makes one bounded sacct query. Missing resources remain unknown;
MaxRSS is max_task_rss, Torch reservation is separate, seed/parent/method/workload
cohorts stay separate, and completed processes may fail scientific gates.
Never invent historical metric epochs/progress, forecasts, scaling evidence or
Tower passports. Validate the bundled schemas and actual emitted samples;
native Tower validation requires the user's existing installation. Imports are
the next runbook step for this completed pilot, without another GPU campaign.
Pre-integration frozen runs remain historical evidence; source recovery guards
must continue refusing changed implementations rather than rewriting snapshots.

Retain signed atoms at continuous coordinates; never multiply an atom by cell
width or normalize signed masses as probabilities. Nonlinear field objectives
use payoff jumps and base traces. Chart JVPs must retain mixed weight/parameter
gradients, direction linearity and consistency with the associated state map.
Treat unresolved/grazing events explicitly; no invented two-sided derivative.

Keep all variants of a physical parent in one split. Select checkpoints and
protocols using independent validation queries; never tune on the sealed test
pool. Record raw errors, uncertainty, failures, complete costs and actual
hardware. A short passing smoke is implementation evidence, not a research
advantage or GPU-performance claim. Follow the proposal's gates before adding
learned events, wider PDE classes, BF16/compile, KANs or large campaigns.
