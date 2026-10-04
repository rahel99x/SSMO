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
