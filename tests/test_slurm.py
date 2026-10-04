"""Pure mocked CARC integration tests: never invoke a live scheduler."""
from __future__ import annotations

import csv
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from singular_sensitivity.runtime import confined_path


REPO = Path(__file__).resolve().parents[1]
CARC_ROOT = "/home1/aadaniel/projects/SSMO"


class SlurmWorkflowTests(unittest.TestCase):
    def setUp(self):
        # On CARC even test temporaries remain in the project, never /tmp.
        base = confined_path("local/test-tmp")
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ssmo-slurm-", dir=base)
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        for dirname in ("scripts", "slurm"):
            shutil.copytree(REPO / dirname, self.root / dirname)
            (self.root / dirname).chmod(0o700)
            # Rewrite the fixed production root only in this disposable fixture.
            # Production --submit never offers a storage/identity bypass flag.
            for path in (self.root / dirname).glob("*"):
                if path.is_file():
                    path.chmod(0o700)
                    path.write_text(path.read_text().replace(CARC_ROOT, str(self.root)))
        (self.root / "configs").mkdir()
        (self.root / "configs" / "carc_smoke.yaml").write_text("name: mock\n")
        (self.root / "README.md").write_text("immutable source\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.bin = Path(self.temp.name) / "bin"
        self.bin.mkdir()
        self.calls = Path(self.temp.name) / "sbatch-calls.txt"
        self.counter = Path(self.temp.name) / "counter"
        self.counter.write_text("100\n")
        self.tower_calls = Path(self.temp.name) / "tower-calls.jsonl"
        self.stub("id", "printf 'aadaniel\\n'\n")
        self.stub("squeue", 'if [[ " $* " = *" -u "* ]]; then printf "%s" "${MOCK_USER_QUEUE:-${MOCK_QUEUE:-}}"; else printf "%s" "${MOCK_ACCOUNT_QUEUE:-${MOCK_QUEUE:-}}"; fi\n')
        self.stub("sbatch", """n=$(cat "$MOCK_COUNTER")
n=$((n+1))
printf '%s\\n' "$n" > "$MOCK_COUNTER"
printf '%s\\t' "$@" >> "$MOCK_CALLS"
printf '\\n' >> "$MOCK_CALLS"
if [[ ${MOCK_FAIL_STAGE:-} != '' && " $* " = *" ${MOCK_FAIL_STAGE} "* ]]; then exit 17; fi
printf '%s;mockcluster\\n' "$n"
""")
        self.env = dict(os.environ, SSMO_PROJECT_ROOT=str(self.root), MOCK_CALLS=str(self.calls),
                        MOCK_COUNTER=str(self.counter), MOCK_TOWER_CALLS=str(self.tower_calls),
                        PATH=f"{self.bin}:{os.environ['PATH']}")
        self.env.pop("SSMO_ROOT", None)
        self.discovery = self.root / "runs" / "observed" / "discovery"
        self.discovery.mkdir(parents=True)
        values = {
            "status.tsv": "command\texit_code\nidentity\t0\n",
            "manifest.txt": "observed test fixture\n",
            "identity.txt": "aadaniel\n",
            "associations.txt": "anakano_81|aadaniel||normal|||\n",
            "modules.txt": "python/3.12.8 python/3.12.0\n",
            "partitions.txt": "main*|up|(null)|1-00:00:00|64|256000\ngpu|up|gpu:a100:4|1-00:00:00|64|256000\n",
            "gpu_nodes.txt": "n1|gpu:a100:4|a100-40gb|256000|64\nn2|gpu:a40:4|(null)|256000|64\nn3|gpu:a30:4|(null)|256000|64\nn4|gpu:l40:4|(null)|256000|64\nn5|gpu:l40s:4|(null)|256000|64\n",
            "main.txt": "PartitionName=main State=UP\n",
            "gpu.txt": "PartitionName=gpu State=UP\n",
            "account_jobs.txt": "",
        }
        for name, value in values.items():
            (self.discovery / name).write_text(value)

    def tearDown(self):
        # Snapshots deliberately become readonly; restore only fixture modes.
        for path in Path(self.temp.name).rglob("*"):
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\nset -euo pipefail\n" + body)
        path.chmod(0o700)

    def batch_fixture(self, stage="validate", profile="cpu", run_id="job"):
        """Mock only producer/srun boundaries; never execute research work."""
        self.stub("module", "exit 0\n")
        source = self.root / "runs" / run_id / "source"
        source.mkdir(parents=True)
        shutil.copytree(self.root / "scripts", source / "scripts")
        package = source / "singular_sensitivity"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "tower_reporting.py").write_text(
            "import json, os, pathlib, sys\n"
            "with open(os.environ['MOCK_TOWER_CALLS'], 'a') as out:\n"
            "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1] == 'start':\n"
            "    path = pathlib.Path(sys.argv[sys.argv.index('--attempt-dir') + 1])\n"
            "    (path / 'logs').mkdir()\n"
            "    for name in ('application.stdout.log', 'application.stderr.log'):\n"
            "        (path / 'logs' / name).touch()\n"
            "    (path / 'mock-gpu-trace.json').write_text(json.dumps({key: os.environ.get(key) for key in\n"
            "        ('SSMO_TOWER_GPU_TRACE_STATUS', 'SSMO_TOWER_GPU_TRACE_REASON', 'SSMO_TOWER_GPU_TRACE_SELECTOR')}))\n"
            "    sys.exit(int(os.environ.get('MOCK_TOWER_START_EXIT', '0')))\n"
            "sys.exit(int(os.environ.get('MOCK_TOWER_FINISH_EXIT', '0')))\n"
        )
        run = self.root / "runs" / run_id
        (run / "logs").mkdir()
        attempt = run / "tower" / f"01-{stage}-measure-seed7"
        attempt.mkdir(parents=True)
        args = [stage, str(run), str(self.root / "configs" / "carc_smoke.yaml"),
                str(source), profile, "measure", "7", "none", "none"]
        env = dict(self.env, SLURM_JOB_ID="22", SLURM_JOB_ACCOUNT="anakano_81",
                   SSMO_TOWER_RUN_DIR=str(attempt), SSMO_TOWER_CPUS="2",
                   SSMO_TOWER_MEM_GB="4", SSMO_TOWER_WALLTIME="00:10:00",
                   SSMO_TOWER_GPU_COUNT="0", SSMO_TOWER_GPU_TYPE="",
                   SSMO_TOWER_SCHEDULER_OUT=str(run / "logs" / f"{stage}-measure-%j.out"),
                   SSMO_TOWER_SCHEDULER_ERR=str(run / "logs" / f"{stage}-measure-%j.err"))
        return args, env, attempt

    def run_wrapper(self, *args, live=False):
        flags = ["--run-id", "fixture"]
        if live:
            flags += ["--submit", "--discovery", "observed",
                      "--account-free-cpus", "4", "--account-free-mem-gb", "8",
                      "--account-free-gpus", "1"]
        return subprocess.run(["bash", str(self.root / "scripts" / "submit.sh"), *flags, *args],
                              env=self.env, text=True, capture_output=True)

    def verified_environment(self):
        shutil.copytree(REPO / "requirements", self.root / "requirements")
        venv = self.root / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("version = 3.12.8\n")
        interpreter = venv / "bin" / "python"
        interpreter.write_text("#!/bin/bash\nexit 0\n")
        interpreter.chmod(0o700)
        (venv / "ssmo-dependency-freeze.txt").write_text("torch==2.10.0+cu126\n")
        helper = shlex.quote(str(self.root / "scripts" / "carc_env.sh"))
        result = subprocess.run(["bash", "-c", f"source {helper}; ssmo_init_root && ssmo_environment_signature python/3.12.8"],
                                env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (venv / "ssmo-environment-signature.txt").write_text(result.stdout)
        return venv

    def test_setup_allocates_only_install_validation_and_gpu_audit(self):
        result = self.run_wrapper("--pipeline", "setup", live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "runs/fixture/jobs.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual([row["stage"] for row in rows], ["install", "validate", "gpu-smoke"])
        self.assertEqual(rows[-1]["dependency"], rows[-2]["job_id"])

    def test_environment_reuse_omits_install_and_rejects_dependency_drift(self):
        self.verified_environment()
        result = self.run_wrapper("--skip-install", live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "runs/fixture/jobs.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["stage"], "validate")
        self.assertNotIn("install", [row["stage"] for row in rows])
        # Allocated audits copy read-only sources. Change only the disposable
        # fixture's file mode before deliberately simulating requirement drift.
        (self.root / "requirements/base.txt").chmod(0o600)
        (self.root / "requirements/base.txt").write_text("numpy==0.0.0\n")
        before = self.calls.read_text()
        result = self.run_wrapper("--run-id", "drift", "--skip-install", live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("environment unavailable or changed", result.stderr)
        self.assertEqual(self.calls.read_text(), before)

    def test_reuse_refuses_missing_marker_wrong_python_or_cpu_wheel(self):
        result = self.run_wrapper("--skip-install")
        self.assertNotEqual(result.returncode, 0)
        venv = self.verified_environment()
        for name, content in [("pyvenv.cfg", "version = 3.11.9\n"),
                              ("ssmo-dependency-freeze.txt", "torch==2.10.0+cpu\n")]:
            with self.subTest(name=name):
                original = (venv / name).read_text()
                (venv / name).write_text(content)
                result = self.run_wrapper("--skip-install", live=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.calls.exists())
                (venv / name).write_text(original)

    def test_missing_requirements_propagate_signature_failure(self):
        helper = shlex.quote(str(self.root / "scripts" / "carc_env.sh"))
        result = subprocess.run(["bash", "-c", f"source {helper}; ssmo_init_root && ssmo_environment_signature"],
                                env=self.env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)


    def test_dry_run_has_exact_account_resources_and_does_not_mutate(self):
        result = self.run_wrapper("--after", "77")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("sbatch ")]
        self.assertEqual(len(commands), 7)
        self.assertTrue(all("--account=anakano_81" in cmd for cmd in commands))
        self.assertTrue(all("--nodes=1" in cmd for cmd in commands))
        self.assertIn("--dependency=afterok:77", commands[0])
        self.assertIn("--partition=main", commands[0])
        self.assertIn("--partition=gpu", commands[3])
        self.assertIn("--gpus-per-task=a100:1", commands[3])
        self.assertIn("--constraint=a100-40gb", commands[3])
        for index, command in enumerate(commands, start=1):
            workdir = next(arg.removeprefix("--chdir=") for arg in command if arg.startswith("--chdir="))
            self.assertTrue(workdir.startswith(f"{self.root}/runs/fixture/tower/{index:02d}-"))
            exported = next(arg for arg in command if arg.startswith("--export="))
            self.assertIn(f"SSMO_TOWER_RUN_DIR={workdir}", exported)
            self.assertIn("SSMO_TOWER_GPU_COUNT=" + ("1" if "--partition=gpu" in command else "0"), exported)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / "runs" / "fixture").exists())
        self.assertFalse((self.root / "local").exists())

    def test_live_mock_ids_dependencies_frozen_source_and_logs(self):
        result = self.run_wrapper(live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        run = self.root / "runs" / "fixture"
        with (run / "jobs.tsv").open() as file:
            jobs = list(csv.DictReader(file, delimiter="\t"))
        self.assertEqual([row["job_id"] for row in jobs], [str(n) for n in range(101, 108)])
        self.assertEqual([row["dependency"] for row in jobs], ["none", "101", "102", "103", "104", "105", "106"])
        self.assertTrue((run / "logs").is_dir())
        self.assertTrue((run / "source-sha256.txt").is_file())
        self.assertFalse((run / "source" / "runs").exists())
        self.assertEqual((run / "source" / "README.md").read_text(), "immutable source\n")
        self.assertFalse((run / "source" / "README.md").stat().st_mode & 0o222)
        attempts = sorted((run / "tower").iterdir())
        self.assertEqual(len(attempts), 7)
        self.assertTrue(all(path.is_dir() and not any(path.iterdir()) for path in attempts))
        self.assertFalse((run / "source" / "tower").exists())
        (self.root / "README.md").write_text("later edit\n")
        self.assertEqual((run / "source" / "README.md").read_text(), "immutable source\n")
        duplicate = self.run_wrapper(live=True)
        self.assertNotEqual(duplicate.returncode, 0)

    def test_submission_failure_preserves_real_prior_ids(self):
        self.env["MOCK_FAIL_STAGE"] = "validate"
        result = self.run_wrapper(live=True)
        self.assertEqual(result.returncode, 17, result.stderr)
        lines = (self.root / "runs" / "fixture" / "jobs.tsv").read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("101", lines[1])
        self.assertNotIn("scancel", result.stdout + result.stderr)

    def test_paths_and_existing_cache_symlinks_cannot_escape(self):
        result = self.run_wrapper("--config", "/etc/passwd")
        self.assertNotEqual(result.returncode, 0)
        (self.root / "local").mkdir()
        (self.root / "local" / "cache").symlink_to(self.bin, target_is_directory=True)
        script = f"source {shlex.quote(str(self.root / 'scripts' / 'carc_env.sh'))}; ssmo_env"
        result = subprocess.run(["bash", "-c", script], env=self.env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("escapes", result.stderr)

    def test_sourced_helper_preserves_options_and_returns(self):
        helper = shlex.quote(str(self.root / "scripts" / "carc_env.sh"))
        script = f'pre=$(set +o); source {helper}; post=$(set +o); [[ "$pre" = "$post" ]] || exit 5; SSMO_PROJECT_ROOT=/nonexistent; ssmo_env; printf "shell-alive\\n"'
        result = subprocess.run(["bash", "-c", script], env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("shell-alive", result.stdout)

    def test_root_namespace_and_legacy_alias_are_consistent(self):
        helper = shlex.quote(str(self.root / "scripts" / "carc_env.sh"))
        script = f'source {helper}; ssmo_init_root || exit; printf "%s\\n" "$SSMO_PROJECT_ROOT"'
        for use_legacy in (False, True):
            with self.subTest(use_legacy=use_legacy):
                env = dict(self.env, PROJECT_ROOT=str(self.bin))
                if use_legacy:
                    env["SSMO_ROOT"] = env.pop("SSMO_PROJECT_ROOT")
                result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), str(self.root))

    def test_conflicting_roots_stop_before_cache_creation(self):
        helper = shlex.quote(str(self.root / "scripts" / "carc_env.sh"))
        env = dict(self.env, SSMO_ROOT=str(self.bin))
        result = subprocess.run(["bash", "-c", f"source {helper}; ssmo_env"],
                                env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disagree", result.stderr)
        self.assertFalse((self.root / "local").exists())
        self.assertFalse((self.bin / "local").exists())

    def test_carc_root_guard_rejects_other_directory_before_cache_creation(self):
        helper = shlex.quote(str(self.root / "scripts" / "carc_env.sh"))
        for rejected in (self.bin, Path("/home1/aadaniel/projects/SSNO")):
            with self.subTest(root=rejected):
                env = dict(self.env, SSMO_PROJECT_ROOT=str(rejected))
                result = subprocess.run(["bash", "-c", f"source {helper}; ssmo_env"],
                                        env=env, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CARC storage requires", result.stderr)
        self.assertFalse((self.bin / "local").exists())

    def test_new_running_allocations_and_insufficient_free_resources_block_before_submit(self):
        self.env["MOCK_ACCOUNT_QUEUE"] = "9|other|other-project|RUNNING|4|8G|gpu:a100:1"
        result = self.run_wrapper(live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("allocations changed", result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / "runs/fixture").exists())
        self.env.pop("MOCK_ACCOUNT_QUEUE")
        result = self.run_wrapper("--account-free-cpus", "1", live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("insufficient", result.stderr)

    def test_many_pending_other_projects_and_new_ssmo_pending_jobs_allow_verified_reuse(self):
        self.verified_environment()
        account_rows = []
        user_rows = []
        for project in range(1, 6):
            for job in range(20):
                job_id = 1000 + project * 20 + job
                name = f"OTHER-{project}-small-{job}"
                account_rows.append(f"{job_id}|aadaniel|{name}|PENDING|1|1G|N/A")
                user_rows.append(f"{job_id}|{name}|PENDING|anakano_81|1|1G|N/A")
        account_rows.append("2001|aadaniel|SSMO-previous-train-measure|PENDING|2|8G|gpu:a100:1")
        user_rows.append("2001|SSMO-previous-train-measure|PENDING|anakano_81|2|8G|gpu:a100:1")
        self.env["MOCK_ACCOUNT_QUEUE"] = "\n".join(account_rows)
        self.env["MOCK_USER_QUEUE"] = "\n".join(user_rows)

        result = self.run_wrapper("--skip-install", live=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "runs/fixture/jobs.tsv").open() as stream:
            jobs = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(jobs), 6)
        self.assertEqual(jobs[0]["stage"], "validate")
        self.assertNotIn("install", [row["stage"] for row in jobs])
        self.assertNotIn("scancel", result.stdout + result.stderr + self.calls.read_text())

    def test_allocation_queue_reordering_and_pending_changes_do_not_invalidate_discovery(self):
        self.verified_environment()
        first = "51|other|OTHER-running-cpu|RUNNING|1|1G|N/A"
        second = "52|aadaniel|OTHER-running-gpu|RUNNING|1|2G|gpu:a30:1"
        (self.discovery / "account_jobs.txt").write_text(f"{first}\n{second}\n53|other|old-pending|PENDING|1|1G|N/A\n")
        self.env["MOCK_ACCOUNT_QUEUE"] = f"99|other|new-pending|PENDING|1|1G|N/A\n{second}\n{first}\n"

        result = self.run_wrapper("--skip-install", live=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.calls.exists())

    def test_unknown_gpu_tag_is_rejected_and_l40s_is_distinct(self):
        result = self.run_wrapper("--gpu-profile", "l40s", live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--gpus-per-task=l40s:1", self.calls.read_text())
        (self.discovery / "gpu_nodes.txt").write_text("n1|gpu:l40:4|(null)|256000|64\n")
        result = self.run_wrapper("--run-id", "second", "--gpu-profile", "l40s", live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not observed", result.stderr)

    def test_failed_and_stale_discovery_block_submission(self):
        (self.discovery / "status.tsv").write_text("command\texit_code\nquota\t1\n")
        result = self.run_wrapper(live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed commands", result.stderr)
        (self.discovery / "status.tsv").write_text("command\texit_code\nidentity\t0\n")
        old = time.time() - 7200
        os.utime(self.discovery / "manifest.txt", (old, old))
        result = self.run_wrapper(live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("within one hour", result.stderr)

    def test_pilot_preview_allows_nine_jobs_without_pending_cap_flags(self):
        result = self.run_wrapper("--pipeline", "pilot")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("sbatch ")]
        self.assertEqual(len(commands), 9)
        self.assertTrue(all("--account=anakano_81" in command for command in commands))
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / "runs/fixture").exists())
        self.assertFalse((self.root / "local").exists())

    def test_every_gpu_profile_has_one_gpu_and_at_most_thirty_minutes_per_stage(self):
        for profile in ("a10040", "a40", "a30", "l40", "l40s"):
            with self.subTest(profile=profile):
                result = self.run_wrapper("--pipeline", "pilot", "--gpu-profile", profile)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("sbatch ")]
                gpu_commands = [command for command in commands if "--partition=gpu" in command]
                self.assertEqual(len(gpu_commands), 5)
                for command in gpu_commands:
                    wall = next(arg.removeprefix("--time=") for arg in command if arg.startswith("--time="))
                    hours, minutes, seconds = map(int, wall.split(":"))
                    self.assertLessEqual(hours * 3600 + minutes * 60 + seconds, 1800)
                    requests = [arg for arg in command if arg.startswith("--gpus-per-task=")]
                    self.assertEqual(len(requests), 1)
                    expected = "a100" if profile == "a10040" else profile
                    self.assertEqual(requests[0], f"--gpus-per-task={expected}:1")
                    self.assertIn("--ntasks=1", command)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / "runs/fixture").exists())

    def test_resume_and_future_downstream_use_fresh_id_and_afterok(self):
        previous = self.root / "runs" / "previous" / "artifacts"
        (previous / "data").mkdir(parents=True)
        (previous / "train-measure-seed17").mkdir()
        manifest = previous / "data" / "parents.json"
        manifest.write_text("{}\n")
        checkpoint = previous / "train-measure-seed17" / "last.pt"
        checkpoint.write_bytes(b"mock checkpoint; never loaded")
        result = self.run_wrapper("--stage", "train", "--manifest", str(manifest),
                                  "--resume", str(checkpoint), "--after", "90", live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text()
        self.assertIn("--dependency=afterok:90", calls)
        self.assertIn(str(checkpoint), calls)
        future = self.root / "runs" / "fixture" / "artifacts" / "train-measure-seed17" / "best.pt"
        result = self.run_wrapper("--run-id", "evaluation", "--stage", "evaluate", "--manifest", str(manifest),
                                  "--checkpoint", str(future), "--after", "101", live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dependency=afterok:101", self.calls.read_text())

    def test_partition_associations_features_and_writable_logs_are_checked(self):
        (self.discovery / "associations.txt").write_text("anakano_81|aadaniel|htcondor|normal|||\n")
        result = self.run_wrapper(live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("association authorizes", result.stderr)
        (self.discovery / "associations.txt").write_text("anakano_81|aadaniel||normal|||\n")
        (self.discovery / "gpu_nodes.txt").write_text("n1|gpu:a100:4|other|256000|64\nn2|gpu:a40:4|a100-40gb|256000|64\n")
        result = self.run_wrapper(live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("observed together", result.stderr)
        (self.discovery / "gpu_nodes.txt").write_text("n1|gpu:a100:4|a100-40gb|256000|64\n")
        (self.root / "runs" / "fixture").symlink_to(self.bin, target_is_directory=True)
        result = self.run_wrapper(live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("escapes", result.stderr)

    def test_snapshot_excludes_large_caches_and_rejects_mutable_symlinks(self):
        cache = self.root / ".cache"
        cache.mkdir()
        (cache / "large-generated.dat").write_bytes(b"generated data")
        integration = self.root / ".tower"
        integration.mkdir()
        (integration / "config.json").write_text('{"logs":{"manifest_file":"logs.json"}}\n')
        for directory in ("contracts", "schemas", "definitions", "passports"):
            (integration / directory).mkdir()
            (integration / directory / "example.json").write_text("{}\n")
        result = self.run_wrapper(live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "runs" / "fixture" / "source" / ".cache").exists())
        frozen = self.root / "runs" / "fixture" / "source" / ".tower"
        self.assertTrue((frozen / "config.json").is_file())
        for directory in ("contracts", "schemas", "definitions"):
            self.assertTrue((frozen / directory / "example.json").is_file())
        self.assertFalse((frozen / "passports").exists())
        (self.root / "configs" / "mutable.yaml").symlink_to(self.root / "README.md")
        result = self.run_wrapper("--run-id", "new", live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source snapshot symlink", result.stderr)

    def test_copy_deployment_without_git_has_explicit_hash_provenance(self):
        shutil.rmtree(self.root / ".git")
        result = self.run_wrapper(live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        run = self.root / "runs" / "fixture"
        self.assertIn("Git metadata unavailable", (run / "source-status.txt").read_text())
        self.assertTrue((run / "source-sha256.txt").is_file())

    def test_install_submission_waits_for_other_ssmo_pending_jobs(self):
        self.env["MOCK_USER_QUEUE"] = "9|SSMO-other-train-measure|PENDING|anakano_81|2|8G|gpu:a100:1"
        result = self.run_wrapper(live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("other SSMO queued/running", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_task_freeze_mismatch_stops_before_scientific_computation(self):
        self.stub("module", "exit 0\n")
        venv = self.root / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = mock-module\nversion = 3.12.8\n")
        interpreter = venv / "bin" / "python"
        interpreter.write_text('#!/bin/bash\nif [[ " $* " = *" pip freeze "* ]]; then printf "torch==2.10.0+cpu\\n"; else printf "forbidden computation\\n" >&2; exit 99; fi\n')
        interpreter.chmod(0o700)
        run = self.root / "runs" / "runtime"
        (run / "artifacts").mkdir(parents=True)
        (run / "artifacts" / "dependency-freeze.txt").write_text("torch==2.10.0+cu126\n")
        args = ["data", str(run), str(self.root / "configs" / "carc_smoke.yaml"),
                str(self.root), "a10040", "measure", "17", "none", "none"]
        result = subprocess.run(["bash", str(self.root / "scripts" / "run_stage.sh"), *args], env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("environment changed", result.stderr)
        self.assertNotIn("forbidden computation", result.stderr)

    def test_install_task_rechecks_other_queued_jobs_before_mutation(self):
        self.stub("module", "exit 0\n")
        self.stub("squeue", "printf 'SSMO-other-train-measure\\n'\n")
        run = self.root / "runs" / "runtime"
        run.mkdir()
        args = ["install", str(run), str(self.root / "configs" / "carc_smoke.yaml"),
                str(self.root), "a10040", "measure", "17", "none", "none"]
        result = subprocess.run(["bash", str(self.root / "scripts" / "run_stage.sh"), *args], env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("other SSMO jobs", result.stderr)
        self.assertFalse((self.root / ".venv").exists())

    def test_install_refuses_wrong_python_before_any_pip_mutation(self):
        self.stub("module", "exit 0\n")
        venv = self.verified_environment()
        (venv / "pyvenv.cfg").write_text("version = 3.11.9\n")
        interpreter = venv / "bin" / "python"
        interpreter.write_text('#!/bin/bash\nprintf "forbidden pip mutation\\n" >&2\nexit 99\n')
        run = self.root / "runs" / "runtime"
        run.mkdir()
        args = ["install", str(run), str(self.root / "configs" / "carc_smoke.yaml"),
                str(self.root), "a10040", "measure", "17", "none", "none"]
        result = subprocess.run(["bash", str(self.root / "scripts" / "run_stage.sh"), *args],
                                env=self.env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("venv Python version differs", result.stderr)
        self.assertNotIn("forbidden pip mutation", result.stderr)
        self.assertEqual((venv / "pyvenv.cfg").read_text(), "version = 3.11.9\n")

    def test_remote_task_shared_lock_is_readable_and_held_until_exit(self):
        real_flock = shutil.which("flock")
        self.assertIsNotNone(real_flock)
        self.stub("module", "exit 0\n")
        # Slurm starts a separate task process; an allocation-shell descriptor
        # cannot supply its lock. The stage must open its own descriptor.
        self.stub("srun", 'exec 9>&-\nwhile [[ "$1" = --* ]]; do\n'
                  'case "$1" in --output=*) app_out=${1#--output=} ;; --error=*) app_err=${1#--error=} ;; esac\n'
                  'shift\ndone\nexec "$@" > "$app_out" 2> "$app_err"\n')
        lock_check = self.bin / "network_flock.py"
        lock_check.write_text(
            "import fcntl, os, sys\n"
            "if '-s' in sys.argv:\n"
            "    fd = int(sys.argv[-1])\n"
            # Shared flock on a network filesystem can use POSIX read locks.
            # Exercise that real kernel check even on the local test filesystem.
            "    fcntl.lockf(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)\n"
            "    fcntl.lockf(fd, fcntl.LOCK_UN)\n"
            f"os.execv({real_flock!r}, [{real_flock!r}, *sys.argv[1:]])\n"
        )
        self.stub("flock", f'exec {shlex.quote(sys.executable)} {shlex.quote(str(lock_check))} "$@"\n')
        venv = self.verified_environment()
        interpreter = venv / "bin" / "python"
        interpreter.write_text(
            '#!/bin/bash\nset -euo pipefail\n'
            f'if [[ "${{1:-}} ${{2:-}} ${{3:-}} ${{4:-}}" = "-B -S -m singular_sensitivity.tower_reporting" ]]; then exec {shlex.quote(sys.executable)} "$@"; fi\n'
            'if [[ " $* " = *" pip freeze "* ]]; then printf "torch==2.10.0+cu126\\n"; exit 0; fi\n'
            '[[ "$1 $2 $3" = "-m singular_sensitivity.cli audit" ]] || exit 99\n'
            f'if {shlex.quote(real_flock)} -n -x "$SSMO_PROJECT_ROOT/local/venv.lock" -c true; then '
            'printf "shared lock missing during computation\\n" >&2; exit 99; fi\n'
            'printf "mock validation completed with shared lock\\n"\n'
        )
        args, env, attempt = self.batch_fixture()
        result = subprocess.run(["bash", str(self.root / "slurm" / "stage.sbatch"), *args],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mock validation completed with shared lock", (attempt / "logs/application.stdout.log").read_text())
        calls = [json.loads(line) for line in self.tower_calls.read_text().splitlines()]
        self.assertEqual([call[0] for call in calls], ["start", "finish"])
        self.assertEqual(calls[-1][-2:], ["--exit-code", "0"])
        released = subprocess.run([real_flock, "-n", "-x", str(self.root / "local/venv.lock"),
                                   "-c", "true"], capture_output=True, text=True)
        self.assertEqual(released.returncode, 0, released.stderr)

    def test_shared_lock_conflict_keeps_retry_exit_status(self):
        local = self.root / "local"
        local.mkdir()
        lock = local / "venv.lock"
        helper = shlex.quote(str(self.root / "scripts" / "carc_env.sh"))
        script = (f'source {helper}; ssmo_init_root || exit; '
                  'exec 9<> "$SSMO_PROJECT_ROOT/local/venv.lock"; '
                  "ssmo_lock_venv -s 'venv is being installed; retry after successful installation'")
        with lock.open("a+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(["bash", "-c", script], env=self.env,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("venv is being installed", result.stderr)

    def test_bad_lock_descriptor_is_not_reported_as_install_contention(self):
        helper = shlex.quote(str(self.root / "scripts" / "carc_env.sh"))
        script = (f'source {helper}; ssmo_init_root || exit; exec 9>&-; '
                  "ssmo_lock_venv -s 'venv is being installed; retry after successful installation'")
        result = subprocess.run(["bash", "-c", script], env=self.env,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(result.returncode, 3)
        self.assertIn("flock", result.stderr)
        self.assertNotIn("venv is being installed", result.stderr)

    def test_task_scripts_are_strict_and_propagate_srun_status(self):
        for path in [*REPO.joinpath("scripts").glob("*.sh"), *REPO.joinpath("slurm").glob("*.sbatch")]:
            text = path.read_text()
            self.assertTrue(text.startswith("#!/bin/bash"), path)
            if path.name != "carc_env.sh":
                self.assertIn("set -euo pipefail", text, path)
            subprocess.run(["bash", "-n", str(path)], check=True)
        # Run only a mocked srun; actual tasks are never executed by this test.
        for exit_code in (0, 23, 75, 143):
            with self.subTest(exit_code=exit_code):
                self.stub("srun", f"exit {exit_code}\n")
                args, env, attempt = self.batch_fixture(run_id=f"status-{exit_code}")
                result = subprocess.run(["bash", str(self.root / "slurm" / "stage.sbatch"), *args], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                calls = [json.loads(line) for line in self.tower_calls.read_text().splitlines()]
                self.assertEqual(calls[-1], ["finish", "--attempt-dir", str(attempt), "--exit-code", str(exit_code)])
                start = calls[-2]
                self.assertEqual(start[start.index("--scheduler-out") + 1], str(attempt.parents[1] / "logs/validate-measure-22.out"))
                self.assertEqual(start[start.index("--request-gpus") + 1], "0")
                self.assertEqual(start[start.index("--request-nodes") + 1], "1")

    def test_reporting_failure_retains_original_task_status_and_logs(self):
        self.stub("srun", "exit 75\n")
        args, env, attempt = self.batch_fixture()
        env["MOCK_TOWER_FINISH_EXIT"] = "19"
        result = subprocess.run(["bash", str(self.root / "slurm" / "stage.sbatch"), *args],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("reporting finalization failed", result.stderr)
        self.assertTrue((attempt / "logs/application.stderr.log").is_file())
        calls = [json.loads(line) for line in self.tower_calls.read_text().splitlines()]
        self.assertEqual(calls[-1][-1], "75")

    def test_reporting_start_failure_preserves_attempt_and_prevents_computation(self):
        self.stub("srun", 'printf "forbidden computation\\n" >&2\nexit 99\n')
        args, env, attempt = self.batch_fixture(stage="install")
        env["MOCK_TOWER_START_EXIT"] = "17"
        result = subprocess.run(["bash", str(self.root / "slurm" / "stage.sbatch"), *args],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertNotIn("forbidden computation", result.stderr)
        self.assertTrue((attempt / "logs").is_dir())
        calls = [json.loads(line) for line in self.tower_calls.read_text().splitlines()]
        self.assertEqual([call[0] for call in calls], ["start"])

    def test_task_signal_exit_is_recorded_without_scheduler_cause_inference(self):
        self.stub("srun", "kill -TERM $$\n")
        args, env, attempt = self.batch_fixture()
        result = subprocess.run(["bash", str(self.root / "slurm" / "stage.sbatch"), *args],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 143, result.stderr)
        calls = [json.loads(line) for line in self.tower_calls.read_text().splitlines()]
        self.assertEqual(calls[-1][-1], "143")
        self.assertNotIn("TIMEOUT", result.stdout + result.stderr)
        self.assertNotIn("OUT_OF_MEMORY", result.stdout + result.stderr)

    def test_native_gpu_trace_uses_one_physical_selector_and_cleans_owned_sampler(self):
        trace_calls = self.bin / "nvidia-calls.json"
        trace_stopped = self.bin / "nvidia-stopped.txt"
        sampler = self.bin / "mock_sampler.py"
        sampler.write_text(
            "import json, os, pathlib, signal, sys\n"
            f"calls = pathlib.Path({str(trace_calls)!r})\n"
            f"stopped = pathlib.Path({str(trace_stopped)!r})\n"
            "def finish(signum, frame):\n"
            "    stopped.write_text(str(os.getpid()))\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, finish)\n"
            "print('2026/10/04 00:00:00.000, 2, 7, 1024', flush=True)\n"
            "calls.write_text(json.dumps({'pid': os.getpid(), 'arguments': sys.argv[1:]}))\n"
            "signal.pause()\n"
        )
        self.stub("nvidia-smi", f"exec {shlex.quote(sys.executable)} -B -S {shlex.quote(str(sampler))} \"$@\"\n")
        self.stub("srun", f"for attempt in {{1..200}}; do [[ -s {shlex.quote(str(trace_calls))} ]] && exit 75; sleep 0.01; done\nexit 99\n")
        selectors = ("2", "GPU-12345678-1234-1234-1234-123456789abc")
        for index, selector in enumerate(selectors):
            with self.subTest(selector=selector):
                trace_calls.unlink(missing_ok=True)
                trace_stopped.unlink(missing_ok=True)
                args, env, attempt = self.batch_fixture(stage="train", profile="a10040", run_id=f"trace-{index}")
                env.update(SSMO_TOWER_GPU_COUNT="1", SSMO_TOWER_GPU_TYPE="a100",
                           SLURM_JOB_NUM_NODES="1", SLURM_JOB_GPUS=selector)
                result = subprocess.run(["bash", str(self.root / "slurm/stage.sbatch"), *args],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 75, result.stderr)
                calls = json.loads(trace_calls.read_text())
                self.assertEqual(calls["arguments"], ["--query-gpu=timestamp,index,utilization.gpu,memory.used",
                                                      "--format=csv,noheader,nounits", f"--id={selector}", "--loop=60"])
                self.assertEqual(trace_stopped.read_text(), str(calls["pid"]))
                with self.assertRaises(ProcessLookupError):
                    os.kill(calls["pid"], 0)
                self.assertIn(", 2, 7, 1024", (attempt / "logs/gpu-util-22.csv").read_text())
                context = json.loads((attempt / "mock-gpu-trace.json").read_text())
                self.assertEqual(context["SSMO_TOWER_GPU_TRACE_SELECTOR"], selector)
                self.assertEqual(context["SSMO_TOWER_GPU_TRACE_STATUS"], "eligible")

    def test_native_gpu_trace_skips_unverified_selectors_and_node_counts(self):
        self.stub("nvidia-smi", 'printf "forbidden GPU probe\\n" >&2\nexit 99\n')
        self.stub("srun", "exit 0\n")
        cases = [({}, "unverified_single_node"),
                 ({"SLURM_JOB_NUM_NODES": "2", "SLURM_JOB_GPUS": "0"}, "unverified_single_node"),
                 ({"SLURM_JOB_NUM_NODES": "1", "SLURM_NNODES": "2", "SLURM_JOB_GPUS": "0"}, "unverified_single_node")]
        for selector in ("", "0,1", "0-1", "cuda:0", "MIG-GPU-123", "GPU-not-a-uuid"):
            cases.append(({"SLURM_JOB_NUM_NODES": "1", "SLURM_JOB_GPUS": selector}, "missing_physical_selector"))
        for index, (updates, reason) in enumerate(cases):
            with self.subTest(updates=updates):
                args, env, attempt = self.batch_fixture(stage="train", profile="a10040", run_id=f"skip-{index}")
                env.pop("SLURM_JOB_NUM_NODES", None)
                env.pop("SLURM_NNODES", None)
                env.pop("SLURM_JOB_GPUS", None)
                env.update(SSMO_TOWER_GPU_COUNT="1", SSMO_TOWER_GPU_TYPE="a100", **updates)
                result = subprocess.run(["bash", str(self.root / "slurm/stage.sbatch"), *args],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("forbidden GPU probe", result.stderr)
                self.assertFalse((attempt / "logs/gpu-util-22.csv").exists())
                context = json.loads((attempt / "mock-gpu-trace.json").read_text())
                self.assertEqual(context["SSMO_TOWER_GPU_TRACE_STATUS"], "skipped")
                self.assertEqual(context["SSMO_TOWER_GPU_TRACE_REASON"], reason)
                self.assertIsNone(context["SSMO_TOWER_GPU_TRACE_SELECTOR"])

    def test_native_gpu_trace_skips_cpu_jobs_and_explicit_disablement(self):
        self.stub("nvidia-smi", 'printf "forbidden GPU probe\\n" >&2\nexit 99\n')
        self.stub("srun", "exit 0\n")
        for index, (gpu_count, preference, reason) in enumerate((("0", "1", "cpu_request"),
                                                               ("1", "0", "disabled"),
                                                               ("1", "invalid", "invalid_trace_preference"))):
            with self.subTest(reason=reason):
                args, env, attempt = self.batch_fixture(run_id=f"disabled-{index}")
                env.update(SSMO_TOWER_GPU_COUNT=gpu_count, SSMO_TOWER_GPU_TRACE=preference,
                           SLURM_JOB_NUM_NODES="1", SLURM_JOB_GPUS="0")
                result = subprocess.run(["bash", str(self.root / "slurm/stage.sbatch"), *args],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("forbidden GPU probe", result.stderr)
                self.assertFalse((attempt / "logs/gpu-util-22.csv").exists())
                context = json.loads((attempt / "mock-gpu-trace.json").read_text())
                self.assertEqual(context["SSMO_TOWER_GPU_TRACE_REASON"], reason)


if __name__ == "__main__":
    unittest.main()
