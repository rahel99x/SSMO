"""Pure mocked CARC integration tests: never invoke a live scheduler."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[1]
CARC_ROOT = "/home1/aadaniel/projects/SSMO"


class SlurmWorkflowTests(unittest.TestCase):
    def setUp(self):
        # On CARC even test temporaries remain in the project, never /tmp.
        base = Path(os.environ.get("TMPDIR", REPO / "local" / "test-tmp"))
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
        self.env = dict(os.environ, SSMO_ROOT=str(self.root), MOCK_CALLS=str(self.calls),
                        MOCK_COUNTER=str(self.counter), PATH=f"{self.bin}:{os.environ['PATH']}")
        self.discovery = self.root / "runs" / "observed" / "discovery"
        self.discovery.mkdir(parents=True)
        values = {
            "status.tsv": "command\texit_code\nidentity\t0\n",
            "manifest.txt": "observed test fixture\n",
            "identity.txt": "aadaniel\n",
            "associations.txt": "anakano_81|aadaniel||normal|||\n",
            "modules.txt": "python/3.11.9 python/3.12.0\n",
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

    def run_wrapper(self, *args, live=False):
        flags = ["--run-id", "fixture"]
        if live:
            flags += ["--submit", "--discovery", "observed", "--account-slots", "10",
                      "--account-free-cpus", "4", "--account-free-mem-gb", "8",
                      "--account-free-gpus", "1"]
        return subprocess.run(["bash", str(self.root / "scripts" / "submit.sh"), *flags, *args],
                              env=self.env, text=True, capture_output=True)

    def test_dry_run_has_exact_account_resources_and_does_not_mutate(self):
        result = self.run_wrapper("--after", "77")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("sbatch ")]
        self.assertEqual(len(commands), 7)
        self.assertTrue(all("--account=anakano_81" in cmd for cmd in commands))
        self.assertIn("--dependency=afterok:77", commands[0])
        self.assertIn("--partition=main", commands[0])
        self.assertIn("--partition=gpu", commands[3])
        self.assertIn("--gpus-per-task=a100:1", commands[3])
        self.assertIn("--constraint=a100-40gb", commands[3])
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
        script = f'pre=$(set +o); source {helper}; post=$(set +o); [[ "$pre" = "$post" ]] || exit 5; SSMO_ROOT=/nonexistent; ssmo_env; printf "shell-alive\\n"'
        result = subprocess.run(["bash", "-c", script], env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("shell-alive", result.stdout)

    def test_account_queue_changes_and_caps_block_before_submit(self):
        self.env["MOCK_QUEUE"] = "9|other|other-project|RUNNING|4|8G|gpu:a100:1"
        result = self.run_wrapper(live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("queue changed", result.stderr)
        self.assertFalse(self.calls.exists())
        self.env.pop("MOCK_QUEUE")
        result = self.run_wrapper("--account-free-cpus", "1", live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("insufficient", result.stderr)

    def test_unknown_gpu_tag_is_rejected_and_l40s_is_distinct(self):
        result = self.run_wrapper("--gpu-profile", "l40s", live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--gpus-per-task=l40s:1", self.calls.read_text())
        (self.discovery / "gpu_nodes.txt").write_text("n1|gpu:l40:4|(null)|256000|64\n")
        result = self.run_wrapper("--run-id", "second", "--gpu-profile", "l40s", live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not observed", result.stderr)

    def test_failed_stale_discovery_and_pipeline_limits(self):
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
        result = self.run_wrapper("--pipeline", "pilot")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("9 jobs", result.stderr)

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
        result = self.run_wrapper(live=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "runs" / "fixture" / "source" / ".cache").exists())
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

    def test_install_submission_waits_for_other_project_pending_jobs(self):
        self.env["MOCK_USER_QUEUE"] = "9|SSMO-other-train-measure|PENDING|anakano_81|2|8G|gpu:a100:1"
        result = self.run_wrapper(live=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("other SSMO queued/running", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_task_freeze_mismatch_stops_before_scientific_computation(self):
        self.stub("module", "exit 0\n")
        venv = self.root / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = mock-module\n")
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

    def test_task_scripts_are_strict_and_propagate_srun_status(self):
        for path in [*REPO.joinpath("scripts").glob("*.sh"), *REPO.joinpath("slurm").glob("*.sbatch")]:
            text = path.read_text()
            self.assertTrue(text.startswith("#!/bin/bash"), path)
            if path.name != "carc_env.sh":
                self.assertIn("set -euo pipefail", text, path)
            subprocess.run(["bash", "-n", str(path)], check=True)
        # Run only a mocked srun; actual tasks are never executed by this test.
        self.stub("srun", "exit 23\n")
        source = self.root / "runs" / "job" / "source"
        source.mkdir(parents=True)
        shutil.copytree(self.root / "scripts", source / "scripts")
        args = ["validate", str(self.root / "runs" / "job"), str(self.root / "configs" / "carc_smoke.yaml"), str(source), "cpu", "measure", "7", "none", "none"]
        env = dict(self.env, SLURM_JOB_ID="22", SLURM_JOB_ACCOUNT="anakano_81")
        result = subprocess.run(["bash", str(self.root / "slurm" / "stage.sbatch"), *args], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 23, result.stderr)


if __name__ == "__main__":
    unittest.main()
