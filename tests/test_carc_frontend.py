"""Frontend integration with mocked policy and scheduler; no live CARC calls."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from singular_sensitivity.runtime import confined_path


REPO = Path(__file__).resolve().parents[1]
CARC_ROOT = "/home1/aadaniel/projects/SSMO"


class CarcFrontendTests(unittest.TestCase):
    def setUp(self):
        base = confined_path("local/test-tmp")
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ssmo-frontend-", dir=base)
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        shutil.copytree(REPO / "scripts", self.root / "scripts")
        # Allocated audits execute from a deliberately read-only source snapshot.
        # Only the disposable test fixture becomes writable.
        (self.root / "scripts").chmod(0o700)
        for path in (self.root / "scripts").glob("*.sh"):
            path.chmod(0o700)
            path.write_text(path.read_text().replace(CARC_ROOT, str(self.root)))
        (self.root / "configs").mkdir()
        shutil.copyfile(REPO / "configs" / "carc_site.env", self.root / "configs" / "carc_site.env")
        for name in ("carc_smoke.yaml", "carc_pilot.yaml"):
            (self.root / "configs" / name).write_text("name: mock\n")
        self.calls = self.root / "calls.txt"
        helper = self.root / "scripts" / "carc_env.sh"
        with helper.open("a") as stream:
            stream.write('\nssmo_environment_ready() { [[ ${MOCK_ENV_READY:-0} = 1 ]]; }\n')
        self.bin = Path(self.temp.name) / "bin"
        self.bin.mkdir()
        self.stub(self.bin / "id", "printf 'aadaniel\\n'\n")
        self.stub(self.bin / "squeue", "printf 'queue %s\\n' \"$*\"\n")
        self.stub(self.bin / "sacct", "printf 'accounting %s\\n' \"$*\"\n")
        self.stub(self.root / "scripts" / "discover_carc.sh", """printf 'discovery %s\\n' "$*" >> "$MOCK_CALLS"
[[ ${MOCK_DISCOVERY_FAIL:-0} = 0 ]] || exit 17
mkdir -p "$SSMO_PROJECT_ROOT/runs/$2/discovery"
printf 'fresh policy\\n' > "$SSMO_PROJECT_ROOT/runs/$2/discovery/manifest.txt"
""")
        self.stub(self.root / "scripts" / "submit.sh", """printf 'submit' >> "$MOCK_CALLS"
printf '\\t%s' "$@" >> "$MOCK_CALLS"
printf '\\n' >> "$MOCK_CALLS"
printf 'mock complete plan\\n'
""")
        self.env = dict(os.environ, SSMO_PROJECT_ROOT=str(self.root), MOCK_CALLS=str(self.calls),
                        PATH=f"{self.bin}:{os.environ['PATH']}")
        for key in ("SSMO_ROOT", "SSMO_SITE_CONFIG", "SSMO_GPU_PROFILE", "SSMO_PYTHON_MODULE",
                    "SSMO_ACCOUNT_SLOTS", "SSMO_ACCOUNT_FREE_CPUS", "SSMO_ACCOUNT_FREE_MEM_GB",
                    "SSMO_ACCOUNT_FREE_GPUS", "SSMO_MAX_PROJECT_JOBS"):
            self.env.pop(key, None)

    def tearDown(self):
        self.temp.cleanup()

    def stub(self, path, body):
        path.write_text("#!/bin/bash\nset -euo pipefail\n" + body)
        path.chmod(0o700)

    def run_frontend(self, *args):
        return subprocess.run(["bash", str(self.root / "scripts" / "carc.sh"), *args],
                              env=self.env, text=True, capture_output=True)

    def submitted(self):
        return [line.split("\t")[1:] for line in self.calls.read_text().splitlines()
                if line.startswith("submit\t")]

    def reviewed_capacity(self):
        self.env.update(SSMO_ACCOUNT_SLOTS="9", SSMO_ACCOUNT_FREE_CPUS="4",
                        SSMO_ACCOUNT_FREE_MEM_GB="8", SSMO_ACCOUNT_FREE_GPUS="1")

    def previous_run(self, name="previous", seed=29):
        run = self.root / "runs" / name
        source = run / "source"
        source.mkdir(parents=True)
        for relative in ("singular_sensitivity/__init__.py", "requirements/base.txt", "pyproject.toml"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# science fixture\n")
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        hashes = subprocess.run(["bash", "-c", "find . -type f -print0 | sort -z | xargs -0 sha256sum"],
                                cwd=source, text=True, capture_output=True, check=True).stdout
        (run / "source-sha256.txt").write_text(hashes)
        config = run / "config.yaml"
        config.write_text("name: original frozen config\n")
        (run / "run-sha256.txt").write_text(subprocess.run(["sha256sum", str(config)],
                                                          text=True, capture_output=True, check=True).stdout)
        manifest = run / "artifacts" / "data" / "parents.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n")
        for method in ("measure", "state_only"):
            training = run / "artifacts" / f"train-{method}-seed{seed}"
            training.mkdir()
            for checkpoint in ("last.pt", "best.pt"):
                (training / checkpoint).write_bytes(b"mock checkpoint; never loaded")
        (run / "submission-manifest.txt").write_text(
            f"profile=l40s\npython_module=python/3.12.8\nmethod=measure\nseed={seed}\nmanifest={manifest}\n")
        (run / "jobs.tsv").write_text("stage\tmethod\tseed\tjob_id\tdependency\n"
                                     f"train\tmeasure\t{seed}\t101\tnone\n"
                                     f"train\tstate_only\t{seed}\t102\t101\n")
        return run

    def test_default_smoke_only_previews_and_creates_no_run_or_cache(self):
        result = self.run_frontend("smoke", "--run-id", "smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.submitted()[0]
        self.assertIn("smoke", args)
        self.assertNotIn("--submit", args)
        self.assertNotIn("discovery", self.calls.read_text())
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "local").exists())
        self.assertIn("status --run-id smoke", result.stdout)
        self.assertEqual(args[args.index("--python-module") + 1], "python/3.12.8")

    def test_live_action_captures_fresh_policy_and_forwards_reviewed_capacity(self):
        self.reviewed_capacity()
        result = self.run_frontend("pilot", "--run-id", "pilot", "--gpu-profile", "a30", "--submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = self.calls.read_text().splitlines()
        self.assertTrue(lines[0].startswith("discovery --run-id ssmo-policy-"))
        args = self.submitted()[0]
        self.assertEqual(args[args.index("--discovery") + 1], lines[0].split()[-1])
        self.assertEqual(args[args.index("--account-slots") + 1], "9")
        self.assertEqual(args[args.index("--account-free-cpus") + 1], "4")
        self.assertEqual(args[args.index("--account-free-mem-gb") + 1], "8")
        self.assertEqual(args[args.index("--account-free-gpus") + 1], "1")
        self.assertEqual(args[args.index("--gpu-profile") + 1], "a30")
        self.assertIn("--submit", args)

    def test_unreviewed_capacity_blocks_before_any_writes_or_discovery(self):
        result = self.run_frontend("smoke", "--run-id", "smoke", "--submit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("all four reviewed", result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / "runs").exists())

    def test_discovery_failure_prevents_submission(self):
        self.reviewed_capacity()
        self.env["MOCK_DISCOVERY_FAIL"] = "1"
        result = self.run_frontend("setup", "--run-id", "setup", "--submit")
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(self.submitted(), [])

    def test_verified_environment_skips_install_and_reinstall_overrides(self):
        self.env["MOCK_ENV_READY"] = "1"
        result = self.run_frontend("smoke", "--run-id", "reuse")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--skip-install", self.submitted()[0])
        result = self.run_frontend("setup", "--run-id", "install", "--reinstall")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--skip-install", self.submitted()[1])

    def test_resume_recovers_frozen_config_manifest_method_seed_and_hardware(self):
        previous = self.previous_run()
        result = self.run_frontend("resume", "--from-run", "previous", "--run-id", "resume", "--method", "state_only")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.submitted()[0]
        self.assertEqual(args[args.index("--config") + 1], str(previous / "config.yaml"))
        self.assertEqual(args[args.index("--manifest") + 1], str(previous / "artifacts/data/parents.json"))
        self.assertEqual(args[args.index("--seed") + 1], "29")
        self.assertEqual(args[args.index("--method") + 1], "state_only")
        self.assertEqual(args[args.index("--gpu-profile") + 1], "l40s")
        self.assertEqual(args[args.index("--resume") + 1], str(previous / "artifacts/train-state_only-seed29/last.pt"))

    def test_changed_science_snapshot_or_frozen_config_blocks_recovery(self):
        previous = self.previous_run()
        for change in ("science", "snapshot", "config"):
            with self.subTest(change=change):
                path = {"science": self.root / "singular_sensitivity/__init__.py",
                        "snapshot": previous / "source/singular_sensitivity/__init__.py",
                        "config": previous / "config.yaml"}[change]
                original = path.read_text()
                path.write_text(original + "# changed\n")
                result = self.run_frontend("resume", "--from-run", "previous", "--run-id", "resume")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.calls.exists())
                path.write_text(original)

    def test_missing_best_checkpoint_keeps_original_training_dependency(self):
        previous = self.previous_run()
        (previous / "artifacts/train-measure-seed29/best.pt").unlink()
        result = self.run_frontend("evaluate", "--from-run", "previous", "--run-id", "evaluation")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.submitted()[0]
        self.assertEqual(args[args.index("--after") + 1], "101")
        self.assertEqual(args[args.index("--stage") + 1], "evaluate")
        self.assertIn("Waiting for original training", result.stdout)

    def test_report_uses_readonly_previous_artifact_source(self):
        previous = self.previous_run()
        result = self.run_frontend("report", "--from-run", "previous", "--run-id", "report")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.submitted()[0]
        self.assertEqual(args[args.index("--report-from") + 1], str(previous / "artifacts"))
        self.assertEqual(args[args.index("--after") + 1], "102")
        self.assertFalse((self.root / "runs/report").exists())
        self.assertIn("earlier artifacts are preserved", result.stdout)

    def test_pilot_control_report_uses_selected_checkpoint_instead_of_default_measure(self):
        previous = self.previous_run()
        metadata = previous / "submission-manifest.txt"
        with metadata.open("a") as stream:
            stream.write(f"checkpoint={previous / 'artifacts/train-measure-seed29/best.pt'}\n")
        original = metadata.read_bytes()
        result = self.run_frontend("report", "--from-run", "previous", "--run-id", "control-report",
                                   "--method", "state_only")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.submitted()[0]
        self.assertEqual(args[args.index("--checkpoint") + 1], str(previous / "artifacts/train-state_only-seed29/best.pt"))
        self.assertEqual(args[args.index("--method") + 1], "state_only")
        self.assertEqual(args[args.index("--seed") + 1], "29")
        self.assertEqual(args[args.index("--after") + 1], "102")
        self.assertEqual(metadata.read_bytes(), original)

    def test_report_from_standalone_evaluation_preserves_inputs_and_final_job(self):
        previous = self.previous_run()
        trained_checkpoint = previous / "artifacts/train-measure-seed29/best.pt"
        with (previous / "submission-manifest.txt").open("a") as stream:
            stream.write(f"checkpoint={trained_checkpoint}\n")
        (previous / "jobs.tsv").write_text("stage\tmethod\tseed\tjob_id\tdependency\n"
                                          "evaluate\tmeasure\t29\t104\t101\n")
        result = self.run_frontend("report", "--from-run", "previous", "--run-id", "report")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.submitted()[0]
        self.assertEqual(args[args.index("--checkpoint") + 1], str(trained_checkpoint))
        self.assertEqual(args[args.index("--after") + 1], "104")
        self.assertEqual(args[args.index("--seed") + 1], "29")
        self.assertEqual(args[args.index("--method") + 1], "measure")
        self.assertIn("failed predecessor leaves it pending", result.stdout)

    def test_report_requires_artifact_directory_and_inapplicable_flags_fail(self):
        previous = self.previous_run()
        for command in ("resume", "evaluate", "report", "status", "discover"):
            for option in ("--skip-install", "--reinstall"):
                with self.subTest(command=command, option=option):
                    result = self.run_frontend(command, option)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("apply only", result.stderr)
                    self.assertFalse(self.calls.exists())
        shutil.rmtree(previous / "artifacts")
        result = self.run_frontend("report", "--from-run", "previous", "--run-id", "report")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("artifact directory is missing", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_pipeline_and_recovery_flags_with_no_effect_are_rejected_before_writes(self):
        cases = [(command, "--method", "state_only") for command in ("setup", "smoke", "pilot")]
        cases += [(command, "--checkpoint", "runs/old/best.pt") for command in ("setup", "smoke", "pilot", "resume")]
        cases += [(command, "--resume", "runs/old/last.pt") for command in ("setup", "smoke", "pilot", "evaluate", "report")]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.run_frontend(*arguments, "--submit")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("applies only", result.stderr)
                self.assertFalse(self.calls.exists())
                self.assertFalse((self.root / "runs").exists())
                self.assertFalse((self.root / "local").exists())

    def test_inspection_rejects_action_flags_including_discovery_dry_run(self):
        for command in ("discover", "status"):
            for arguments in (("--submit",), ("--dry-run",), ("--config", "configs/carc_smoke.yaml"),
                              ("--gpu-profile", "a40"), ("--after", "101"), ("--seed", "29"),
                              ("--account-slots", "9"), ("--method", "measure")):
                with self.subTest(command=command, arguments=arguments):
                    result = self.run_frontend(command, "--run-id", "inspection", *arguments)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("accepts only --run-id", result.stderr)
                    self.assertFalse(self.calls.exists())
                    self.assertFalse((self.root / "runs").exists())
                    self.assertFalse((self.root / "local").exists())

    def test_status_inspects_only_recorded_jobs_without_submitting(self):
        self.previous_run()
        result = self.run_frontend("status", "--run-id", "previous")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-j 101,102", result.stdout)
        self.assertIn("report.json", result.stdout)
        self.assertFalse(self.calls.exists())

    def test_invalid_paths_and_mismatched_recovery_settings_are_rejected(self):
        self.previous_run()
        cases = [("smoke", "--run-id", "../escape"),
                 ("resume", "--from-run", "previous", "--gpu-profile", "a40"),
                 ("resume", "--from-run", "previous", "--config", "configs/carc_smoke.yaml"),
                 ("evaluate", "--from-run", "previous", "--checkpoint", "/etc/passwd")]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_frontend(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.calls.exists())

    def test_site_path_cannot_escape_root(self):
        self.env["SSMO_SITE_CONFIG"] = "/etc/passwd"
        result = self.run_frontend("smoke", "--run-id", "smoke")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("escapes project", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_local_site_settings_take_precedence_and_explicit_override_is_respected(self):
        local = self.root / "local"
        local.mkdir()
        (local / "carc_site.env").write_text("SSMO_GPU_PROFILE=a40\nSSMO_PYTHON_MODULE=python/3.12.8\n")
        result = self.run_frontend("smoke", "--run-id", "local-site")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.submitted()[0]
        self.assertEqual(args[args.index("--gpu-profile") + 1], "a40")
        self.env["SSMO_SITE_CONFIG"] = "configs/carc_site.env"
        result = self.run_frontend("smoke", "--run-id", "explicit-site")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.submitted()[1]
        self.assertEqual(args[args.index("--gpu-profile") + 1], "a10040")


if __name__ == "__main__":
    unittest.main()
