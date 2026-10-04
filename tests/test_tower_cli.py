"""Real CLI reporting preserves science, separate attempts and scoped logs."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from singular_sensitivity.cli import main, _audit, _report
from singular_sensitivity.runtime import initialize_storage, project_root
from singular_sensitivity import tower_reporting as tower


class TowerCliTests(unittest.TestCase):
    def setUp(self):
        initialize_storage()
        self.temporary = tempfile.TemporaryDirectory(prefix="ssmo-tower-cli-", dir=project_root() / ".cache/tmp")
        self.directory = Path(self.temporary.name)
        self.config = project_root() / "configs/smoke.yaml"

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, arguments, **environment):
        values = {"SSMO_TOWER_RUN_DIR": "", "SSMO_TOWER_ACTIVE_COMMAND": "", **environment}
        with patch.dict(os.environ, values), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = main(arguments)
            self.assertEqual(os.environ.get("SSMO_TOWER_RUN_DIR"), values["SSMO_TOWER_RUN_DIR"])
            self.assertEqual(os.environ.get("SSMO_TOWER_ACTIVE_COMMAND"), values["SSMO_TOWER_ACTIVE_COMMAND"])
        return result

    def test_real_generate_and_inverse_publish_local_reports_with_unknown_allocation(self):
        for command in ("generate", "inverse"):
            run = self.directory / command
            self.assertEqual(self.invoke([command, "--config", str(self.config), "--run-dir", str(run)]), 0)
            attempts = list((run / "tower").iterdir())
            self.assertEqual(len(attempts), 1)
            attempt = attempts[0]
            summary = json.loads((attempt / "summary.json").read_text())
            self.assertEqual(summary["state"], "COMPLETED")
            self.assertEqual(summary["exit_code"], 0)
            self.assertIn(command, summary["results"])
            for unknown in ("cpus", "nodes", "gpus", "partition", "mem_bytes", "time_seconds"):
                self.assertNotIn(unknown, summary)
            self.assertIn('"status"', (attempt / "logs/application.stdout.log").read_text())
            rows = [json.loads(line) for line in (attempt / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(row["t"] >= 0 for row in rows))

    def test_real_missing_manifest_failure_is_retained_and_retry_has_fresh_identity(self):
        identifiers = []
        for number in (1, 2):
            run = self.directory / f"failure-{number}"
            code = self.invoke(["train", "--config", str(self.config), "--run-dir", str(run),
                                "--manifest", str(self.directory / "missing.json"), "--device", "cpu"])
            self.assertEqual(code, 1)
            self.assertTrue((run / "failure.json").is_file())
            attempt = next((run / "tower").iterdir())
            summary = json.loads((attempt / "summary.json").read_text())
            self.assertEqual((summary["state"], summary["exit_code"]), ("FAILED", 1))
            self.assertIn("FileNotFoundError", (attempt / "logs/application.stderr.log").read_text())
            identifiers.append(summary["id"])
        self.assertNotEqual(*identifiers)

    def test_nested_cli_fixture_does_not_pollute_surrounding_allocated_audit(self):
        pipeline = self.directory / "validation-pipeline"
        pipeline.mkdir()
        parent = pipeline / "tower/validate"
        tower.start_attempt(parent, pipeline_dir=pipeline, stage="validate", job_id="fixture-321",
                            metadata={"simulated_scheduler": True})
        before = {name: (parent / name).read_bytes() for name in ("run.json", "metrics.jsonl", "outputs/analytics.json")}
        other = self.directory / "independent-fixture"
        self.assertEqual(self.invoke(["generate", "--config", str(self.config), "--run-dir", str(other)],
                                    SSMO_TOWER_RUN_DIR=str(parent), SSMO_TOWER_ACTIVE_COMMAND="audit"), 0)
        self.assertEqual(before, {name: (parent / name).read_bytes() for name in before})
        self.assertEqual(len(list((other / "tower").iterdir())), 1)
        local = json.loads((next((other / "tower").iterdir()) / "run.json").read_text())
        self.assertNotIn("job_id", local)
        self.assertTrue(local["provenance"]["script"].endswith("singular_sensitivity/cli.py"))

    def test_pipeline_cli_reuses_stage_attempt_without_finalizing_the_batch(self):
        pipeline = self.directory / "pipeline"
        pipeline.mkdir()
        parent = pipeline / "tower/data"
        tower.start_attempt(parent, pipeline_dir=pipeline, stage="data", job_id="fixture-322",
                            metadata={"simulated_scheduler": True})
        artifact = pipeline / "artifacts/data"
        self.assertEqual(self.invoke(["generate", "--config", str(self.config), "--run-dir", str(artifact)],
                                    SSMO_TOWER_RUN_DIR=str(parent)), 0)
        self.assertFalse((artifact / "tower").exists())
        self.assertFalse((parent / "summary.json").exists())
        self.assertEqual(json.loads((parent / "run.json").read_text())["state"], "RUNNING")
        self.assertIn("generate", json.loads((parent / "outputs/analytics.json").read_text())["results"])

    def test_allocated_audit_isolates_nested_workloads_and_restores_its_context(self):
        pipeline = self.directory / "audit-pipeline"
        pipeline.mkdir()
        parent = pipeline / "tower/validate"
        tower.start_attempt(parent, pipeline_dir=pipeline, stage="validate", job_id="fixture-323")
        before = (parent / "metrics.jsonl").read_bytes()
        nested = self.directory / "audit-test-generated"
        audit = pipeline / "artifacts/validate"
        audit.mkdir(parents=True)
        def fixture_workload():
            self.assertNotIn("SSMO_TOWER_RUN_DIR", os.environ)
            self.assertEqual(main(["generate", "--config", str(self.config), "--run-dir", str(nested)]), 0)
        suite = unittest.TestSuite([unittest.FunctionTestCase(fixture_workload)])
        with patch.dict(os.environ, {"SSMO_TOWER_RUN_DIR": str(parent), "SSMO_TOWER_ACTIVE_COMMAND": "audit"}), \
                patch("unittest.TestLoader.discover", return_value=suite), redirect_stdout(io.StringIO()):
            result = _audit(audit)
            self.assertEqual(os.environ["SSMO_TOWER_RUN_DIR"], str(parent))
            self.assertEqual(os.environ["SSMO_TOWER_ACTIVE_COMMAND"], "audit")
        self.assertEqual(result["passed"], 1)
        self.assertEqual((parent / "metrics.jsonl").read_bytes(), before)
        self.assertEqual(len(list((nested / "tower").iterdir())), 1)

    def test_report_excludes_reporting_sidecars_and_plan_creates_no_attempt(self):
        original = self.directory / "original"
        (original / "tower/attempt/outputs").mkdir(parents=True)
        (original / "training_summary.json").write_text('{"status":"complete"}')
        (original / "tower/attempt/outputs/analytics.json").write_text('{"status":"must not recursively aggregate"}')
        report = _report(self.directory / "report", original)
        self.assertEqual([row["path"] for row in report["artifacts"]], ["training_summary.json"])
        absent = self.directory / "never-created"
        self.assertEqual(self.invoke(["plan", "--config", str(self.config)], SSMO_TOWER_RUN_DIR=str(absent)), 0)
        self.assertFalse(absent.exists())


if __name__ == "__main__":
    unittest.main()
