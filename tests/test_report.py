"""Reports reuse prior evidence without rewriting it; tasks respect CPU limits."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from singular_sensitivity.cli import _report, main
from singular_sensitivity.config import load_config
from singular_sensitivity.runtime import initialize_storage, project_root


class ReportTests(unittest.TestCase):
    def setUp(self):
        initialize_storage()
        self.temporary = tempfile.TemporaryDirectory(prefix="ssmo-report-", dir=project_root() / ".cache/tmp")
        self.root = Path(self.temporary.name)
        self.previous = self.root / "previous" / "artifacts"
        self.current = self.root / "current" / "artifacts"
        self.previous.mkdir(parents=True)
        self.current.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, scope, relative, value):
        path = scope / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_report_includes_prior_and_new_evidence_without_changing_prior_files(self):
        self.write_json(self.previous, "train-measure/training_summary.json", {"status": "complete", "steps_completed": 20})
        self.write_json(self.previous, "evaluate/evaluation.json", {"status": "evaluated", "worst_error": 0.07})
        for path in ("report.json", "provenance.json", "data/parents.json", "source/test.json", ".cache/test.json"):
            self.write_json(self.previous, path, {"status": "must not be copied"})
        self.write_json(self.current, "representation/representation.json", {"status": "passed"})
        originals = {path: path.read_bytes() for path in self.previous.rglob("*") if path.is_file()}

        result = _report(self.current, self.previous)

        self.assertEqual(len(result["artifacts"]), 3)
        by_source = {artifact["source_directory"] for artifact in result["artifacts"]}
        self.assertEqual(by_source, {str(self.previous), str(self.current)})
        self.assertEqual({artifact["path"] for artifact in result["artifacts"]},
                         {"train-measure/training_summary.json", "evaluate/evaluation.json", "representation/representation.json"})
        self.assertFalse(result["claims"]["current_instance_artifacts_only"])
        self.assertFalse(result["claims"]["research_advantage_established"])
        self.assertEqual(json.loads((self.current / "report.json").read_text()), result)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_same_source_and_overlapping_sources_do_not_duplicate_artifacts(self):
        self.write_json(self.current, "evaluation.json", {"status": "evaluated"})
        identical = _report(self.current, self.current)
        self.assertEqual(identical["artifact_sources"], [str(self.current)])
        self.assertEqual(len(identical["artifacts"]), 1)
        overlapping = _report(self.current, self.root)
        self.assertEqual(len(overlapping["artifacts"]), 1)

    def test_external_and_escaping_source_paths_are_rejected(self):
        escape = self.root / "external"
        escape.symlink_to("/tmp", target_is_directory=True)
        for source in (Path("/tmp"), escape):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    _report(self.current, source)
        self.assertFalse((self.current / "report.json").exists())

    def test_escaping_artifact_symlink_is_rejected_before_reading(self):
        (self.previous / "outside.json").symlink_to("/etc/passwd")
        with self.assertRaises(ValueError):
            _report(self.current, self.previous)
        self.assertFalse((self.current / "report.json").exists())

    def test_missing_source_is_rejected_and_previous_report_is_not_modified(self):
        existing = self.write_json(self.current, "report.json", {"status": "previous attempt"})
        before = existing.read_bytes()
        with self.assertRaisesRegex(ValueError, "existing"):
            _report(self.current, self.root / "absent")
        self.assertEqual(existing.read_bytes(), before)

    def test_explicit_cache_and_snapshot_sources_are_rejected(self):
        for name in ("source", ".cache", ".venv"):
            directory = self.root / name
            directory.mkdir()
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "snapshot or cache"):
                    _report(self.current, directory)
        self.assertFalse((self.current / "report.json").exists())

    def test_cli_report_accepts_explicit_existing_artifact_source(self):
        self.write_json(self.previous, "training_summary.json", {"status": "complete", "steps_completed": 5})
        with redirect_stdout(io.StringIO()):
            result = main(["report", "--config", str(project_root() / "configs/smoke.yaml"),
                           "--run-dir", str(self.current), "--artifact-source", str(self.previous)])
        self.assertEqual(result, 0)
        report = json.loads((self.current / "report.json").read_text())
        self.assertEqual(report["artifacts"][0]["source_directory"], str(self.previous))

    def test_train_and_evaluate_use_actual_allocation_thread_limit(self):
        import torch
        config = deepcopy(load_config(project_root() / "configs/smoke.yaml"))
        config["threads"] = 8
        config["device"] = "cpu"
        configuration = self.write_json(self.root, "allocated-config.json", config)
        previous_threads = torch.get_num_threads()
        observed = []

        def bounded_workload(*args, **kwargs):
            observed.append(torch.get_num_threads())
            return {"status": "complete"}

        try:
            for command, function in (("train", "singular_sensitivity.training.train"),
                                      ("evaluate", "singular_sensitivity.validation.evaluate")):
                arguments = [command, "--config", str(configuration), "--run-dir", str(self.current / command),
                             "--manifest", str(self.previous / "parents.json"), "--device", "cpu"]
                if command == "evaluate":
                    arguments += ["--checkpoint", str(self.previous / "best.pt")]
                with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2", "SLURM_JOB_ID": ""}), \
                     patch("singular_sensitivity.data.load_manifest", return_value={}), \
                     patch(function, side_effect=bounded_workload), \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(main(arguments), 0)
                recorded = json.loads((self.current / command / "provenance.json").read_text())
                self.assertEqual(recorded["task_threads"], 2)
            self.assertEqual(observed, [2, 2])
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
