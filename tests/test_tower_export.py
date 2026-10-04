"""Historical exports preserve source evidence and do not invent scheduler facts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from singular_sensitivity import tower_export as export


class TowerExportTests(unittest.TestCase):
    def setUp(self):
        original = Path(os.environ.get("SSMO_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
        temporary = original / ".cache" / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="tower-export-", dir=temporary)
        self.root = Path(self.tmp.name)
        (self.root / "runs").mkdir()
        self.env = mock.patch.dict(os.environ, {"SSMO_PROJECT_ROOT": str(self.root)}, clear=False)
        self.env.start()
        self.legacy = os.environ.pop("SSMO_ROOT", None)
        self.slurm = os.environ.pop("SLURM_JOB_ID", None)

    def tearDown(self):
        # Read-only source fixtures are disposable test copies, never originals.
        for path in self.root.rglob("*"):
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
        self.env.stop()
        if self.legacy is not None:
            os.environ["SSMO_ROOT"] = self.legacy
        if self.slurm is not None:
            os.environ["SLURM_JOB_ID"] = self.slurm
        self.tmp.cleanup()

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def fixture(self, stage="train", jobs=("123",), recorded_job="123"):
        run = self.root / "runs" / "ssmo-original"
        run.mkdir()
        (run / "logs").mkdir()
        (run / "jobs.tsv").write_text("stage\tmethod\tseed\tjob_id\tdependency\n" +
                                     "".join(f"{stage}\tmeasure\t17\t{job}\tnone\n" for job in jobs))
        artifact = export.artifact_directory(run, {"stage": stage, "method": "measure", "seed": "17"})
        artifact.mkdir(parents=True, exist_ok=True)
        provenance = {"command": stage, "elapsed_seconds": 41.5, "manifest_sha256": "f" * 64,
                      "source": {"source_sha256": "c" * 64}, "slurm": {"SLURM_JOB_ID": recorded_job}}
        self.put(artifact / "provenance.json", provenance)
        if stage == "train":
            self.put(artifact / "training_summary.json", {"status": "complete", "method": "measure", "seed": 17,
                       "steps_completed": 450, "best_step": 300, "best_validation_score": 0.003,
                       "elapsed_seconds": 40.0, "validation": {"weak_mse": 0.003},
                       "host_peak_rss_bytes": 999, "peak_reserved_bytes": 1024})
            (artifact / "training.jsonl").write_text('{"step":25,"elapsed_seconds":2.0,"loss":0.4}\n')
        return run, artifact

    def imports(self, output="runs/ssmo-export-one", **kwargs):
        result = export.import_runs(self.root, ["ssmo-original"], [], output, **kwargs)
        return [self.root / row["path"] for row in result["attempts"]]

    def load(self, path):
        return json.loads(path.read_text())

    def test_readonly_training_import_uses_current_snapshot_and_recorded_runtime(self):
        run, artifact = self.fixture()
        before = {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in run.rglob("*") if path.is_file()}
        for path in run.rglob("*"):
            path.chmod(0o500 if path.is_dir() else 0o400)
        run.chmod(0o500)
        started = time.time()
        attempt = self.imports()[0]
        summary = self.load(attempt / "summary.json")
        self.assertEqual(summary["state"], "COMPLETED")
        self.assertEqual(summary["runtime_seconds"], 41.5)
        for unknown in ("start", "end", "submit", "gpus", "cpus", "memory_bytes", "cpu_seconds"):
            self.assertNotIn(unknown, summary)
        lines = [json.loads(line) for line in (attempt / "metrics.jsonl").read_text().splitlines()]
        self.assertTrue(lines)
        self.assertTrue(all(row["phase"] == "imported_summary" and row["t"] >= started for row in lines))
        self.assertEqual(lines[0]["metrics"]["steps_completed"], 450)
        after = {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in run.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        for entry in self.load(attempt / "logs.json")["logs"]:
            self.assertFalse(Path(entry["path"]).is_absolute())

    def test_accuracy_failures_are_completed_applications_with_all_method_cohorts(self):
        run, artifact = self.fixture("evaluate")
        cohorts = [{"method": method, "split": split, "family": family, "physical_parents": 2,
                    "accuracy_gate_parents_passed": 1, "accuracy_gate_parents_failed": 1,
                    "invalid_chart_rows": 0, "metrics": {"absolute_error_max": {"worst": 0.02}}}
                   for method in ("measure_chart", "exact_front", "classical_front_regression", "direct_grid_sensitivity", "learned_grid_state_autodiff")
                   for split, family in (("audit", "shock"), ("test", "constant"), ("test", "shock"), ("ood", "shock"))]
        self.put(artifact / "evaluation.json", {"status": "completed_scoped_pilot", "physical_parents": 8,
                 "parent_aggregates": cohorts, "accuracy_gate": {"absolute_tolerance": 0.01,
                 "parent_method_failures": [{"parent_id": "audit-0", "method": "measure_chart"}]},
                 "inverse": {"measure_chart": {"total_seconds": 8, "exact_gradient_fallbacks": 1}},
                 "direction_and_chart_consistency": [{"method": "measure_chart", "additivity_error": 0.001}]})
        summary = self.load(self.imports()[0] / "summary.json")
        self.assertEqual(summary["state"], "COMPLETED")
        result = summary["results"]["evaluate"]
        self.assertEqual(len(result["method_cohorts"]), 5)
        self.assertEqual(result["method_cohorts"]["measure_chart"]["ood/shock"]["accuracy_gate_parents_failed"], 1)
        self.assertEqual(result["inverse_audits"]["measure_chart"]["exact_gradient_fallbacks"], 1)

    def test_missing_app_artifact_is_unknown_despite_scheduler_log_text(self):
        run, artifact = self.fixture("validate")
        (artifact / "provenance.json").unlink()
        (run / "logs" / "validate-measure-123.err").write_text("srun task failed: lock unavailable")
        summary = self.load(self.imports()[0] / "summary.json")
        self.assertEqual(summary["state"], "UNKNOWN")
        self.assertNotIn("exit_code", summary)
        self.assertNotIn("runtime_seconds", summary)

    def test_retry_artifact_is_attached_only_to_its_recorded_actual_job(self):
        self.fixture(jobs=("122", "123"), recorded_job="123")
        attempts = self.imports()
        summaries = [self.load(path / "summary.json") for path in attempts]
        self.assertEqual([row["job_id"] for row in summaries], ["122", "123"])
        self.assertEqual([row["state"] for row in summaries], ["UNKNOWN", "COMPLETED"])
        self.assertEqual(summaries[0]["results"], {})

    def test_retry_without_job_provenance_is_not_invented_as_two_successes(self):
        self.fixture(jobs=("122", "123"), recorded_job=None)
        self.assertEqual([self.load(path / "summary.json")["state"] for path in self.imports()], ["UNKNOWN", "UNKNOWN"])

    def test_accounting_preserves_timeout_resources_and_maximum_task_rss(self):
        self.fixture()
        raw = "# ssmo-sacct-timezone=UTC\nJobIDRaw|State|ExitCode|ElapsedRaw|TotalCPU|AllocCPUS|NNodes|AllocTRES|ReqMem|TimelimitRaw|Partition|Account|QOS|Submit|Start|End|MaxRSS\n" + \
              "123|TIMEOUT|0:9|600|00:03:20|2|1|cpu=2,mem=8G,gres/gpu=1,gres/gpu:a100=1|8Gn|10|gpu|anakano_81|normal|2026-10-03T00:00:00|2026-10-03T00:01:00|2026-10-03T00:11:00|\n" + \
              "123.batch|TIMEOUT|0:9||||||||||||||2097152K\n" + \
              "123.extern|COMPLETED|0:0||||||||||||||3145728K\n"
        accounting = self.root / "runs" / "accounting.txt"
        accounting.write_text(raw)
        summary = self.load(self.imports(accounting_file=accounting)[0] / "summary.json")
        self.assertEqual(summary["state"], "TIMEOUT")
        self.assertNotIn("exit_code", summary)
        self.assertEqual(summary["gpus"], 1)
        self.assertEqual(summary["cpus"], 2)
        self.assertEqual(summary["mem_bytes"], 8 * 1024 ** 3)
        self.assertEqual(summary["memory_bytes"], 3 * 1024 ** 3)
        self.assertEqual(summary["memory_scope"], "max_task_rss")
        self.assertEqual(summary["runtime_seconds"], 600)
        self.assertEqual(summary["cpu_seconds"], 200)
        self.assertEqual(summary["end"] - summary["start"], 600)

    def test_accounting_omits_unknown_timezones_and_distinguishes_cpu_gpu_zero(self):
        result = export.parse_accounting("JobIDRaw|State|AllocTRES|Start\n123|COMPLETED|cpu=2,mem=4G|2026-10-03T01:00:00\n124|PENDING||Unknown\n")
        self.assertEqual(result["123"]["fields"]["gpus"], 0)
        self.assertNotIn("start", result["123"]["fields"])
        self.assertNotIn("gpus", result["124"]["fields"])
        self.assertEqual(result["124"]["state"], "PENDING")

    def test_compressed_evidence_remains_reference_and_is_not_a_text_log(self):
        run, artifact = self.fixture("evaluate")
        self.put(artifact / "evaluation.json", {"status": "completed_scoped_pilot", "parent_aggregates": []})
        compressed = artifact / "query_errors.jsonl.gz"
        compressed.write_bytes(b"not decompressed or read by this exporter")
        attempt = self.imports()[0]
        self.assertTrue(any(row["path"].endswith("query_errors.jsonl.gz") for row in self.load(attempt / "outputs/analytics.json")["artifact_references"]))
        self.assertFalse(any(row["path"].endswith(".gz") for row in self.load(attempt / "logs.json")["logs"]))
        self.assertEqual(compressed.read_bytes(), b"not decompressed or read by this exporter")

    def test_repeated_imports_are_one_planning_attempt_and_overwrite_is_explicit(self):
        self.fixture()
        self.imports("runs/ssmo-export-one")
        self.imports("runs/ssmo-export-two")
        bundle = export.planning(self.root, [], ["runs/ssmo-export-one", "runs/ssmo-export-two"], "reports/planning-one.json")
        self.assertEqual(len(bundle["history"]), 1)
        self.assertNotIn("scaling", bundle)
        self.assertNotIn("observations", bundle)
        with self.assertRaises(FileExistsError):
            export.planning(self.root, [], ["runs/ssmo-export-one"], "reports/planning-one.json")
        export.planning(self.root, [], ["runs/ssmo-export-one"], "reports/planning-one.json", replace=True)

    def test_planning_reference_is_explicit_and_failures_are_retained(self):
        run, artifact = self.fixture()
        self.put(artifact / "failure.json", {"status": "failed", "reason": "controlled", "command": "train"})
        attempt = self.imports()[0]
        bundle = export.planning(self.root, [attempt], [], "reports/planning.json", reference=attempt)
        self.assertEqual(bundle["history"][0]["state"], "FAILED")
        self.assertEqual(bundle["query"]["name"], bundle["history"][0]["name"])
        self.assertNotIn("cpus", bundle["query"])
        with self.assertRaises(ValueError):
            export.planning(self.root, [attempt], [], "reports/planning-other.json", reference=run)

    def test_review_cost_nulls_remain_unavailable(self):
        result = export.compact_result("imported-review", {"runs": [{"run_id": "pilot17", "seed": 17,
                 "methods": {"measure": {"endpoint_costs": {"comparisons": [{"control": "exact_front",
                 "status": "unavailable", "workloads": [{"directions": 3, "queries": 6,
                 "warm": {"status": "unavailable", "learned_over_control": None},
                 "cold": {"status": "unavailable", "learned_over_control": None}}]}]}}}}]})
        comparison = result["method_records"]["pilot17.measure"]["endpoint_comparisons"]["exact_front"]
        self.assertEqual(comparison["status"], "unavailable")
        workload = result["endpoint_records"]["pilot17.measure.exact_front.directions3-queries6"]
        self.assertEqual(workload["warm_status"], "unavailable")
        self.assertNotIn("warm_learned_over_control_median", workload)

    def test_positive_cost_review_preserves_ratios_seed_identity_and_source_receipt(self):
        review = self.root / "runs" / "ssmo-cost-review"
        review.mkdir()
        workload = {"directions": 3, "queries": 6, "matched_physical_parents": 62,
                    "warm": {"status": "available", "learned_over_control": {"median": 16.5, "p90": 18}},
                    "cold": {"status": "available", "learned_over_control": {"median": 20}}}
        self.put(review / "summary.json", {"runs": [{"run_id": "pilot17", "seed": 17,
                  "methods": {"measure": {"total": {"physical_parents": 62, "passed": 61},
                  "endpoint_costs": {"comparisons": [{"control": "exact_front", "status": "available",
                  "workloads": [workload]}]}}}}, {"run_id": "pilot29", "seed": 29,
                  "methods": {"measure": {"total": {"physical_parents": 62, "passed": 55}}}}]})
        (review / "SHA256SUMS").write_text("retained original receipt\n")
        result = export.import_runs(self.root, [], [review], "runs/ssmo-review-export")
        attempt = self.root / result["attempts"][0]["path"]
        imported = self.load(attempt / "summary.json")["results"]["imported-review"]
        data = imported["method_records"]
        self.assertEqual(data["pilot17.measure"]["seed"], 17)
        self.assertEqual(data["pilot29.measure"]["seed"], 29)
        self.assertEqual(imported["endpoint_records"]["pilot17.measure.exact_front.directions3-queries6"]["warm_learned_over_control_median"], 16.5)
        receipts = self.load(self.root / "runs/ssmo-review-export/sources.json")["inputs"]
        self.assertEqual(receipts[str((review / "summary.json").relative_to(self.root))]["sha256"], hashlib.sha256((review / "summary.json").read_bytes()).hexdigest())
        logs = self.load(attempt / "logs.json")["logs"]
        self.assertEqual(sum(entry["path"].endswith("SHA256SUMS") for entry in logs), 1)

    def test_endpoint_protocols_are_separate_and_duplicate_parent_records_rejected(self):
        base = {"parent_id": "parent0", "method": "measure_chart", "directions": 3, "queries": 6,
                "endpoint": "complete", "measurement": {"warmup": 1, "repeats": 3,
                "cuda_synchronized": True, "median_seconds": 0.01, "cold_call_seconds": 0.02}}
        second = json.loads(json.dumps(base))
        second["measurement"]["warmup"] = 2
        compact = export.compact_result("evaluate", {"timings": [base, second]})
        self.assertEqual(len(compact["endpoint_measurements"]["measure_chart"]), 2)
        with self.assertRaises(ValueError):
            export.compact_result("evaluate", {"timings": [base, base]})

    def test_explicit_stage_artifact_directory_imports_existing_inverse_without_execution(self):
        directory = self.root / "runs" / "ssmo-local-inverse"
        directory.mkdir()
        self.put(directory / "inverse.json", {"status": "completed", "fixed_steps": 2,
                  "accepted_steps": 2, "exact_gradient_fallbacks": 0,
                  "initial_trusted_objective": 1, "final_trusted_objective": 0.25,
                  "total_seconds": 0.2, "trusted_gradient_reference_evaluations": 2})
        self.put(directory / "provenance.json", {"command": "inverse", "elapsed_seconds": 0.3,
                  "slurm": {"SLURM_JOB_ID": None}})
        before = (directory / "inverse.json").read_bytes()
        result = export.import_runs(self.root, [], [directory], "runs/ssmo-inverse-export")
        attempt = self.root / result["attempts"][0]["path"]
        summary = self.load(attempt / "summary.json")
        self.assertEqual(summary["state"], "COMPLETED")
        self.assertEqual(summary["runtime_seconds"], 0.3)
        self.assertEqual(summary["results"]["inverse"]["exact_gradient_fallbacks"], 0)
        self.assertNotIn("job_id", summary)
        self.assertEqual((directory / "inverse.json").read_bytes(), before)

    def test_source_symlinks_duplicate_json_nonfinite_and_budgets_are_rejected(self):
        self.fixture()
        linked = self.root / "runs" / "ssmo-link"
        linked.symlink_to(self.root / "runs" / "ssmo-original", target_is_directory=True)
        with self.assertRaises(ValueError):
            export.import_runs(self.root, ["ssmo-link"], [], "runs/ssmo-no-export")
        path = self.root / "runs" / "duplicate.json"
        path.write_text('{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            export.read_json(self.root, path)
        with self.assertRaises(ValueError):
            export.encoded({"bad": float("nan")})
        with self.assertRaises(ValueError):
            export.encoded({"large": "x" * (1 << 20)})
        deep = {}
        for _ in range(34):
            deep = {"nested": deep}
        with self.assertRaises(ValueError):
            export.encoded(deep)


if __name__ == "__main__":
    unittest.main()
