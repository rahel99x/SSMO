"""Project-side Tower reports preserve identity, scope and incomplete evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from singular_sensitivity import tower_reporting as reporting


REPOSITORY = Path(__file__).resolve().parents[1]


class TowerReportingTests(unittest.TestCase):
    def setUp(self):
        cache = Path(os.environ.get("SSMO_PROJECT_ROOT", REPOSITORY)) / ".cache/tmp"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ssmo-tower-test-", dir=cache)
        self.root = Path(self.temporary.name)
        self.pipeline = self.root / "runs/pilot-001"
        self.pipeline.mkdir(parents=True)
        self.config = self.root / "configs/pilot.yaml"
        self.config.parent.mkdir()
        self.config.write_text("training: frozen\n")
        self.source = self.root / "source"
        (self.source / "slurm").mkdir(parents=True)
        (self.source / "slurm/stage.sbatch").write_text("#!/bin/bash\nexit 0\n")
        self.environment = patch.dict(os.environ, {"SSMO_PROJECT_ROOT": str(self.root), "SSMO_ROOT": "",
            "SSMO_TOWER_RUN_DIR": "", "SLURM_JOB_ID": "", "SLURM_ARRAY_JOB_ID": "",
            "SLURM_ARRAY_TASK_ID": "", "SSMO_TOWER_GPU_TRACE_STATUS": ""})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def attempt(self, name="train-measure-seed17", **extra):
        path = self.pipeline / "tower" / name
        reporting.start_attempt(path, pipeline_dir=self.pipeline, stage="train", method="measure", seed=17,
            config_path=self.config, source_dir=self.source, **extra)
        return path

    def load(self, path):
        return json.loads(path.read_text())

    def test_successful_application_retains_scientific_failure_and_process_scope(self):
        path = self.attempt(resources={"cpus": 2, "nodes": 1, "gpus": 1, "gpu_type": "a100",
            "mem_bytes": 8 * 2**30, "time_seconds": 1800}, job_id="12345")
        artifacts = self.pipeline / "artifacts/train-measure-seed17"
        artifacts.mkdir(parents=True)
        (artifacts / "training_summary.json").write_text('{"status":"completed"}\n')
        result = {"status": "completed_scoped_pilot", "accuracy_gate": {"passed": False},
                  "host_peak_rss_bytes": 900, "peak_reserved_bytes": 50000}
        with patch.object(reporting.time, "process_time", return_value=2.5):
            reporting.report_phase("evaluate", result, artifacts, {"seed": 17}, {"parents": [{"parent_id": "p1"}]}, 3.0,
                {"source": {"source_sha256": "a" * 64}, "process_cpu_seconds": 2.5} , attempt_dir=path)
        summary = reporting.finish_attempt(path, 0)
        self.assertEqual(summary["state"], "COMPLETED")
        self.assertFalse(summary["results"]["evaluate"]["accuracy_gate"]["passed"])
        self.assertEqual(summary["cpu_seconds"], 2.5)
        self.assertEqual(summary["memory_scope"], "max_task_rss")
        self.assertNotEqual(summary["memory_bytes"], 50000)
        self.assertEqual(summary["mem_bytes"], 8 * 2**30)
        self.assertIn("excludes children", summary["metadata"]["application_observations"]["cpu_scope"])
        self.assertEqual(summary["input_size"], 1)
        self.assertEqual(summary["job_id"], "12345")
        self.assertEqual(self.load(path / "run.json")["state"], "COMPLETED")
        self.assertEqual(self.load(path / "outputs/analytics.json")["results"]["evaluate"]["status"], "completed_scoped_pilot")

    def test_distinct_command_cpu_windows_sum_but_process_rss_peaks_do_not(self):
        path = self.attempt()
        artifacts = self.pipeline / "artifacts/windows"
        artifacts.mkdir(parents=True)
        for command, cpu, rss in (("representation", 2.0, 10000), ("numerical", 3.0, 15000)):
            reporting.report_phase(command, {"status": "completed"}, artifacts,
                provenance_record={"process_cpu_seconds": cpu, "memory": {"host_peak_rss_bytes": rss}}, attempt_dir=path)
        summary = reporting.finish_attempt(path, 0)
        self.assertEqual(summary["cpu_seconds"], 5.0)
        self.assertEqual(summary["memory_bytes"], 15000)
        self.assertEqual(summary["memory_scope"], "max_task_rss")

    def test_in_memory_tuple_query_protocol_normalizes_only_reporting_copy(self):
        path = self.attempt()
        artifacts = self.pipeline / "artifacts/train-with-tuples"
        artifacts.mkdir(parents=True)
        protocol = {"domain": (-2.0, 2.0), "queries": [{"kind": "bump", "domain": (-2.0, 2.0)}],
                    "hardware": {"capability": (8, 0)}}
        result = {"status": "completed", "query_protocol": protocol, "steps_completed": 40,
                  "best_validation_score": 0.01}
        reporting.report_phase("train", result, artifacts, config={"domain": (-2.0, 2.0)}, attempt_dir=path)
        summary = reporting.finish_attempt(path, 0, summary_fields={"parameters": {"domain": (-2.0, 2.0)}})
        self.assertEqual(summary["state"], "COMPLETED")
        saved = self.load(path / "summary.json")
        self.assertEqual(saved["results"]["train"]["query_protocol"]["domain"], [-2.0, 2.0])
        self.assertEqual(saved["parameters"]["domain"], [-2.0, 2.0])
        self.assertIsInstance(protocol["domain"], tuple)
        self.assertIsInstance(protocol["queries"][0]["domain"], tuple)

    def test_multiline_scientific_evidence_valid_but_native_field_controls_refused(self):
        path = self.attempt()
        artifacts = self.pipeline / "artifacts/gpu-audit"
        artifacts.mkdir(parents=True)
        evidence = "NVIDIA A100, GPU-one, 40960 MiB\nNVIDIA A100, GPU-two, 40960 MiB"
        reporting.report_phase("gpu-audit", {"status": "passed", "nvidia_smi": evidence}, artifacts, attempt_dir=path)
        reporting.finish_attempt(path, 0)
        self.assertEqual(self.load(path / "summary.json")["results"]["gpu-audit"]["nvidia_smi"], evidence)
        for label in ("hidden\u200bmetric", "line\nmetric", "bad\x1bmetric"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                reporting.emit_event("fit", {label: 1}, attempt_dir=path)
        with self.assertRaises(ValueError):
            reporting.emit_event("hidden\u200bphase", {}, attempt_dir=path)
        with self.assertRaises(ValueError):
            reporting.register_log(path, "valid.id", self.pipeline / "safe.log", label="hidden\u200blabel")

    def test_failed_paused_and_signalled_attempts_preserved(self):
        for code, state in ((1, "FAILED"), (75, "INTERRUPTED"), (129, "INTERRUPTED"), (130, "INTERRUPTED"), (143, "INTERRUPTED")):
            with self.subTest(code=code):
                path = self.attempt("attempt-" + str(code))
                (path / "logs/application.stderr.log").write_text("Original failure evidence\n")
                summary = reporting.finish_attempt(path, code)
                self.assertEqual(summary["state"], state)
                self.assertEqual(summary["exit_code"], code)
                self.assertEqual((path / "logs/application.stderr.log").read_text(), "Original failure evidence\n")
                self.assertNotIn("memory_bytes", summary)
                self.assertNotIn("cpu_seconds", summary)
                with self.assertRaises(FileExistsError):
                    reporting.finish_attempt(path, 0)

    def test_attempt_reuse_refused_and_retry_workload_name_stable(self):
        first = self.attempt()
        before = (first / "run.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.attempt()
        self.assertEqual(before, (first / "run.json").read_bytes())
        second = self.attempt("retry-train-measure-seed17")
        first_manifest, second_manifest = self.load(first / "run.json"), self.load(second / "run.json")
        self.assertNotEqual(first_manifest["run_id"], second_manifest["run_id"])
        self.assertEqual(first_manifest["name"], second_manifest["name"])
        self.assertNotIn("pilot-001", first_manifest["name"])
        empty = self.pipeline / "tower/precreated-empty"
        empty.mkdir()
        reporting.start_attempt(empty, stage="audit")

    def test_read_only_source_may_be_project_root_but_attempt_may_not(self):
        (self.root / "slurm").mkdir()
        (self.root / "slurm/stage.sbatch").write_text("#!/bin/bash\nexit 0\n")
        (self.root / "singular_sensitivity").mkdir()
        (self.root / "singular_sensitivity/cli.py").write_text("# actual local entrypoint\n")
        path = self.pipeline / "tower/source-at-project-root"
        with patch.dict(os.environ, {"SLURM_JOB_ID": "98765"}), patch("singular_sensitivity.runtime.CARC_PROJECT_ROOT", self.root):
            manifest = reporting.start_attempt(path, stage="report", source_dir=self.root, job_id="",
                metadata={"execution_entrypoint": "singular_sensitivity/cli.py"})
        self.assertEqual(manifest["provenance"]["script"], "singular_sensitivity/cli.py")
        self.assertNotIn("job_id", manifest)
        with self.assertRaises(ValueError):
            reporting.start_attempt(self.root, stage="report")

    def test_unknown_resources_and_historical_measurements_not_invented(self):
        path = self.attempt(imported=True, state="UNKNOWN")
        artifacts = self.pipeline / "artifacts"
        artifacts.mkdir()
        reporting.report_phase("historical", {"status": "failed", "peak_reserved_bytes": 5000000}, artifacts, attempt_dir=path)
        summary = reporting.finish_attempt(path, None, state="UNKNOWN")
        for key in ("gpus", "cpus", "runtime_seconds", "memory_bytes", "cpu_seconds", "start", "end", "exit_code"):
            self.assertNotIn(key, summary)
        self.assertEqual((path / "metrics.jsonl").read_bytes(), b"")
        pending = self.attempt("pending", imported=True, state="PENDING")
        self.assertEqual(reporting.finish_attempt(pending, None, state="PENDING")["state"], "PENDING")

    def test_repeated_historical_exports_keep_source_attempt_identity(self):
        first = self.attempt("historical", imported=True, metadata={"source_identity": "pilot/train/measure/17"})
        second = self.root / "exports/second/historical"
        reporting.start_attempt(second, stage="train", method="measure", seed=17, imported=True,
            config_path=self.config, metadata={"source_identity": "pilot/train/measure/17"})
        self.assertEqual(self.load(first / "run.json")["run_id"], self.load(second / "run.json")["run_id"])

    def test_live_historical_recorded_identities_and_canonical_fallback_match(self):
        config = {"label": "caf\u00e9", "domain": (-2.0, 2.0)}
        dataset = {"parents": [{"parent_id": "same-parent"}]}
        provenance = {"config_sha256": reporting._identity_hash(config), "manifest_sha256": reporting._identity_hash(dataset),
                      "python": "3.12.8", "software": {"torch": "2.10.0+cu126", "numpy": "2.4.3", "missing": None}}
        hardware = {"device": "cuda:0", "gpu_name": "NVIDIA A100-PCIE-40GB", "gpu_total_bytes": 40 * 2**30,
                    "peak_reserved_bytes": 1000}
        artifacts = self.pipeline / "artifacts/identity"
        artifacts.mkdir(parents=True)
        (artifacts / "hardware.json").write_text(json.dumps(hardware))
        parameters = []
        for name, imported in (("live-identity", False), ("imported-identity", True), ("fallback-identity", False)):
            path = self.attempt(name, imported=imported)
            reporting.report_phase("train", {"status": "completed"}, artifacts,
                config=config, dataset=dataset, provenance_record=provenance if name != "fallback-identity" else None, attempt_dir=path)
            parameters.append(reporting.finish_attempt(path, 0)["parameters"])
        for field in ("effective_config_sha256", "dataset_sha256", "hardware_sha256"):
            self.assertEqual(parameters[0][field], parameters[1][field])
            self.assertEqual(parameters[0][field], parameters[2][field])
        self.assertEqual(parameters[0]["software_sha256"], parameters[1]["software_sha256"])
        self.assertNotIn("software_sha256", parameters[2])
        empty = self.pipeline / "artifacts/no-identity"
        empty.mkdir()
        unknown = self.attempt("unknown-identity", imported=True)
        reporting.report_phase("historical", {"status": "completed"}, empty, attempt_dir=unknown)
        unknown_parameters = reporting.finish_attempt(unknown, 0)["parameters"]
        for field in ("effective_config_sha256", "dataset_sha256", "hardware_sha256", "software_sha256"):
            self.assertNotIn(field, unknown_parameters)

    def test_metrics_finite_numeric_progress_and_noop(self):
        self.assertIsNone(reporting.emit_event("disabled", {"bad": object()}))
        path = self.attempt()
        event = reporting.emit_event("train", {"signed_loss": -0.25, "observed_zero": 0}, step=1,
            completed=1, total=10, unit="steps", attempt_dir=path)
        self.assertEqual(event["metrics"]["signed_loss"], -0.25)
        for values in ({"loss": float("nan")}, {"loss": float("inf")}, {"loss": True}, {"loss": None}, {"loss": "4 GiB"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                reporting.emit_event("train", values, attempt_dir=path)
        for kwargs in ({"completed": 2}, {"completed": 11, "total": 10}, {"completed": 0, "total": 0},
                       {"step": -1}, {"step": 1 << 63}, {"unit": "steps"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reporting.emit_event("train", {}, attempt_dir=path, **kwargs)
        stream = (path / "metrics.jsonl").read_bytes()
        self.assertTrue(stream.endswith(b"\n"))
        self.assertEqual(len(stream.splitlines()), 2)

    def test_metric_series_and_phase_order_bounded(self):
        path = self.attempt()
        reporting.emit_event("fit", {f"metric_{n}": n for n in range(64)}, step=2,
            completed=2, total=10, unit="steps", attempt_dir=path)
        before = (path / "metrics.jsonl").read_bytes()
        with self.assertRaises(ValueError):
            reporting.emit_event("fit", {"another_metric": 1}, attempt_dir=path)
        with self.assertRaises(ValueError):
            reporting.emit_event("fit", {}, step=1, attempt_dir=path)
        with self.assertRaises(ValueError):
            reporting.emit_event("fit", {}, completed=1, total=10, unit="steps", attempt_dir=path)
        self.assertEqual(before, (path / "metrics.jsonl").read_bytes())
        reporting.emit_event("new_phase", {}, step=0, completed=0, total=10, unit="steps", attempt_dir=path)

    def test_exact_log_index_and_original_files_never_modified(self):
        scheduler = self.pipeline / "logs/train-12345.err"
        scheduler.parent.mkdir()
        scheduler.write_text("Scheduler error retained\n")
        path = self.attempt(job_id="12345", scheduler_logs={"stderr": scheduler})
        index = self.load(path / "logs.json")
        self.assertEqual(index["job_id"], "12345")
        location = next(row for row in index["logs"] if row["id"] == "scheduler.stderr")
        self.assertEqual((path / location["path"]).resolve(), scheduler)
        extra = self.pipeline / "logs/exact.log"
        extra.write_text("readonly application evidence\n")
        reporting.register_log(path, "science.exact", extra, group="Evidence")
        with self.assertRaises(ValueError):
            reporting.register_log(path, "science.exact", self.pipeline / "logs/new.log")
        with self.assertRaises(ValueError):
            reporting.register_log(path, "another.id", extra)
        self.assertEqual(extra.read_text(), "readonly application evidence\n")
        with self.assertRaises(ValueError):
            self.attempt("duplicate-scheduler", scheduler_logs={"stdout": extra, "stderr": extra})

    def test_known_gpu_trace_index_and_skipped_trace_reason(self):
        with patch.dict(os.environ, {"SSMO_TOWER_GPU_TRACE_STATUS": "eligible", "SSMO_TOWER_GPU_TRACE_REASON": "verified",
                                    "SSMO_TOWER_GPU_TRACE_SELECTOR": "GPU-explicit", "SLURM_JOB_ID": "24680"}), \
             patch("singular_sensitivity.runtime.CARC_PROJECT_ROOT", self.root):
            path = self.attempt("gpu-traced", resources={"gpus": 1})
        index = self.load(path / "logs.json")
        self.assertEqual(next(row["path"] for row in index["logs"] if row["id"] == "gpu.trace"), "logs/gpu-util-24680.csv")
        self.assertEqual(self.load(path / "run.json")["metadata"]["gpu_trace"]["selector"], "GPU-explicit")
        with patch.dict(os.environ, {"SSMO_TOWER_GPU_TRACE_STATUS": "skipped", "SSMO_TOWER_GPU_TRACE_REASON": "missing_physical_selector"}):
            skipped = self.attempt("gpu-skipped", resources={"gpus": 1})
        self.assertFalse(any(row["id"] == "gpu.trace" for row in self.load(skipped / "logs.json")["logs"]))
        self.assertEqual(self.load(skipped / "run.json")["metadata"]["gpu_trace"]["reason"], "missing_physical_selector")

    def test_project_path_escape_symlinks_and_hardlink_aliases_refused(self):
        with self.assertRaises(ValueError):
            reporting.start_attempt(REPOSITORY / "local/outside-fixture", stage="audit")
        with self.assertRaises(ValueError):
            reporting.start_attempt("runs/../escape", stage="audit")
        link = self.root / "alias"
        link.symlink_to(self.pipeline, target_is_directory=True)
        with self.assertRaises(ValueError):
            reporting.start_attempt(link / "new", stage="audit")
        path = self.attempt()
        original = self.pipeline / "hardlink-original.log"
        original.write_text("retained\n")
        alias = self.pipeline / "hardlink-alias.log"
        os.link(original, alias)
        with self.assertRaises(ValueError):
            reporting.register_log(path, "unsafe.alias", alias)

    def test_atomic_json_budget_and_invalid_summary_leave_prior_manifest(self):
        path = self.attempt()
        before = (path / "run.json").read_bytes()
        for fields in ({"memory_bytes": 100}, {"memory_bytes": 100, "memory_scope": "gpu_allocator"},
                       {"runtime_seconds": float("nan")}, {"cpus": 0}, {"parameters": {"nested": {"x": {"y": {"z": {"v": 1}}}}}}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                reporting.finish_attempt(path, 0, summary_fields=fields)
            self.assertFalse((path / "summary.json").exists())
            self.assertEqual(before, (path / "run.json").read_bytes())
        with self.assertRaises(ValueError):
            reporting._atomic(path / "bounded.json", {"large": "x" * 2000}, limit=1024)
        self.assertFalse((path / "bounded.json").exists())
        self.assertEqual(list(path.glob(".*.tmp")), [])

    def test_evaluation_cohorts_cost_protocols_and_compressed_refs_preserved(self):
        path = self.attempt()
        artifacts = self.pipeline / "artifacts/evaluate"
        artifacts.mkdir(parents=True)
        (artifacts / "evaluation.json").write_text('{}\n')
        (artifacts / "query_errors.jsonl.gz").write_bytes(b"compressed evidence untouched")
        cohorts = [{"method": "measure_chart", "split": "audit", "family": "shock", "physical_parents": 1,
            "accuracy_gate_parents_passed": 0, "accuracy_gate_parents_failed": 1, "invalid_chart_rows": 0,
            "metrics": {"absolute_error_max": {"worst": 0.0118}}}]
        timings = []
        for method, sync, seconds in (("measure_chart", True, 0.02), ("exact_front", False, 0.001)):
            timings.append({"method": method, "parent_id": "parent-1", "directions": 3, "queries": 6,
                "endpoint": "initial inputs -> all weak queries", "measurement": {"warmup": 1, "repeats": 3,
                    "cuda_synchronized": sync, "median_seconds": seconds, "cold_call_seconds": seconds * 2}})
        result = {"status": "completed_scoped_pilot", "parent_aggregates": cohorts, "timings": timings,
                  "inverse": {"measure_chart": {"trusted_gradient_reference_evaluations": 24, "history": [{"step": 0}]}}}
        reporting.report_phase("evaluate", result, artifacts, attempt_dir=path)
        analytics = self.load(path / "outputs/analytics.json")
        summarized = analytics["results"]["evaluate"]
        self.assertEqual(summarized["evaluation_cohorts"][0]["accuracy_gate_parents_failed"], 1)
        self.assertEqual(len(summarized["endpoint_workloads"]), 2)
        self.assertNotEqual(summarized["endpoint_workloads"][0]["cuda_synchronized"], summarized["endpoint_workloads"][1]["cuda_synchronized"])
        self.assertIn("exact diagnostic gradients", summarized["inverse_scope"])
        self.assertTrue(any(ref["path"].endswith(".gz") for ref in analytics["artifact_references"]))
        self.assertFalse(any(row["path"].endswith(".gz") for row in self.load(path / "logs.json")["logs"]))

    def test_cli_stdlib_start_and_finish_request_semantics(self):
        attempt = self.pipeline / "tower/cli-attempt"
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-B", "-S", "-m", "singular_sensitivity.tower_reporting"]
        start = subprocess.run(command + ["start", "--attempt-dir", str(attempt), "--pipeline-dir", str(self.pipeline),
            "--stage", "validate", "--request-cpus", "2", "--request-mem-gb", "4", "--request-time", "00:10:00",
            "--request-gpus", "0"], cwd=REPOSITORY, env=environment, capture_output=True, text=True)
        self.assertEqual(start.returncode, 0, start.stderr)
        finish = subprocess.run(command + ["finish", "--attempt-dir", str(attempt), "--exit-code", "0"],
            cwd=REPOSITORY, env=environment, capture_output=True, text=True)
        self.assertEqual(finish.returncode, 0, finish.stderr)
        summary = self.load(attempt / "summary.json")
        self.assertEqual((summary["cpus"], summary["mem_bytes"], summary["time_seconds"], summary["gpus"]), (2, 4 * 2**30, 600, 0))
        self.assertNotIn("memory_bytes", summary)


if __name__ == "__main__":
    unittest.main()
