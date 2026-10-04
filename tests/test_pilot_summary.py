"""Frozen replication reviews reject changed evidence and preserve source runs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import shlex


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ssmo_pilot_summary", REPOSITORY / "scripts/summarize_pilots.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class PilotSummaryTests(unittest.TestCase):
    def setUp(self):
        cache = Path(os.environ.get("SSMO_PROJECT_ROOT", REPOSITORY)).resolve() / ".cache/tmp"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ssmo-pilot-review-test-", dir=cache)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "runs").mkdir()
        self.carc_root = patch.object(review, "CARC_ROOT", self.root)
        self.carc_root.start()
        self.environment = patch.dict(os.environ, {"SSMO_PROJECT_ROOT": str(self.root)}, clear=False)
        self.environment.start()
        self.old_legacy = os.environ.pop("SSMO_ROOT", None)
        self.old_slurm = os.environ.pop("SLURM_JOB_ID", None)
        self.gpu = {"status": "passed", "device": "cuda:0", "cuda_executed": True,
                    "kernel_checks": ["forward", "jvp", "mixed_derivative_backward"],
                    "model": "NVIDIA A100-SXM4-40GB", "total_memory_bytes": 40 * 2**30,
                    "capability": [8, 0], "torch_version": "2.10.0+cu126", "cuda_runtime": "12.6", "visible_device_uuid": "GPU-one"}
        self.manifest = {"schema_version": 1, "domain": [-2, 2], "parents": [
            {"parent_id": "test-one", "split": "test", "family": "shock"},
            {"parent_id": "audit-shock", "split": "audit", "family": "shock"}]}

    def tearDown(self):
        if self.old_legacy is not None:
            os.environ["SSMO_ROOT"] = self.old_legacy
        if self.old_slurm is not None:
            os.environ["SLURM_JOB_ID"] = self.old_slurm
        self.environment.stop()
        self.carc_root.stop()
        self.temporary.cleanup()

    def copied_helper(self):
        # Only this disposable copy uses the fixture as CARC_ROOT. Production
        # has no test bypass; allocated CARC validation can exercise the helper.
        helper = self.root / "scripts/summarize_pilots.py"
        helper.parent.mkdir(exist_ok=True)
        helper.write_text((REPOSITORY / "scripts/summarize_pilots.py").read_text().replace(
            'CARC_ROOT = Path("/home1/aadaniel/projects/SSMO")', f"CARC_ROOT = Path({str(self.root)!r})"))
        return helper

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def cohort(self, method, split, passed, error):
        return {"method": method, "split": split, "family": "shock", "physical_parents": 1,
                "accuracy_gate_parents_passed": int(passed), "accuracy_gate_parents_failed": int(not passed),
                "invalid_chart_rows": 0,
                "metrics": {metric: {"mean": error, "median": error, "q90": error, "worst": error, "physical_parents": 1}
                            for metric in review.ERRORS}}

    def make_run(self, seed, run_id=None):
        run_id = run_id or f"ssmo-pilot-seed{seed}"
        run = self.root / "runs" / run_id
        run.mkdir()
        (run / "config.yaml").write_text("schema_version: 1\ntraining: {seeds: [17, 29, 43]}\n")
        (run / "submission-manifest.txt").write_text(f"run_id={run_id}\nssmo_project_root={self.root}\npipeline=pilot\nseed={seed}\nprofile=a10040\n")
        (run / "jobs.tsv").write_text("stage\tmethod\tseed\tjob_id\tdependency\n" + "".join(
            f"{stage}\t{method}\t{seed}\t{100 + index}\tnone\n"
            for index, (stage, method) in enumerate((stage, method) for method in review.METHODS for stage in ("train", "evaluate"))))
        self.write(run / "artifacts/data/parents.json", self.manifest)
        (run / "source/singular_sensitivity").mkdir(parents=True)
        (run / "source/singular_sensitivity/reference.py").write_text("FORMULA_VERSION = 1\n")
        (run / "source/requirements").mkdir()
        (run / "source/requirements/base.txt").write_text("numpy==2.2.6\n")
        (run / "source/pyproject.toml").write_text('[project]\nname = "SSMO"\n')
        (run / "artifacts/dependency-freeze.txt").write_text("numpy==2.2.6\ntorch==2.10.0+cu126\n")
        (run / "source-revision.txt").write_text(f"revision-for-seed-{seed}\n")
        self.write(run / "artifacts/gpu-smoke/gpu_audit.json", self.gpu)
        manifest_hash = review.content_hash(self.manifest)
        for method in review.METHODS:
            train_dir = run / f"artifacts/train-{method}-seed{seed}"
            eval_dir = run / f"artifacts/evaluate-{method}-seed{seed}"
            train_dir.mkdir()
            checkpoint = train_dir / "best.pt"
            checkpoint.write_bytes(b"not loaded by the review")
            config_hash = review.content_hash({"method": method, "seed": seed})
            training = {"status": "complete", "method": method, "seed": seed,
                        "stopped_reason": "validation_patience", "steps_completed": 200 + seed, "best_step": 50,
                        "best_validation_score": 0.001, "elapsed_seconds": 40.0 if method == "measure" else 10.0,
                        "peak_reserved_bytes": 64 * 2**20, "dataset_hash": manifest_hash,
                        "config_hash": config_hash, "best_checkpoint": str(checkpoint),
                        "hardware": {"gpu_name": self.gpu["model"], "gpu_total_bytes": self.gpu["total_memory_bytes"],
                                     "torch_version": self.gpu["torch_version"]},
                        "query_protocol": {"checkpoint_score": "validation only"}, "direction_convention": "fixed physical x"}
            self.write(train_dir / "training_summary.json", training)
            self.write(train_dir / "provenance.json", {"manifest_sha256": manifest_hash, "config_sha256": config_hash})
            self.write(train_dir / "hardware.json", self.gpu | {"visible_device_uuid": f"train-{method}-{seed}"})
            self.write(eval_dir / "hardware.json", self.gpu | {"visible_device_uuid": f"evaluate-{method}-{seed}"})
            learned = method + "_chart"
            evaluation = {
                "status": "completed_scoped_pilot", "domain": [-2, 2], "selected_parent_ids": ["audit-shock", "test-one"],
                "held_out_query_bank": [{"kind": "sin", "frequency": 1.7}],
                "checkpoints": {method: {"path": str(checkpoint), "dataset_hash": manifest_hash, "step": 50, "best_step": 50}},
                "parent_aggregates": [self.cohort(learned, "test", True, 0.001), self.cohort(learned, "audit", method == "state_only", 0.0081 if method == "state_only" else 0.0118),
                                      self.cohort(review.CLASSICAL, "test", True, 1e-14), self.cohort(review.CLASSICAL, "audit", True, 1e-14)],
                "fixed_audit_coverage": {"available": 3, "evaluated": 3, "complete": True},
                "unresolved_records": [{"status": "unresolved", "family": "collision", "reason": "exact event derivative deliberately unresolved"}],
                "direction_and_chart_consistency": [{"method": learned, "zero_direction_max_error": 0.0, "additivity_max_error": 1e-9,
                                                     "scaling_max_error": 1e-10, "chart_finite_variation_max_error": 1e-6}],
                "timings": [{"method": learned, "parent_id": "test-one", "directions": 3, "queries": 8,
                             "endpoint": "complete chart and weak queries", "measurement": {"median_seconds": 0.002}}],
                "accuracy_gate": {"absolute_tolerance": 0.01, "parent_method_failures": []},
            }
            self.write(eval_dir / "evaluation.json", evaluation)
        return run_id, run

    def mutate(self, path, change):
        data = json.loads(path.read_text())
        change(data)
        self.write(path, data)

    def test_three_seeds_derive_paths_keep_counts_separate_and_classical_once(self):
        ids = [self.make_run(seed, "ssmo-pilot-001" if seed == 17 else None)[0] for seed in (17, 29, 43)]
        before = {path: path.read_bytes() for path in (self.root / "runs").rglob("*") if path.is_file()}
        output = self.root / "runs/review"
        summary, text = review.summarize(self.root, ids, output)
        self.assertEqual([row["seed"] for row in summary["runs"]], [17, 29, 43])
        self.assertIsNone(summary["pooled_parent_count"])
        for row in summary["runs"]:
            self.assertEqual(row["classical"]["total"]["physical_parents"], 2)
            self.assertEqual(row["methods"]["measure"]["total"]["passed"], 1)
            self.assertEqual(row["methods"]["state_only"]["total"]["passed"], 2)
            self.assertEqual(row["methods"]["measure"]["training"]["peak_reserved_mib"], 64)
            self.assertEqual(row["methods"]["measure"]["cohorts"][0]["worst_errors"]["absolute_error_max"], 0.0118)
        self.assertIn("no pooled sample size", text)
        self.assertIn("weak=0.0118", text)
        self.assertIn("chart_finite_variation_max_error", text)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        receipt = json.loads((output / "provenance.json").read_text())
        self.assertIn("runs/ssmo-pilot-seed43/artifacts/train-measure-seed43/training_summary.json", receipt["inputs"])
        self.assertEqual(receipt["outputs"]["summary.txt"], hashlib.sha256((output / "summary.txt").read_bytes()).hexdigest())
        self.assertEqual(set(path.name for path in output.iterdir()), {"summary.json", "summary.txt", "provenance.json"})

    def test_changed_frozen_config_or_manifest_refuses_before_creating_output(self):
        first, _ = self.make_run(17)
        second, run = self.make_run(29)
        config = run / "config.yaml"
        original = config.read_bytes()
        config.write_bytes(original + b"precision: bf16\n")
        output = self.root / "runs/refused"
        with self.assertRaisesRegex(ValueError, "frozen scientific config differs"):
            review.summarize(self.root, [first, second], output)
        self.assertFalse(output.exists())
        config.write_bytes(original)
        self.mutate(run / "artifacts/data/parents.json", lambda data: data["parents"][0].update(parent_id="changed"))
        with self.assertRaisesRegex(ValueError, "training dataset hash mismatch"):
            review.summarize(self.root, [first, second], output)
        self.assertFalse(output.exists())
        preserved = self.root / "runs/preserved"
        preserved.mkdir()
        (preserved / "summary.json").write_text("existing evidence")
        with self.assertRaisesRegex(ValueError, "already exists"):
            review.summarize(self.root, [first, second], preserved)
        self.assertEqual((preserved / "summary.json").read_text(), "existing evidence")

    def test_duplicate_classical_values_and_swapped_checkpoint_are_rejected(self):
        run_id, run = self.make_run(17)
        path = run / "artifacts/evaluate-state_only-seed17/evaluation.json"
        original = path.read_bytes()
        self.mutate(path, lambda data: data["parent_aggregates"][-1]["metrics"]["absolute_error_max"].update(worst=0.02))
        with self.assertRaisesRegex(ValueError, "duplicated classical control differs"):
            review.summarize(self.root, [run_id], "runs/review")
        path.write_bytes(original)
        self.mutate(path, lambda data: data["checkpoints"]["state_only"].update(step=100))
        with self.assertRaisesRegex(ValueError, "selected checkpoint evidence mismatch"):
            review.summarize(self.root, [run_id], "runs/review")
        self.assertFalse((self.root / "runs/review").exists())

    def test_scientific_source_change_rejected_but_docs_and_git_revisions_may_differ(self):
        first, _ = self.make_run(17)
        second, run = self.make_run(29)
        (run / "source/README.md").write_text("Different documentation has no scientific effect.\n")
        review.summarize(self.root, [first, second], "runs/docs-only-review")
        (run / "source/singular_sensitivity/reference.py").write_text("FORMULA_VERSION = 2\n")
        with self.assertRaisesRegex(ValueError, "frozen scientific source differs"):
            review.summarize(self.root, [first, second], "runs/science-review")
        self.assertFalse((self.root / "runs/science-review").exists())

    def test_cpu_fallback_missing_kernels_and_hardware_change_are_rejected(self):
        run_id, run = self.make_run(17)
        audit = run / "artifacts/gpu-smoke/gpu_audit.json"
        original = audit.read_bytes()
        self.mutate(audit, lambda data: data.update(cuda_executed=False))
        with self.assertRaisesRegex(ValueError, "required CUDA kernels"):
            review.summarize(self.root, [run_id], "runs/refused")
        audit.write_bytes(original)
        self.mutate(run / "artifacts/train-measure-seed17/hardware.json", lambda data: data.update(model="NVIDIA A40"))
        with self.assertRaisesRegex(ValueError, "GPU/software stratum changed"):
            review.summarize(self.root, [run_id], "runs/refused")
        self.assertFalse((self.root / "runs/refused").exists())

    def test_outside_inputs_outputs_and_root_aliases_are_rejected(self):
        run_id, run = self.make_run(17)
        with self.assertRaisesRegex(ValueError, "escapes SSMO"):
            review.summarize(self.root, [run_id], "/tmp/ssmo-must-not-write")
        alias = self.root / "escape"
        alias.symlink_to("/tmp", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes SSMO"):
            review.summarize(self.root, [run_id], alias / "review")
        path = run / "artifacts/data/parents.json"
        path.unlink()
        path.symlink_to("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "escapes SSMO"):
            review.summarize(self.root, [run_id], "runs/review")
        root_alias = self.root / "root-alias"
        root_alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "canonical path"):
            review.project_root(root_alias)
        self.assertFalse((self.root / "runs/review").exists())

    def test_method_seed_and_parent_cohort_mismatch_are_rejected(self):
        run_id, run = self.make_run(29)
        train = run / "artifacts/train-measure-seed29/training_summary.json"
        original = train.read_bytes()
        self.mutate(train, lambda data: data.update(seed=17))
        with self.assertRaisesRegex(ValueError, "method/seed mismatch"):
            review.summarize(self.root, [run_id], "runs/review")
        train.write_bytes(original)
        self.mutate(run / "artifacts/evaluate-state_only-seed29/evaluation.json", lambda data: data.update(selected_parent_ids=["different"]))
        with self.assertRaisesRegex(ValueError, "protocols differ"):
            review.summarize(self.root, [run_id], "runs/review")

    def test_standalone_cli_uses_no_site_packages_and_reports_saved_destination(self):
        run_id, _ = self.make_run(43)
        result = subprocess.run([sys.executable, "-S", str(self.copied_helper()),
                                 "--project-root", str(self.root), "--run", run_id, "--output-dir", "runs/cli-review"],
                                capture_output=True, text=True, env=os.environ.copy(), timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("seed=43", result.stdout)
        self.assertIn(f"Saved review: {self.root / 'runs/cli-review'}", result.stdout)
        self.assertTrue((self.root / "runs/cli-review/summary.json").is_file())

    def test_bash_wrapper_defaults_review_all_three_seeds_in_allocated_environment(self):
        for seed in (17, 29, 43):
            self.make_run(seed, "ssmo-pilot-001" if seed == 17 else None)
        self.copied_helper()
        scripts = self.root / "scripts"
        (scripts / "summarize_pilots.sh").write_text((REPOSITORY / "scripts/summarize_pilots.sh").read_text().replace(
            "/home1/aadaniel/projects/SSMO", str(self.root)))
        copied_env = (REPOSITORY / "scripts/carc_env.sh").read_text().replace("/home1/aadaniel/projects/SSMO", str(self.root))
        # Python-module/venv installation is already independently tested; here
        # use the current interpreter to check wrapper argument integration.
        copied_env += f"\nssmo_load_python() {{ :; }}\nssmo_check_venv() {{ export SSMO_PYTHON={shlex.quote(sys.executable)}; }}\n"
        (scripts / "carc_env.sh").write_text(copied_env)
        environment = os.environ.copy() | {"SLURM_JOB_ID": "12345"}
        result = subprocess.run(["bash", str(scripts / "summarize_pilots.sh"), "--output-dir", "runs/wrapper-review"],
                                capture_output=True, text=True, env=environment, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.root / "runs/wrapper-review/summary.json").read_text())
        self.assertEqual([run["seed"] for run in summary["runs"]], [17, 29, 43])
        self.assertIn("Saved review:", result.stdout)


if __name__ == "__main__":
    unittest.main()
