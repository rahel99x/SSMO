"""Existing-artifact diagnostics preserve evidence and refuse wrong raw shards."""
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from singular_sensitivity.baselines import FrontPrediction
from singular_sensitivity.queries import query_bank
from singular_sensitivity.reference import ParentInput, reference_sensitivity
from singular_sensitivity.validation import front_errors


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))
import summarize_pilots as review
import diagnose_pilot_failures as diagnostic


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


class FailureDiagnosticTests(unittest.TestCase):
    def setUp(self):
        cache = Path(os.environ.get("SSMO_PROJECT_ROOT", REPOSITORY)).resolve() / ".cache/tmp"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ssmo-failure-diagnostic-test-", dir=cache)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "runs").mkdir()
        shutil.copytree(REPOSITORY / "singular_sensitivity", self.root / "singular_sensitivity",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Audit source snapshots are read-only. copytree preserves their modes,
        # so enable corruption injection only in this disposable copied fixture.
        copied_science = self.root / "singular_sensitivity"
        for copied in (copied_science, *copied_science.rglob("*")):
            copied.chmod(copied.stat().st_mode | 0o200)
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("summarize_pilots.py", "pilot_error_calculus.py", "diagnose_pilot_failures.py"):
            data = (REPOSITORY / "scripts" / name).read_text()
            if name == "summarize_pilots.py":
                # Only the disposable subprocess fixture changes the root;
                # production has no containment bypass.
                data = data.replace('CARC_ROOT = Path("/home1/aadaniel/projects/SSMO")', f"CARC_ROOT = Path({str(self.root)!r})")
            (scripts / name).write_text(data)
        self.environment = patch.dict(os.environ, {"SSMO_PROJECT_ROOT": str(self.root)}, clear=False)
        self.environment.start()
        self.old_legacy = os.environ.pop("SSMO_ROOT", None)
        self.old_slurm = os.environ.pop("SLURM_JOB_ID", None)
        self.carc_root = patch.object(review, "CARC_ROOT", self.root)
        self.carc_root.start()
        self.review_dir = self.root / "runs/completed-review"
        self.inputs = {}
        self.runs = []
        self.parent = {"parent_id": "audit-000000", "split": "audit", "family": "shock", "parameters": [2.0, -0.5, 0.1],
                       "times": [0.4, 0.6], "directions": [[0.2, -0.3, 0.4], [0.0, 0.0, 0.0]], "reference_tier": 0}
        self.queries = query_bank((-2.0, 2.0), held_out=True)
        self.manifest = {"schema_version": 1, "domain": [-2.0, 2.0], "parents": [self.parent]}
        for seed in (17, 29):
            self.make_run(seed)
        self.write_review()

    def tearDown(self):
        if self.old_legacy is not None:
            os.environ["SSMO_ROOT"] = self.old_legacy
        if self.old_slurm is not None:
            os.environ["SLURM_JOB_ID"] = self.old_slurm
        self.carc_root.stop()
        self.environment.stop()
        self.temporary.cleanup()

    def write(self, path, data, recorded=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if recorded:
            self.inputs[str(path.relative_to(self.root))] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def rows(self, seed, method):
        result = []
        for time in self.parent["times"]:
            parent = ParentInput(self.parent["parent_id"], "shock", tuple(self.parent["parameters"]), time, (-2.0, 2.0))
            for direction_index, direction in enumerate(self.parent["directions"]):
                ref = reference_sensitivity(parent, direction)
                exact_position = float(ref.state.edges[1])
                exact_motion = direction[2] + 0.5 * time * (direction[0] + direction[1])
                if method == "measure":
                    prediction = FrontPrediction(np.array([2.04, -0.53]), exact_position + 0.18 + 0.0005 * seed,
                                                 np.array([direction[0] * 1.1, direction[1] * 0.85]), exact_motion * 1.06, parent.domain)
                else:
                    prediction = FrontPrediction(np.array([2.0, -0.5]), exact_position, np.asarray(direction[:2]), exact_motion, parent.domain)
                for query_index, query in enumerate(self.queries):
                    exact, predicted = ref.linear_pairing(query), prediction.linear_pairing(query)
                    metadata = {key: self.parent[key] for key in ("parent_id", "split", "family", "parameters")}
                    row = metadata | {"domain": list(parent.domain), "time": time, "direction": direction,
                                      "direction_index": direction_index, "direction_norm": float(np.linalg.norm(direction)),
                                      "status": "regular", "reference_tier": "exact_analytic_burgers", "event_distance": None,
                                      "method": method + "_chart", "query_index": query_index, "query": asdict(query),
                                      "exact_value": exact, "value": predicted, "absolute_error": abs(predicted - exact),
                                      "relative_error": abs(predicted - exact) / max(0.01, abs(exact)),
                                      "nonlinear_gradient_error": abs(prediction.nonlinear_objective_derivative(query) - ref.nonlinear_objective_derivative(query))}
                    result.append(row | front_errors(ref, prediction))
        # Real JSON artifacts use list-valued query domains rather than tuples.
        return json.loads(json_bytes(result))

    def make_run(self, seed):
        run_id = f"ssmo-pilot-seed{seed}"
        directory = self.root / "runs" / run_id
        directory.mkdir()
        config = b"domain: [-2, 2]\nevaluation: {absolute_tolerance: 0.01}\n"
        self.write(directory / "config.yaml", config, recorded=True)
        self.write(directory / "artifacts/data/parents.json", json_bytes(self.manifest), recorded=True)
        hashes = {}
        for source in sorted((REPOSITORY / "singular_sensitivity").rglob("*.py")):
            destination = directory / "source" / "singular_sensitivity" / source.relative_to(REPOSITORY / "singular_sensitivity")
            self.write(destination, source.read_bytes(), recorded=True)
            hashes[str(destination.relative_to(directory / "source"))] = hashlib.sha256(source.read_bytes()).hexdigest()
        frozen_protocol = {"domain": [-2.0, 2.0], "selected_parent_ids": [self.parent["parent_id"]],
                           "held_out_query_bank": json.loads(json_bytes([asdict(query) for query in self.queries])),
                           "fixed_audit_coverage": {"available": 1, "evaluated": 1, "complete": True}, "absolute_tolerance": 0.01}
        methods = {}
        for method in review.METHODS:
            evaluation_dir = directory / f"artifacts/evaluate-{method}-seed{seed}"
            raw = evaluation_dir / "query_errors.jsonl"
            rows = self.rows(seed, method)
            self.write(raw, b"".join(json_bytes(row) for row in rows))
            worst = max(row["absolute_error"] for row in rows if row["direction_norm"] > 0)
            failures = [{"method": method + "_chart", "parent_id": self.parent["parent_id"], "split": "audit",
                         "invalid_chart_rows": 0, "worst_absolute_error": worst}] if worst > 0.01 else []
            evaluation = {key: value for key, value in frozen_protocol.items() if key != "absolute_tolerance"} | {
                "status": "completed_scoped_pilot", "accuracy_gate": {"absolute_tolerance": 0.01, "parent_method_failures": failures},
                "artifacts": {"query_errors": str(raw)}}
            self.write(evaluation_dir / "evaluation.json", json_bytes(evaluation), recorded=True)
            methods[method] = {"failures": failures}
        self.assertEqual(len(methods["measure"]["failures"]), 1)
        self.assertEqual(len(methods["state_only"]["failures"]), 0)
        self.runs.append({"run_id": run_id, "seed": seed, "methods": methods, "evaluation_protocol": frozen_protocol,
                          "parent_manifest_sha256": review.content_hash(self.manifest), "frozen_config_sha256": hashlib.sha256(config).hexdigest(),
                          "scientific_source_sha256": review.content_hash(hashes)})

    def write_review(self):
        summary = {"schema_version": 1, "runs": self.runs, "pooled_parent_count": None}
        self.write(self.review_dir / "summary.json", json_bytes(summary))
        self.write(self.review_dir / "summary.txt", b"Recorded completed frozen pilot review.\n")
        receipt = {"project_root": str(self.root), "inputs": self.inputs, "read_only_source_runs": [run["run_id"] for run in self.runs],
                   "outputs": {name: hashlib.sha256((self.review_dir / name).read_bytes()).hexdigest() for name in ("summary.json", "summary.txt")}}
        self.write(self.review_dir / "provenance.json", json_bytes(receipt))

    def raw(self, seed=17, method="measure"):
        return self.root / "runs" / f"ssmo-pilot-seed{seed}" / f"artifacts/evaluate-{method}-seed{seed}/query_errors.jsonl"

    def compact(self, seed):
        artifacts = self.root / "runs" / f"ssmo-pilot-seed{seed}" / "artifacts"
        entries = []
        for method in review.METHODS:
            original = self.raw(seed, method)
            archive = original.with_suffix(".jsonl.gz")
            data = original.read_bytes()
            self.write(archive, gzip.compress(data, compresslevel=1, mtime=0))
            entries.append({"original": str(original.relative_to(artifacts)), "archive": str(archive.relative_to(artifacts)),
                            "sha256_uncompressed": hashlib.sha256(data).hexdigest(), "original_bytes": len(data),
                            "compressed_bytes": archive.stat().st_size, "state": "compacted"})
            original.unlink()
        self.write(artifacts / "log-compaction.json", json_bytes({"status": "completed_lossless_log_compaction", "files": entries}))

    def output(self, name="diagnostics"):
        return self.root / "runs" / name

    def diagnose(self, name="diagnostics"):
        return diagnostic.diagnose(self.root, self.review_dir, self.output(name))

    def test_both_methods_full_rows_reconstruct_signed_errors_without_mutating_runs(self):
        before = {path: path.read_bytes() for path in (self.root / "runs").rglob("*") if path.is_file()}
        summary, report = self.diagnose()
        self.assertIsNone(summary["pooled_parent_count"])
        self.assertEqual([run["seed"] for run in summary["runs"]], [17, 29])
        self.assertIn("not a causal proof", report)
        self.assertIn("its own worst query", report)
        self.assertIn("diffuse_support=", report)
        self.assertIn("nonlinear terms:", report)
        self.assertIn("state_only: rows=24 weak_gate=pass", report)
        records = [json.loads(line) for line in (self.output() / "failures.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 96)
        self.assertEqual({record["method"] for record in records}, {"measure", "state_only"})
        self.assertEqual(sum(record["raw"]["direction_norm"] == 0 for record in records), 48)
        for record in records:
            ledger = record["attribution"]
            self.assertAlmostEqual(math.fsum(ledger["contributions"].values()), ledger["signed_error"], places=12)
            self.assertAlmostEqual(math.fsum(ledger["nonlinear"]["contributions"].values()), ledger["nonlinear"]["signed_error"], places=12)
        for run in summary["runs"]:
            methods = run["parents"][0]["methods"]
            self.assertFalse(methods["measure"]["weak_gate_passed"])
            self.assertTrue(methods["state_only"]["weak_gate_passed"])
            self.assertEqual(methods["measure"]["nonzero_query_rows"], 12)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        receipt = json.loads((self.output() / "provenance.json").read_text())
        for name, checksum in receipt["outputs"].items():
            self.assertEqual(checksum, hashlib.sha256((self.output() / name).read_bytes()).hexdigest())

    def test_lossless_archives_are_streamed_and_bad_receipt_refused(self):
        self.compact(17)
        self.compact(29)
        summary, _ = self.diagnose("gzip-diagnostics")
        self.assertTrue(all(row["compressed"] for row in summary["raw_counters"]))
        path = self.root / "runs/ssmo-pilot-seed17/artifacts/log-compaction.json"
        mapping = json.loads(path.read_text())
        mapping["files"][0]["sha256_uncompressed"] = "0" * 64
        self.write(path, json_bytes(mapping))
        with self.assertRaisesRegex(ValueError, "uncompressed checksum/bytes mismatch"):
            self.diagnose("bad-gzip")
        self.assertFalse(self.output("bad-gzip").exists())

    def test_missing_rows_duplicate_rows_and_wrong_protocol_are_rejected_before_output(self):
        path = self.raw()
        original = path.read_bytes()
        lines = original.splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:-1]))
        with self.assertRaisesRegex(ValueError, "missing selected parent query rows"):
            self.diagnose("missing")
        path.write_bytes(original + lines[0])
        with self.assertRaisesRegex(ValueError, "duplicate selected raw query row"):
            self.diagnose("duplicate")
        wrong = json.loads(lines[0])
        wrong["parameters"][0] = 2.01
        path.write_bytes(json_bytes(wrong) + b"".join(lines[1:]))
        with self.assertRaisesRegex(ValueError, "raw parent parameters disagrees"):
            self.diagnose("wrong-parameters")
        path.write_bytes(original)
        wrong = json.loads(lines[0])
        wrong["query"]["frequency"] += 0.2
        path.write_bytes(json_bytes(wrong) + b"".join(lines[1:]))
        with self.assertRaisesRegex(ValueError, "raw direction/query disagrees"):
            self.diagnose("wrong-query")
        self.assertTrue(all(not self.output(name).exists() for name in ("missing", "duplicate", "wrong-parameters", "wrong-query")))

    def test_missing_archive_corrupt_bytes_and_archive_escape_are_rejected(self):
        path = self.raw()
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing with no verified archive"):
            self.diagnose("missing-archive")
        path.write_bytes(original)
        self.compact(17)
        archive = path.with_suffix(".jsonl.gz")
        archive.unlink()
        archive.symlink_to(self.raw(29))
        with self.assertRaisesRegex(ValueError, "canonical path"):
            self.diagnose("alias-archive")
        archive.unlink()
        archive.write_bytes(b"corrupt gzip archive")
        with self.assertRaisesRegex(ValueError, "compressed raw log byte count"):
            self.diagnose("corrupt-archive")

    def test_changed_review_summary_evaluation_manifest_or_formula_provenance_is_refused(self):
        summary_path = self.review_dir / "summary.json"
        original_summary = summary_path.read_bytes()
        summary_path.write_bytes(original_summary + b" ")
        with self.assertRaisesRegex(ValueError, "review output checksum mismatch"):
            self.diagnose("changed-review")
        summary_path.write_bytes(original_summary)
        for relative in ("artifacts/evaluate-measure-seed17/evaluation.json", "artifacts/data/parents.json", "source/singular_sensitivity/queries.py"):
            path = self.root / "runs/ssmo-pilot-seed17" / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "review provenance input checksum mismatch"):
                self.diagnose("changed-input")
            path.write_bytes(original)
        current_formula = self.root / "singular_sensitivity/reference.py"
        current_formula.write_bytes(current_formula.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "current formula source differs"):
            self.diagnose("changed-formula")
        self.assertTrue(all(not self.output(name).exists() for name in ("changed-review", "changed-input", "changed-formula")))

    def test_invalid_unresolved_nonfinite_and_wrong_recorded_worst_error_are_refused(self):
        path = self.raw()
        original = path.read_bytes()
        lines = original.splitlines(keepends=True)
        for status in ("invalid_predicted_chart", "unresolved"):
            changed = json.loads(lines[0])
            changed["status"] = status
            path.write_bytes(json_bytes(changed) + b"".join(lines[1:]))
            with self.assertRaisesRegex(ValueError, "chart is invalid or unresolved"):
                self.diagnose(status)
        changed = json.loads(lines[0])
        changed["value"] = float("nan")
        path.write_bytes((json.dumps(changed) + "\n").encode() + b"".join(lines[1:]))
        with self.assertRaises(ValueError):
            self.diagnose("nonfinite")
        path.write_bytes(original)
        self.runs[0]["methods"]["measure"]["failures"][0]["worst_absolute_error"] += 0.001
        # Lock a self-consistent modified summary/evaluation pair. Raw evidence
        # still disagrees, and must independently expose this corruption.
        evaluation_path = self.root / "runs/ssmo-pilot-seed17/artifacts/evaluate-measure-seed17/evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        evaluation["accuracy_gate"]["parent_method_failures"] = self.runs[0]["methods"]["measure"]["failures"]
        self.write(evaluation_path, json_bytes(evaluation), recorded=True)
        self.write_review()
        with self.assertRaisesRegex(ValueError, "raw worst error disagrees"):
            self.diagnose("wrong-worst")
        self.assertFalse(self.output("wrong-worst").exists())

    def test_source_aliases_output_escape_existing_output_and_nested_output_are_refused(self):
        with self.assertRaisesRegex(ValueError, "escapes SSMO"):
            diagnostic.diagnose(self.root, self.review_dir, "/tmp/ssmo-must-not-write")
        self.output().mkdir()
        marker = self.output() / "summary.json"
        marker.write_bytes(b"preserved existing result")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.diagnose()
        self.assertEqual(marker.read_bytes(), b"preserved existing result")
        with self.assertRaisesRegex(ValueError, "separate from all source runs"):
            diagnostic.diagnose(self.root, self.review_dir, self.root / "runs/ssmo-pilot-seed17/new-output")
        with self.assertRaisesRegex(ValueError, "separate from the completed review"):
            diagnostic.diagnose(self.root, self.review_dir, self.review_dir / "new-output")
        review_alias = self.root / "review-alias"
        review_alias.symlink_to(self.review_dir, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "existing canonical path"):
            diagnostic.diagnose(self.root, review_alias, self.output("aliased-review"))
        path = self.raw()
        path.unlink()
        path.symlink_to(self.raw(29))
        with self.assertRaisesRegex(ValueError, "canonical path"):
            self.diagnose("aliased-log")
        self.assertFalse(self.output("aliased-log").exists())

    def test_metadata_and_helper_changes_mid_diagnosis_preserve_failed_output(self):
        original_stream = diagnostic.stream_context
        for label, path in (("metadata", self.review_dir / "summary.txt"),
                            ("helper", self.root / "scripts/pilot_error_calculus.py")):
            original = path.read_bytes()
            mutated = False
            def mutate_after_output(context, destination=None):
                nonlocal mutated
                result = original_stream(context, destination)
                if destination is not None and not mutated:
                    path.write_bytes(original + b"\n")
                    mutated = True
                return result
            with patch.object(diagnostic, "stream_context", side_effect=mutate_after_output):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    self.diagnose(f"changed-{label}")
            output = self.output(f"changed-{label}")
            self.assertTrue((output / "failed.json").is_file())
            self.assertTrue((output / "failures.jsonl").is_file())
            self.assertFalse((output / "summary.json").exists())
            path.write_bytes(original)

    def test_standalone_python_without_site_packages_and_bash_wrapper_no_jobs(self):
        helper = self.root / "scripts/diagnose_pilot_failures.py"
        result = subprocess.run([sys.executable, "-B", "-S", str(helper), "--project-root", str(self.root),
                                 "--review-dir", str(self.review_dir), "--output-dir", "runs/stdlib-review"],
                                capture_output=True, text=True, env=os.environ.copy(), timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved diagnosis:", result.stdout)
        self.assertTrue(self.output("stdlib-review").is_dir())
        scripts = self.root / "scripts"
        (scripts / "diagnose_pilot_failures.sh").write_text((REPOSITORY / "scripts/diagnose_pilot_failures.sh").read_text().replace(
            "/home1/aadaniel/projects/SSMO", str(self.root)))
        copied_env = (REPOSITORY / "scripts/carc_env.sh").read_text().replace("/home1/aadaniel/projects/SSMO", str(self.root))
        copied_env += f"\nssmo_load_python() {{ :; }}\nssmo_check_venv() {{ export SSMO_PYTHON={shlex.quote(sys.executable)}; }}\n"
        (scripts / "carc_env.sh").write_text(copied_env)
        result = subprocess.run(["bash", str(scripts / "diagnose_pilot_failures.sh"), "--review-dir", str(self.review_dir),
                                 "--output-dir", "runs/bash-review"], capture_output=True, text=True,
                                env=os.environ.copy() | {"SLURM_JOB_ID": "12345"}, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved diagnosis:", result.stdout)
        self.assertTrue(self.output("bash-review").is_dir())


if __name__ == "__main__":
    unittest.main()
