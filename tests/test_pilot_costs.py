"""Artifact-only costs pair complete endpoints and expose inverse oracle work."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


SPEC = importlib.util.spec_from_file_location("ssmo_cost_fixture", Path(__file__).with_name("test_pilot_summary.py"))
fixture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_module)
review = fixture_module.review


class PilotCostTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.PilotSummaryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.run_id, self.run = self.fixture.make_run(17)
        self.device = self.fixture.gpu["device"]

    def evaluation(self, method="measure"):
        path = self.run / f"artifacts/evaluate-{method}-seed17/evaluation.json"
        return path, json.loads(path.read_text())

    def endpoints(self, method="measure"):
        rows = []
        for parent, index in (("test-one", 0), ("audit-shock", 1)):
            for directions, queries in ((1, 1), (3, 8)):
                for name in (method + "_chart", "exact_front", review.CLASSICAL):
                    warm = ((0.002, 0.012) if directions == 1 else (0.006, 0.008))[index] if name.endswith("_chart") else ((0.001, 0.002) if directions == 1 else (0.002, 0.004))[index]
                    cold = (0.01, 0.02)[index] if name.endswith("_chart") else (0.01, 0.04)[index]
                    rows.append({"method": name, "parent_id": parent, "directions": directions, "queries": queries,
                                 "endpoint": "initial inputs -> chart -> directions -> weak queries; includes host transfer",
                                 "measurement": {"median_seconds": warm, "seconds": [warm, warm * 0.9, warm * 1.1],
                                                 "cold_call_seconds": cold, "warmup": 2, "repeats": 3,
                                                 "cuda_synchronized": name.endswith("_chart")}})
        # Components are intentionally huge: they overlap and must never enter
        # endpoint totals or speed ratios.
        rows.append({"method": method + "_chart", "parent_id": "test-one", "kind": "components",
                     "forward_state": {"median_seconds": 9999.0}})
        return rows

    def inverse(self, status="completed"):
        record = {"initial_parameters": [1.1, -0.9, 0.25], "bounds": [[0.2, 2.0], [-2.0, -0.2], [-0.7, 0.7]],
                  "target": {"kind": "sin", "frequency": 1.0, "domain": [-2.0, 2.0]}, "observation_time": 0.4,
                  "fixed_steps": 10, "learning_rate": 0.1, "max_backtracks": 10,
                  "objective": "0.5 integral (u-target)^2; exact payoff-jump derivative", "status": status,
                  "accepted_steps": 10 if status == "completed" else 4, "initial_trusted_objective": 1.7,
                  "final_trusted_objective": 0.026, "trusted_objective_evaluations": 18,
                  "trusted_gradient_reference_evaluations": 30, "exact_gradient_fallbacks": 2,
                  "gradient_error_l2_max": 0.12, "total_seconds": 0.03,
                  "identifiability": "not established", "history": [{"step": 0, "trusted_objective": 1.7}]}
        exact = copy.deepcopy(record)
        exact.update(status="completed", accepted_steps=10, exact_gradient_fallbacks=0,
                     gradient_error_l2_max=0.0, total_seconds=0.01)
        return {"measure_chart": record, "exact_front": exact}

    def add_cost_artifacts(self, method="measure"):
        path, evaluation = self.evaluation(method)
        evaluation["timings"] = self.endpoints(method)
        evaluation["inverse"] = {key.replace("measure_chart", method + "_chart"): value for key, value in self.inverse().items()}
        evaluation["baseline_training"] = {"status": "fitted", "fit_seconds": 0.125, "training_parents": 8,
                                           "information": "exact labels from training parents only",
                                           "front_target": "classical front", "state_target": "ordinary grid state",
                                           "direct_target": "signed sensitivity", "fairness": "different optimizers and label costs"}
        self.fixture.write(path, evaluation)
        return path, evaluation

    def test_paired_parent_ratios_are_separate_for_each_workload_and_cold_call(self):
        _, evaluation = self.add_cost_artifacts()
        costs = review.endpoint_costs(evaluation, "measure", self.device)
        self.assertEqual(len(costs["comparisons"]), 2)
        for comparison in costs["comparisons"]:
            self.assertEqual(comparison["status"], "available")
            single, batch = comparison["workloads"]
            self.assertEqual((single["directions"], single["queries"]), (1, 1))
            self.assertEqual(single["matched_physical_parents"], 2)
            self.assertEqual(single["warm"]["learned_over_control"]["median"], 4.0)
            self.assertEqual(single["warm"]["learned_seconds"]["median"], 0.007)
            self.assertEqual(single["warm"]["control_seconds"]["median"], 0.0015)
            self.assertAlmostEqual(single["warm"]["learned_over_control"]["p90"], 5.6)
            self.assertEqual(batch["warm"]["learned_over_control"]["median"], 2.5)
            self.assertEqual(single["cold"]["learned_over_control"]["median"], 0.75)
            self.assertEqual(single["learned_device"], "cuda:0")
            self.assertEqual(single["control_device"], "cpu")
            self.assertEqual(len(single["paired_records"]), 2)
        self.assertEqual(len(costs["raw_endpoint_records"]), 12)
        self.assertIn("not process/model cold start", costs["cold_scope"])

    def test_each_method_uses_its_own_control_execution_and_preserves_shared_parent_counts(self):
        self.add_cost_artifacts()
        path, evaluation = self.add_cost_artifacts("state_only")
        for row in evaluation["timings"]:
            if row["method"] == "exact_front":
                measurement = row["measurement"]
                measurement["median_seconds"] *= 2
                measurement["seconds"] = [value * 2 for value in measurement["seconds"]]
        self.fixture.write(path, evaluation)
        train = self.run / "artifacts/train-measure-seed17/training_summary.json"
        self.fixture.mutate(train, lambda data: data.update(training_teacher_seconds=3.2, validation_teacher_seconds=0.4))
        summary, text = review.summarize(self.root, [self.run_id], "runs/cost-review")
        run = summary["runs"][0]
        methods = run["methods"]
        measure = methods["measure"]["endpoint_costs"]["comparisons"][0]["workloads"][0]
        state = methods["state_only"]["endpoint_costs"]["comparisons"][0]["workloads"][0]
        self.assertEqual(measure["warm"]["learned_over_control"]["median"], 4.0)
        self.assertEqual(state["warm"]["learned_over_control"]["median"], 2.0)
        self.assertEqual(run["classical"]["total"]["physical_parents"], 2)
        self.assertIsNone(summary["pooled_parent_count"])
        self.assertEqual(methods["measure"]["training"]["teacher_cost_seconds"]["training_teacher_seconds"], 3.2)
        self.assertEqual(methods["measure"]["training"]["elapsed_seconds"], 40.0)
        self.assertIn("aggregate_control_fit_s=0.125", text)
        self.assertIn("exact labels + front/state/direct-grid fits together", text)
        self.assertIn("reference_gradient_calls=30", text)
        self.assertIn("not process/model startup", text)

    def test_legacy_and_missing_metadata_are_explicitly_unavailable(self):
        _, evaluation = self.evaluation()
        costs = review.endpoint_costs(evaluation, "measure", self.device)
        self.assertTrue(all(row["status"] == "unavailable" for row in costs["comparisons"]))
        self.assertEqual(review.inverse_costs(evaluation, "measure", self.device)["status"], "unavailable")
        self.assertEqual(review.baseline_costs(evaluation)["status"], "unavailable")
        evaluation["timings"] = self.endpoints()
        del evaluation["timings"][0]["measurement"]["warmup"]
        result = review.endpoint_costs(evaluation, "measure", self.device)["comparisons"][0]["workloads"][0]
        self.assertEqual(result["warm"]["status"], "unavailable")
        self.assertIsNone(result["warm"]["learned_over_control"])
        self.assertEqual(len(result["paired_records"]), 2)

    def test_missing_cold_call_does_not_become_zero_or_partial_speed_claim(self):
        _, evaluation = self.add_cost_artifacts()
        del evaluation["timings"][0]["measurement"]["cold_call_seconds"]
        single = review.endpoint_costs(evaluation, "measure", self.device)["comparisons"][0]["workloads"][0]
        self.assertEqual(single["warm"]["status"], "available")
        self.assertEqual(single["cold"]["status"], "unavailable")
        self.assertIsNone(single["cold"]["learned_over_control"])
        self.assertEqual(single["cold"]["recorded_physical_parents"], 1)

    def test_duplicate_and_unmatched_endpoint_workloads_are_rejected(self):
        _, evaluation = self.add_cost_artifacts()
        original = copy.deepcopy(evaluation)
        evaluation["timings"].append(copy.deepcopy(evaluation["timings"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate endpoint timing"):
            review.endpoint_costs(evaluation, "measure", self.device)
        original["timings"].pop(1)
        with self.assertRaisesRegex(ValueError, "unmatched physical-parent"):
            review.endpoint_costs(original, "measure", self.device)

    def test_changed_endpoint_timing_protocol_or_device_is_rejected(self):
        _, original = self.add_cost_artifacts()
        for field, value in (("endpoint", "query-only"), ("warmup", 4), ("repeats", 2), ("cuda_synchronized", True)):
            with self.subTest(field=field):
                evaluation = copy.deepcopy(original)
                row = evaluation["timings"][1]
                if field == "endpoint":
                    row[field] = value
                else:
                    row["measurement"][field] = value
                    if field == "repeats":
                        row["measurement"]["seconds"] = [row["measurement"]["median_seconds"]] * value
                with self.assertRaises(ValueError):
                    review.endpoint_costs(evaluation, "measure", self.device)

    def test_nonpositive_nonfinite_and_inconsistent_endpoint_times_are_rejected(self):
        _, original = self.add_cost_artifacts()
        for value in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.subTest(value=value):
                evaluation = copy.deepcopy(original)
                evaluation["timings"][0]["measurement"]["median_seconds"] = value
                with self.assertRaises(ValueError):
                    review.endpoint_costs(evaluation, "measure", self.device)
        original["timings"][0]["measurement"]["median_seconds"] *= 2
        with self.assertRaisesRegex(ValueError, "median disagrees"):
            review.endpoint_costs(original, "measure", self.device)

    def test_inverse_preserves_status_fallbacks_and_diagnostic_oracle_cost(self):
        evaluation = {"inverse": self.inverse("no_trusted_descent_within_fixed_line_search_budget")}
        result = review.inverse_costs(evaluation, "measure", self.device)
        self.assertEqual(result["status"], "available")
        learned = result["methods"]["measure_chart"]
        self.assertEqual(learned["status"], "no_trusted_descent_within_fixed_line_search_budget")
        self.assertEqual(learned["accepted_steps"], 4)
        self.assertEqual(learned["trusted_gradient_reference_evaluations"], 30)
        self.assertEqual(learned["exact_gradient_fallbacks"], 2)
        self.assertEqual(learned["total_seconds"], 0.03)
        self.assertNotIn("history", learned)
        self.assertEqual(learned["gradient_provider_device"], "cuda:0")
        self.assertEqual(learned["trusted_reference_device"], "cpu")
        self.assertIn("every iteration", result["scope"])
        self.assertIn("no autonomous inverse speedup", result["limitations"])

    def test_changed_inverse_protocol_is_rejected_even_if_another_field_is_missing(self):
        for field in review.INVERSE_PROTOCOL:
            with self.subTest(field=field):
                inverse = self.inverse()
                inverse["exact_front"][field] = "changed"
                del inverse["measure_chart"]["total_seconds"]
                with self.assertRaises(ValueError):
                    review.inverse_costs({"inverse": inverse}, "measure", self.device)

    def test_invalid_inverse_counts_status_and_runtime_are_rejected(self):
        for field, value in (("total_seconds", 0), ("total_seconds", float("nan")), ("gradient_error_l2_max", float("inf")),
                             ("status", "succeeded"), ("accepted_steps", 11), ("exact_gradient_fallbacks", -1),
                             ("exact_gradient_fallbacks", 11), ("trusted_objective_evaluations", True), ("accepted_steps", 9)):
            with self.subTest(field=field, value=value):
                inverse = self.inverse()
                inverse["measure_chart"][field] = value
                with self.assertRaises(ValueError):
                    review.inverse_costs({"inverse": inverse}, "measure", self.device)

    def test_absent_inverse_metrics_preserve_available_fields_without_claim(self):
        inverse = self.inverse()
        del inverse["measure_chart"]["trusted_gradient_reference_evaluations"]
        result = review.inverse_costs({"inverse": inverse}, "measure", self.device)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["missing"]["measure_chart"], ["trusted_gradient_reference_evaluations"])
        self.assertEqual(result["methods"]["measure_chart"]["exact_gradient_fallbacks"], 2)

    def test_aggregate_baseline_fit_keeps_label_generation_and_shared_fit_scope(self):
        _, evaluation = self.add_cost_artifacts()
        costs = review.baseline_costs(evaluation)
        self.assertEqual(costs["aggregate_control_fit_seconds"], 0.125)
        self.assertEqual(costs["metadata"]["state_target"], "ordinary grid state")
        self.assertIn("not an isolated classical-front fit", costs["scope"])
        self.assertIn("not independent", costs["accounting"])
        for value in (0, -1, float("nan")):
            evaluation["baseline_training"]["fit_seconds"] = value
            with self.assertRaises(ValueError):
                review.baseline_costs(evaluation)

    def test_cost_summary_standalone_cli_needs_no_site_packages_or_gpu(self):
        self.add_cost_artifacts()
        self.add_cost_artifacts("state_only")
        result = subprocess.run([sys.executable, "-B", "-S", str(self.fixture.copied_helper()),
                                 "--project-root", str(self.root), "--run", self.run_id, "--output-dir", "runs/cli-costs"],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("devices=cuda:0/cpu", result.stdout)
        self.assertIn("exact_fallbacks=2", result.stdout)
        self.assertTrue((self.root / "runs/cli-costs/provenance.json").is_file())

    def test_invalid_partial_cost_evidence_refuses_before_any_output(self):
        path, evaluation = self.add_cost_artifacts()
        evaluation["timings"].pop(1)
        self.fixture.write(path, evaluation)
        with self.assertRaisesRegex(ValueError, "unmatched physical-parent"):
            review.summarize(self.root, [self.run_id], "runs/refused-costs")
        self.assertFalse((self.root / "runs/refused-costs").exists())
