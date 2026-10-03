"""Conservation, entropy flux, refinement, and bounded numeric-audit tests."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from singular_sensitivity.numerical import godunov_flux, godunov_state, numerical_audit
from singular_sensitivity.queries import query_bank
from singular_sensitivity.reference import ParentInput, reference_state
from singular_sensitivity.runtime import project_root


class NumericalTests(unittest.TestCase):
    domain = (-2.0, 2.0)

    def test_entropy_flux_selects_correct_wave_and_sonic_states(self):
        left = [1, -2, -1, 2, 1, -1, 2]
        right = [2, -1, 1, -1, -2, -2, 1]
        np.testing.assert_array_equal(godunov_flux(left, right), [0.5, 0.5, 0, 2, 2, 2, 2])

    def test_constant_preservation_and_exact_mass_balance(self):
        cases = [("constant", (0.3, 0.3, 0.13)), ("shock", (2.0, -0.5, 0.13)),
                 ("rarefaction", (-0.8, 1.2, 0.13))]
        for family, parameters in cases:
            parent = ParentInput(family, family, parameters, 0.4, self.domain)
            numerical = godunov_state(parent, 128)
            initial = parameters[0] * (parameters[2] + 2) + parameters[1] * (2 - parameters[2])
            expected_mass = initial + 0.4 * 0.5 * (parameters[0] ** 2 - parameters[1] ** 2)
            self.assertAlmostEqual(numerical.integral(), expected_mass, places=13)
            values = numerical.coefficients[:, 0]
            self.assertTrue(np.isfinite(values).all())
            self.assertLessEqual(np.max(values), max(parameters[:2]) + 1e-13)
            self.assertGreaterEqual(np.min(values), min(parameters[:2]) - 1e-13)
            if family == "constant":
                np.testing.assert_allclose(values, 0.3, atol=2e-15, rtol=0)
            elif family == "shock":
                self.assertLessEqual(np.max(np.diff(values)), 1e-14)
            else:
                self.assertGreaterEqual(np.min(np.diff(values)), -1e-14)

    def test_single_shock_and_rarefaction_forward_queries_refine(self):
        for family, parameters in [("shock", (2, -0.5, 0.1)), ("rarefaction", (-0.8, 1.2, 0.1))]:
            parent = ParentInput(family, family, parameters, 0.4, self.domain)
            exact = reference_state(parent)
            errors = []
            for n in (64, 128, 256):
                state = godunov_state(parent, n)
                errors.append(max(abs(state.linear_query(q) - exact.linear_query(q)) for q in query_bank(self.domain) + query_bank(self.domain, held_out=True)))
                # Conservative representation has cell values; it has no inferred physical atom weights.
                self.assertEqual(state.coefficients.shape, (n, 2))
                np.testing.assert_array_equal(state.coefficients[:, 1], 0)
            self.assertLess(errors[-1], errors[0] * 0.7, (family, errors))
            self.assertLess(errors[-1], 0.01)

    def test_numeric_audit_logs_raw_errors_and_explicit_gates(self):
        root = project_root()
        with tempfile.TemporaryDirectory(prefix="numeric-test-", dir=root) as directory:
            summary = numerical_audit({"domain": [-2, 2]}, Path(directory))
            self.assertEqual(summary["status"], "passed", summary["cases"])
            self.assertEqual(summary["solve_count"], 114)
            persisted = json.loads((Path(directory) / "numeric_teacher_audit.json").read_text())
            self.assertEqual(persisted["science_gate_status"], "passed")
            records = [json.loads(line) for line in (Path(directory) / "numeric_teacher_audit.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), summary["raw_record_count"])
            fd = [r for r in records if r["record_type"] == "finite_difference"]
            self.assertEqual(len(fd), 324)
            self.assertTrue(all("observed_amplified_forward_error" in r and "resolution_difference" in r for r in fd))
            self.assertTrue(any(r["resolution_difference"] is not None for r in fd))
            self.assertTrue(all(r["derivative_status"] == "numerical_fd_observation" for r in fd))
            # Rerunning atomically replaces the raw audit instead of duplicating observations.
            second = numerical_audit({"domain": [-2, 2], "numerical": {"forward_tolerance": 1e-14}}, Path(directory))
            self.assertEqual(second["status"], "failed")
            self.assertEqual(second["setup_gate_status"], "passed")
            self.assertEqual(second["raw_record_count"], len(records))

    def test_invalid_cfl_mesh_and_audit_cost_budgets_rejected(self):
        parent = ParentInput("shock", "shock", (2, -0.5, 0.1), 0.4, self.domain)
        for cfl in (0, 0.46, float("nan")):
            with self.assertRaises(ValueError):
                godunov_state(parent, 64, cfl)
        with self.assertRaises(ValueError):
            godunov_state(parent, 2)
        with tempfile.TemporaryDirectory(prefix="numeric-test-", dir=project_root()) as directory:
            for options in ({"resolutions": [128, 4096]}, {"resolutions": [128, 128]}, {"fd_steps": [0.01, 0.05]}):
                with self.assertRaises(ValueError):
                    numerical_audit({"numerical": options}, Path(directory))


if __name__ == "__main__":
    unittest.main()
