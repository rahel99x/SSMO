"""Independent weak diagnostic and trusted inverse correctness checks."""
import unittest
import numpy as np

from singular_sensitivity.baselines import (FrontPrediction, PolynomialFeatures,
                                            density_cell_masses, fit_baselines)
from singular_sensitivity.inverse import inverse_tracking
from singular_sensitivity.queries import Query
from singular_sensitivity.reference import ParentInput, reference_sensitivity
from singular_sensitivity.validation import bounded_lipschitz_lp, discretize_measure, representation_rows


class WeakDiagnostics(unittest.TestCase):
    def test_signed_atomic_lp_and_physical_normalization(self):
        cases = [([0.25], [2.0], (0.0, 1.0), 2.0),
                 ([0.2, 0.5], [1.0, -1.0], (0.0, 1.0), 0.3),
                 ([-1.0, 1.0], [-1.0, 1.0], (-2.0, 2.0), 0.5),
                 ([0.2, 0.2], [2.0, -2.0], (0.0, 1.0), 0.0)]
        for positions, masses, domain, expected in cases:
            with self.subTest(expected=expected):
                result = bounded_lipschitz_lp(positions, masses, domain)
                self.assertAlmostEqual(result["value"], expected, places=11)
                self.assertTrue(result["residual_checks_passed"])
                self.assertFalse(result["rigorous_certificate"])

    def test_diffuse_cancellation_keeps_total_variation_radius(self):
        mass, variation = density_cell_masses([0.0, 0.5, 1.0], [1.0, -1.0], [0.0, 1.0])
        self.assertEqual(mass[0], 0.0)
        self.assertEqual(variation[0], 1.0)
        parent = ParentInput("cancellation", "step", (1.0, -1.0, 0.5), 0.0, (0.0, 1.0))
        measure = reference_sensitivity(parent, [1.0, -1.0, 0.0])
        positions, masses, radius = discretize_measure(measure, resolution=1)
        self.assertAlmostEqual(radius, 0.5)
        diagnostic = bounded_lipschitz_lp(positions, masses, parent.domain, radius)
        self.assertEqual(diagnostic["value"], 0.0)
        # The continuous signed density is not zero despite a zero cell mass.
        self.assertLessEqual(abs(measure.linear_pairing(Query("polynomial", parent.domain))), radius)
        self.assertGreater(radius, diagnostic["value"])

    def test_representation_refinement_and_exact_finite_variation(self):
        parent = ParentInput("refinement", "shock", (2.0, -0.5, 0.1), 0.6, (-2.0, 2.0))
        query = Query("sin", parent.domain, frequency=1.3)
        rows = representation_rows(parent, [0.2, -0.3, 0.4], [query],
                                   resolutions=(32, 256), fd_steps=(1e-3,), smoothing_widths=())
        cell_rows = [r for r in rows if r["method"] == "grid_cell_average_fd"]
        self.assertEqual(len(cell_rows), 2)
        self.assertLess(cell_rows[1]["absolute_error"], cell_rows[0]["absolute_error"])
        for row in cell_rows:
            self.assertLess(row["exact_forward_fd_error"], 1e-7)
            self.assertLessEqual(abs(row["value"] - row["exact_forward_fd"]),
                                 row["forward_quadrature_quotient_error_bound"] + 1e-10)

    def test_exact_collision_is_scoped_unresolved(self):
        parent = ParentInput("collision", "collision", (1.0, 0.0, -1.0, -0.4, 0.4), 0.8, (-2.0, 2.0))
        rows = representation_rows(parent, np.ones(5), [Query("constant", parent.domain)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "unresolved")
        self.assertNotIn("value", rows[0])

    def test_constant_regression_roundoff_is_zero_jump_control(self):
        almost_constant = FrontPrediction(np.array([0.2 - 2e-14, 0.2]), 0.1,
                                          np.array([0.3, -0.4]), 0.7, (-2.0, 2.0))
        self.assertTrue(almost_constant.admissible)
        self.assertEqual(almost_constant.weight, 0.0)
        rarefaction_chart = FrontPrediction(np.array([0.2, 0.2001]), 0.1, np.zeros(2), 0.0, (-2.0, 2.0))
        self.assertFalse(rarefaction_chart.admissible)

    def test_polynomial_feature_jvp_matches_finite_difference(self):
        rng = np.random.default_rng(61)
        features = PolynomialFeatures(rng.normal(size=(20, 4)), degree=3)
        point, direction = rng.normal(size=4), rng.normal(size=4)
        h = 1e-5
        finite = (features.values(point + h * direction) - features.values(point - h * direction)) / (2 * h)
        np.testing.assert_allclose(features.jvp(point, direction), finite, rtol=1e-8, atol=1e-8)

    def test_train_only_classical_front_fit_and_direction_linearity(self):
        rng = np.random.default_rng(19)
        parents = []
        for index in range(12):
            alpha = [float(rng.uniform(0.5, 1.2)), float(rng.uniform(-1.2, -0.5)), float(rng.uniform(-0.3, 0.3))]
            parents.append({"parent_id": f"train-{index}", "split": "train", "family": "shock", "parameters": alpha,
                            "times": [0.2, 0.5, 0.7], "directions": np.eye(3).tolist()})
        parents.append({"parent_id": "test-unseen", "split": "test", "family": "shock", "parameters": [1.1, -0.7, 0.1],
                        "times": [0.4], "directions": np.eye(3).tolist()})
        controls, report = fit_baselines({"domain": [-2.0, 2.0], "parents": parents}, resolution=16)
        self.assertNotIn("test-unseen", report["training_parent_ids"])
        direction = np.asarray([0.2, -0.3, 0.4])
        prediction = controls["classical_front_regression"].predict([1.1, -0.7, 0.1], 0.4, direction)
        self.assertAlmostEqual(prediction.position, 0.18, places=10)
        self.assertAlmostEqual(prediction.motion, 0.38, places=10)
        scaled = controls["direct_grid_sensitivity"].predict([1.1, -0.7, 0.1], 0.4, 2 * direction)
        original = controls["direct_grid_sensitivity"].predict([1.1, -0.7, 0.1], 0.4, direction)
        np.testing.assert_allclose(scaled, 2 * original, atol=1e-11)

    def test_trusted_nonlinear_tracking_descent_and_learned_fallback(self):
        exact = inverse_tracking(steps=5)
        self.assertLess(exact["final_trusted_objective"], exact["initial_trusted_objective"])
        objectives = [step["trusted_objective"] for step in exact["history"]]
        self.assertTrue(all(a > b for a, b in zip(objectives[:-1], objectives[1:])))
        # A deliberately invalid surrogate cannot accept an untrusted candidate.
        fallback = inverse_tracking(steps=2, gradient_provider=lambda *args: [np.nan, np.nan, np.nan])
        self.assertEqual(fallback["exact_gradient_fallbacks"], 2)
        self.assertLess(fallback["final_trusted_objective"], fallback["initial_trusted_objective"])


if __name__ == "__main__":
    unittest.main()
