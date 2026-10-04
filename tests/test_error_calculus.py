"""Independent production-formula parity and signed cancellation checks."""
from dataclasses import asdict
import copy
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np

from singular_sensitivity.baselines import FrontPrediction
from singular_sensitivity.queries import Query, query_bank
from singular_sensitivity.reference import ParentInput, reference_sensitivity


SPEC = importlib.util.spec_from_file_location("ssmo_error_calculus", Path(__file__).resolve().parents[1] / "scripts/pilot_error_calculus.py")
calculus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calculus
SPEC.loader.exec_module(calculus)


def raw_row(parent, direction, prediction, query):
    reference = reference_sensitivity(parent, direction)
    exact = reference.linear_pairing(query)
    value = prediction.linear_pairing(query)
    return {"parent_id": parent.parent_id, "family": parent.family, "status": "regular",
            "parameters": list(parent.parameters), "time": parent.time, "direction": list(direction),
            "domain": list(parent.domain), "query": asdict(query), "exact_value": exact, "value": value,
            "absolute_error": abs(value - exact), "predicted_position": prediction.position,
            "predicted_atom_weight": prediction.weight, "predicted_motion": prediction.motion,
            "predicted_states": prediction.states.tolist(), "predicted_diffuse": prediction.diffuse.tolist(),
            "nonlinear_gradient_error": abs(prediction.nonlinear_objective_derivative(query) - reference.nonlinear_objective_derivative(query))}


class ErrorCalculusTests(unittest.TestCase):
    def fixture(self, query=None):
        parent = ParentInput("audit-000000", "shock", (2.0, -0.5, 0.1), 0.6, (-2.0, 2.0))
        direction = (0.2, -0.3, 0.4)
        prediction = FrontPrediction(np.array([1.96, -0.53]), 0.603,
                                     np.array([0.211, -0.288]), 0.37, parent.domain)
        return raw_row(parent, direction, prediction, query or Query("cos", parent.domain, frequency=3.3))

    def test_scalar_queries_match_production_integrals_and_normalization(self):
        domain = (-3.7, 2.6)
        queries = query_bank(domain) + query_bank(domain, held_out=True) + [
            Query("polynomial", domain, degree=0), Query("polynomial", domain, degree=5),
            Query("bump", domain, center=1.4, width=0.08)]
        rng = random.Random(704)
        for query in queries:
            scalar = calculus.ScalarQuery.from_dict(asdict(query))
            self.assertAlmostEqual(scalar.normalization, query.normalization, places=14)
            for _ in range(20):
                first, second = [rng.uniform(domain[0] - 0.3, domain[1] + 0.3) for _ in range(2)]
                self.assertAlmostEqual(scalar.value(first), query.value(first), places=13)
                self.assertAlmostEqual(scalar.integral(first, second), query.integral(first, second), places=13)

    def test_random_front_errors_reconstruct_actual_reference_and_prediction(self):
        rng = random.Random(2917)
        queries = query_bank((-2.0, 2.0)) + query_bank((-2.0, 2.0), held_out=True)
        for index in range(45):
            left, right = rng.uniform(0.1, 1.8), rng.uniform(-1.8, -0.1)
            initial, time = rng.uniform(-0.5, 0.5), rng.uniform(0.1, 0.65)
            parent = ParentInput(f"shock-{index}", "shock", (left, right, initial), time, (-2.0, 2.0))
            direction = tuple(rng.uniform(-0.7, 0.7) for _ in range(3))
            support = initial + 0.5 * time * (left + right)
            motion = direction[2] + 0.5 * time * (direction[0] + direction[1])
            prediction = FrontPrediction(np.array([left + rng.uniform(-0.05, 0.05), right + rng.uniform(-0.05, 0.05)]),
                                         support + rng.uniform(-0.12, 0.12),
                                         np.array([direction[0] + rng.uniform(-0.08, 0.08), direction[1] + rng.uniform(-0.08, 0.08)]),
                                         motion + rng.uniform(-0.05, 0.05), parent.domain)
            for query in queries:
                row = raw_row(parent, direction, prediction, query)
                ledger = calculus.attribute_row(row)
                self.assertAlmostEqual(ledger["reconstructed_exact_value"], row["exact_value"], places=13)
                self.assertAlmostEqual(ledger["reconstructed_prediction_value"], row["value"], places=13)
                self.assertAlmostEqual(math.fsum(ledger["contributions"].values()), row["value"] - row["exact_value"], places=13)
                self.assertAlmostEqual(math.fsum(ledger["nonlinear"]["contributions"].values()), ledger["nonlinear"]["signed_error"], places=13)
                self.assertAlmostEqual(abs(ledger["nonlinear"]["signed_error"]), row["nonlinear_gradient_error"], places=13)

    def test_signed_errors_preserve_cancellation_instead_of_summing_magnitudes(self):
        parent = ParentInput("cancel", "shock", (2.0, -0.5, 0.1), 0.6, (-2.0, 2.0))
        prediction = FrontPrediction(np.array([2.0, -0.5]), 0.55, np.array([0.3, -0.3]), 0.268, parent.domain)
        ledger = calculus.attribute_row(raw_row(parent, (0.2, -0.3, 0.4), prediction, Query("constant", parent.domain)))
        self.assertAlmostEqual(ledger["contributions"]["diffuse_coefficient"], 0.255, places=14)
        self.assertAlmostEqual(ledger["contributions"]["atom_weight"], -0.255, places=14)
        self.assertAlmostEqual(ledger["signed_error"], 0.0, places=14)
        self.assertGreater(sum(abs(term) for term in ledger["contributions"].values()), 0.5)

    def test_constant_and_roundoff_jump_keep_zero_atomic_mass(self):
        parent = ParentInput("constant", "constant", (0.2, 0.2, -0.1), 0.4, (-2.0, 2.0))
        prediction = FrontPrediction(np.array([0.2 + 1e-15, 0.2]), 0.12,
                                     np.array([0.1, 0.1]), 0.45, parent.domain)
        self.assertEqual(prediction.weight, 0.0)
        for query in query_bank(parent.domain, held_out=True):
            ledger = calculus.attribute_row(raw_row(parent, (0.1, 0.1, 0.3), prediction, query))
            self.assertEqual(ledger["reference_front"]["weight"], 0.0)
            self.assertEqual(ledger["prediction_front"]["weight"], 0.0)
            self.assertEqual(ledger["contributions"]["atom_weight"], 0.0)
            self.assertEqual(ledger["contributions"]["atom_position"], 0.0)

    def test_nonlinear_payoff_jump_cannot_be_replaced_by_a_single_trace_atom(self):
        row = self.fixture(Query("constant", (-2.0, 2.0)))
        ledger = calculus.attribute_row(row)
        front = ledger["reference_front"]
        wrong = 0.2 * (2.0 - 1.0) * (front["position"] + 2.0)
        wrong += -0.3 * (-0.5 - 1.0) * (2.0 - front["position"])
        wrong += (2.0 - 1.0) * front["weight"]
        self.assertGreater(abs(wrong - ledger["nonlinear"]["reference_gradient"]), 1.0)

    def test_malformed_or_inconsistent_frozen_records_are_rejected(self):
        changes = [{"status": "unresolved"}, {"family": "collision"}, {"time": -0.1},
                   {"predicted_position": 2.0}, {"predicted_position": float("nan")},
                   {"predicted_states": [-0.5, 2.0]}, {"predicted_diffuse": [0.1]},
                   {"predicted_atom_weight": 7.0}, {"exact_value": 3.0}, {"value": 3.0},
                   {"absolute_error": -0.01}, {"nonlinear_gradient_error": -0.01},
                   {"nonlinear_gradient_error": 5.0}, {"direction": [0.1, True, 0.2]},
                   {"domain": [-1.0, 1.0]}, {"predicted_motion": 10 ** 400},
                   {"predicted_motion": 1e308, "predicted_atom_weight": 0.0}]
        for change in changes:
            with self.subTest(change=change):
                row = copy.deepcopy(self.fixture())
                row.update(change)
                with self.assertRaises(ValueError):
                    calculus.attribute_row(row)
        row = self.fixture()
        row["query"]["domain"] = [-1.0, 1.0]
        with self.assertRaises(ValueError):
            calculus.attribute_row(row)
        for first, second in ((1.0, math.inf), (math.inf, 1.0), (math.nan, 1.0)):
            with self.assertRaises(ValueError):
                calculus.check_close(first, second, "overflowed product")


if __name__ == "__main__":
    unittest.main()
