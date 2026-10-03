"""Independent FP64 calculus checks; no training or PDE speed claims."""
import dataclasses
import importlib.util
import math
import unittest
import numpy as np

from singular_sensitivity.events import collision_event
from singular_sensitivity.queries import Query, query_bank
from singular_sensitivity.reference import ParentInput, reference_state, reference_sensitivity
from singular_sensitivity.representation import PiecewiseState


def perturbed(parent, direction, h):
    parameters = tuple(np.asarray(parent.parameters) + h * np.asarray(direction))
    family = parent.family
    if family in {"constant", "shock", "rarefaction"}:
        family = "shock" if parameters[0] >= parameters[1] else "rarefaction"
    return dataclasses.replace(parent, parameters=parameters, family=family)


class ReferenceTests(unittest.TestCase):
    domain = (-2.0, 2.0)

    def shock(self):
        return ParentInput("shock", "shock", (2.0, -0.5, 0.1), 0.6, self.domain)

    def collision(self, time=1.0):
        return ParentInput("collision", "collision", (2.0, 1.0, -1.0, -0.5, 0.5), time, self.domain)

    def assert_fd(self, parent, direction, nonlinear=False):
        result = reference_sensitivity(parent, direction)
        for query in query_bank(parent.domain) + query_bank(parent.domain, held_out=True):
            exact = result.nonlinear_objective_derivative(query) if nonlinear else result.linear_pairing(query)
            errors = []
            for h in (1e-2, 1e-3, 1e-4, 1e-5):
                plus, minus = reference_state(perturbed(parent, direction, h)), reference_state(perturbed(parent, direction, -h))
                if nonlinear:
                    fd = (plus.squared_tracking_objective(query) - minus.squared_tracking_objective(query)) / (2 * h)
                else:
                    fd = (plus.linear_query(query) - minus.linear_query(query)) / (2 * h)
                errors.append(abs(fd - exact))
            self.assertLess(errors[-1], 3e-8, (parent.family, query.kind, errors))
            self.assertLess(errors[-1], max(5e-9, errors[0] / 20.0), errors)

    def test_translated_increasing_step_sign(self):
        parent = ParentInput("step", "translated_step", (0.0, 1.0, 0.1), 0.7, self.domain)
        result = reference_sensitivity(parent, [0, 0, 1])
        np.testing.assert_array_equal(result.atom_weights, [-1.0])
        np.testing.assert_array_equal(result.diffuse_density, [0.0, 0.0])
        self.assertEqual(result.linear_pairing(Query("constant", self.domain)), -1.0)
        self.assert_fd(parent, [0.0, 0.0, 1.0])

    def test_worked_shock_and_nonlinear_jump(self):
        result = reference_sensitivity(self.shock(), [0.2, -0.3, 0.4])
        self.assertAlmostEqual(result.atom_positions[0], 0.55, places=15)
        self.assertAlmostEqual(result.position_jvps[0], 0.37, places=15)
        self.assertAlmostEqual(result.atom_weights[0], 0.925, places=15)
        np.testing.assert_allclose(result.diffuse_density, [0.2, -0.3], atol=0, rtol=0)
        target = Query("constant", self.domain)
        s = 0.55
        bulk = (2.0 - 1.0) * 0.2 * (s + 2.0) + (-0.5 - 1.0) * -0.3 * (2.0 - s)
        correct_jump = 0.5 * ((2.0 - 1.0) ** 2 - (-0.5 - 1.0) ** 2) * 0.37
        self.assertAlmostEqual(result.nonlinear_objective_derivative(target), bulk + correct_jump, places=14)
        incorrect = bulk + (2.0 - 1.0) * result.atom_weights[0]
        self.assertGreater(abs(incorrect - result.nonlinear_objective_derivative(target)), 1.0)
        self.assert_fd(self.shock(), [0.2, -0.3, 0.4])
        self.assert_fd(self.shock(), [0.2, -0.3, 0.4], nonlinear=True)

    def test_rarefaction_has_no_fan_edge_atoms(self):
        parent = ParentInput("fan", "rarefaction", (-0.8, 1.2, 0.1), 0.5, self.domain)
        result = reference_sensitivity(parent, [0.2, -0.4, 0.3])
        self.assertEqual(len(result.atom_positions), 0)
        np.testing.assert_allclose(result.diffuse_density, [0.2, -0.6, -0.4], atol=1e-15)
        for edge in result.state.edges[1:-1]:
            i = np.searchsorted(result.state.edges, edge) - 1
            left = result.state.coefficients[i, 0] + edge * result.state.coefficients[i, 1]
            right = result.state.coefficients[i + 1, 0] + edge * result.state.coefficients[i + 1, 1]
            self.assertAlmostEqual(left, right, places=14)
        self.assert_fd(parent, [0.2, -0.4, 0.3])
        self.assert_fd(parent, [0.2, -0.4, 0.3], nonlinear=True)

    def test_constant_zero_jump_control_and_generic_tangent(self):
        parent = ParentInput("constant", "constant", (0.4, 0.4, 0.1), 0.6, self.domain)
        zero = reference_sensitivity(parent, [0, 0, 0])
        self.assertEqual(zero.total_variation, 0)
        self.assertEqual(len(zero.atom_positions), 0)
        np.testing.assert_allclose(zero.state.cell_averages(11), 0.4, atol=2e-15)
        for direction in ([0.2, 0.2, 0.3], [0.2, -0.3, 0.4]):
            result = reference_sensitivity(parent, direction)
            self.assertEqual(len(result.atom_weights), 0)
            self.assert_fd(parent, direction)
            self.assert_fd(parent, direction, nonlinear=True)

    def test_collision_event_time_term_and_fd(self):
        direction = [0.1, -0.3, 0.2, 0.15, -0.2]
        result = reference_sensitivity(self.collision(), direction)
        event = collision_event(self.collision().parameters, 1.0, direction)
        self.assertAlmostEqual(event.time, 2.0 / 3.0, places=15)
        self.assertAlmostEqual(event.time_jvp, -19.0 / 90.0, places=15)
        ds_without_event = 0.15 - 0.1 * (2 / 3) + 0.15 * (1 / 3)
        self.assertAlmostEqual(result.position_jvps[0], ds_without_event + event.time_jvp, places=15)
        self.assertGreater(abs(result.position_jvps[0] - ds_without_event), 0.2)
        for time in (0.2, 1.0):
            self.assert_fd(self.collision(time), direction)
            self.assert_fd(self.collision(time), direction, nonlinear=True)

    def test_exact_event_is_unresolved_and_near_event_is_scoped(self):
        tau = 2.0 / 3.0
        query = Query("constant", self.domain)
        for side in (None, "left", "right"):
            result = reference_sensitivity(self.collision(tau), [0, 0, 0, 1, 0], side=side)
            self.assertEqual(result.status, "unresolved")
            self.assertTrue(np.isnan(result.atom_weights).all())
            with self.assertRaisesRegex(ValueError, "unresolved"):
                result.linear_pairing(query)
            with self.assertRaisesRegex(ValueError, "unresolved"):
                result.nonlinear_objective_derivative(query)
        for time, count in ((tau - 1e-7, 2), (tau + 1e-7, 1)):
            result = reference_sensitivity(self.collision(time), [0, 0, 0, 1, 0], event_tolerance=1e-6)
            self.assertEqual(result.status, "near_event")
            self.assertEqual(len(result.atom_positions), count)
            self.assertTrue(np.isfinite(result.linear_pairing(query)))
        self.assertEqual(reference_sensitivity(self.collision(1.0), np.zeros(5)).status, "regular")

    def test_direction_linearity_zero_and_shared_support(self):
        parents = [self.shock(), self.collision(0.2), self.collision(1.0),
                   ParentInput("fan", "rarefaction", (-0.8, 1.2, 0.1), 0.5, self.domain)]
        for parent in parents:
            v = np.linspace(-0.2, 0.4, len(parent.parameters))
            w = np.linspace(0.5, -0.1, len(parent.parameters))
            rv, rw, rsum = (reference_sensitivity(parent, d) for d in (v, w, v + w))
            rscale, rzero = reference_sensitivity(parent, -2.5 * v), reference_sensitivity(parent, np.zeros_like(v))
            np.testing.assert_array_equal(rv.atom_positions, rw.atom_positions)
            np.testing.assert_array_equal(rv.left_traces, rw.left_traces)
            np.testing.assert_allclose(rsum.atom_weights, rv.atom_weights + rw.atom_weights, atol=3e-16)
            self.assertEqual(rzero.total_variation, 0)
            for query in query_bank(self.domain, held_out=True):
                self.assertAlmostEqual(rsum.linear_pairing(query), rv.linear_pairing(query) + rw.linear_pairing(query), places=14)
                self.assertAlmostEqual(rscale.linear_pairing(query), -2.5 * rv.linear_pairing(query), places=14)

    def test_atomic_permutation_and_resolution_refinement(self):
        result = reference_sensitivity(self.collision(0.2), [0.1, -0.3, 0.2, 0.15, -0.2])
        permuted = dataclasses.replace(result, **{name: getattr(result, name)[::-1]
            for name in ("atom_positions", "atom_weights", "left_traces", "right_traces", "position_jvps")})
        diffuse = PiecewiseState(result.state.edges, np.column_stack([result.diffuse_density, np.zeros(len(result.diffuse_density))]))
        for query in query_bank(self.domain, held_out=True):
            self.assertAlmostEqual(result.linear_pairing(query), permuted.linear_pairing(query), places=15)
            self.assertAlmostEqual(result.nonlinear_objective_derivative(query), permuted.nonlinear_objective_derivative(query), places=15)
            errors = []
            for n in (16, 64, 256):
                dx = 4.0 / n
                x = -2.0 + dx * (np.arange(n) + 0.5)
                raster_bulk = np.sum(diffuse.cell_averages(n) * dx * query.value(x))
                pairing = raster_bulk + np.sum(result.atom_weights * query.value(result.atom_positions))
                errors.append(abs(pairing - result.linear_pairing(query)))
                self.assertAlmostEqual(np.sum(result.state.cell_averages(n) * dx), result.state.integral(), places=13)
            self.assertLess(errors[-1], max(1e-7, errors[0] / 4), errors)

    def test_invalid_regimes_and_shapes_rejected(self):
        with self.assertRaises(ValueError):
            ParentInput("fan", "rarefaction", (0, 1, 0), 0, self.domain)
        with self.assertRaises(ValueError):
            reference_state(ParentInput("boundary", "shock", (5, 4, 0), 1, self.domain))
        with self.assertRaises(ValueError):
            reference_sensitivity(self.shock(), [1, 2])
        with self.assertRaises(ValueError):
            collision_event((1, 1, 0, 0, 1), 1, np.zeros(5))
        with self.assertRaises(ValueError):
            PiecewiseState([0, 0, 1], [[1, 0], [2, 0]])
        with self.assertRaises(ValueError):
            reference_sensitivity(self.shock(), [0, 0, 1]).linear_pairing(Query("constant", (0, 1)))


class QueryTests(unittest.TestCase):
    def test_analytic_integrals_moments_and_squared_integrals(self):
        domain = (-1.7, 2.9)
        nodes, weights = np.polynomial.legendre.leggauss(64)
        for query in query_bank(domain) + query_bank(domain, held_out=True):
            breaks = [domain[0], domain[0] + 0.31 * (domain[1] - domain[0]), domain[1]]
            if query.kind == "bump":
                breaks += [domain[0] + (domain[1] - domain[0]) * z for z in (query.center - query.width, query.center + query.width)
                           if 0 < z < 1]
            breaks = sorted(breaks)
            for lo, hi in zip(breaks[:-1], breaks[1:]):
                x = 0.5 * (hi + lo) + 0.5 * (hi - lo) * nodes
                scale = 0.5 * (hi - lo)
                q = query.value(x)
                self.assertAlmostEqual(query.integral(lo, hi), scale * np.dot(weights, q), places=13)
                self.assertAlmostEqual(query.first_moment(lo, hi), scale * np.dot(weights, x * q), places=13)
                self.assertAlmostEqual(query.square_integral(lo, hi), scale * np.dot(weights, q * q), places=13)

    def test_bank_amplitude_and_nondimensional_lipschitz_normalization(self):
        for domain in ((-2, 2), (0, 0.01), (-100, 50)):
            z = np.linspace(0, 1, 10001)
            x = domain[0] + (domain[1] - domain[0]) * z
            for query in query_bank(domain) + query_bank(domain, held_out=True):
                values = query.value(x)
                self.assertLessEqual(np.max(np.abs(values)), 1.0 + 2e-14)
                self.assertLessEqual(np.max(np.abs(np.diff(values) / np.diff(z))), 1.0 + 2e-10)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch is not installed")
    def test_torch_integrals_preserve_dtype_and_autograd(self):
        import torch
        domain = (-1.7, 2.9)
        for query in query_bank(domain) + query_bank(domain, held_out=True):
            lo = torch.tensor(-0.83, dtype=torch.float64, requires_grad=True)
            hi = torch.tensor(1.51, dtype=torch.float64, requires_grad=True)
            for method, np_method, factors in ((query.torch_integral, query.integral, (1, 1)),
                (query.torch_first_moment, query.first_moment, (lo.detach(), hi.detach())),
                (query.torch_square_integral, query.square_integral, (query.torch_value(lo).detach(), query.torch_value(hi).detach()))):
                integral = method(lo, hi)
                self.assertEqual(integral.dtype, torch.float64)
                self.assertAlmostEqual(integral.item(), np_method(lo.item(), hi.item()), places=14)
                dlo, dhi = torch.autograd.grad(integral, (lo, hi))
                self.assertAlmostEqual(dlo.item(), (-query.torch_value(lo) * factors[0]).item(), places=13)
                self.assertAlmostEqual(dhi.item(), (query.torch_value(hi) * factors[1]).item(), places=13)


if __name__ == "__main__":
    unittest.main()
