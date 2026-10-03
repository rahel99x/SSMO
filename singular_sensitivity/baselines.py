"""Small train-only classical and grid controls for the single-shock pilot.

These polynomial regressors receive only initial parameters and time. They are
controls, not claimed replacements for an entropy solver or performance wins.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement
import time

import numpy as np


def polynomial_powers(dimension, degree=2):
    result = [tuple([0] * dimension)]
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(dimension), order):
            counts = [0] * dimension
            for index in indices:
                counts[index] += 1
            result.append(tuple(counts))
    return np.asarray(result, dtype=int)


class PolynomialFeatures:
    def __init__(self, inputs, degree=2):
        inputs = np.asarray(inputs, dtype=np.float64)
        self.center = inputs.mean(axis=0)
        self.scale = inputs.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        self.powers = polynomial_powers(inputs.shape[1], degree)

    def values(self, inputs):
        z = (np.asarray(inputs, dtype=np.float64) - self.center) / self.scale
        return np.prod(z[..., None, :] ** self.powers, axis=-1)

    def jvp(self, inputs, direction):
        z = (np.asarray(inputs, dtype=np.float64) - self.center) / self.scale
        tangent = np.asarray(direction, dtype=np.float64) / self.scale
        result = np.zeros(z.shape[:-1] + (len(self.powers),), dtype=np.float64)
        for axis in range(z.shape[-1]):
            mask = self.powers[:, axis] > 0
            exponents = self.powers[mask].copy()
            exponents[:, axis] -= 1
            result[..., mask] += (self.powers[mask, axis] * tangent[..., axis, None]
                                  * np.prod(z[..., None, :] ** exponents, axis=-1))
        return result

    def torch_values(self, inputs):
        import torch
        z = (inputs - torch.as_tensor(self.center, dtype=inputs.dtype, device=inputs.device)) / torch.as_tensor(
            self.scale, dtype=inputs.dtype, device=inputs.device)
        powers = torch.as_tensor(self.powers, device=inputs.device)
        return (z.unsqueeze(-2) ** powers).prod(dim=-1)


@dataclass
class FrontPrediction:
    states: np.ndarray
    position: float
    diffuse: np.ndarray
    motion: float
    domain: tuple

    @property
    def weight(self):
        jump = float(self.states[0] - self.states[1])
        tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(self.states))))
        return 0.0 if abs(jump) <= tolerance else float(jump * self.motion)

    @property
    def admissible(self):
        tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(self.states))))
        return bool(self.domain[0] < self.position < self.domain[1]
                    and self.states[0] >= self.states[1] - tolerance
                    and np.isfinite(np.r_[self.states, self.position, self.diffuse, self.motion]).all())

    def linear_pairing(self, query):
        lo, hi = self.domain
        return float(self.diffuse[0] * query.integral(lo, self.position)
                     + self.diffuse[1] * query.integral(self.position, hi)
                     + self.weight * query.value(self.position))

    def state_values(self, x):
        return np.where(np.asarray(x) < self.position, self.states[0], self.states[1])

    def linear_state_query(self, query):
        lo, hi = self.domain
        return float(self.states[0] * query.integral(lo, self.position)
                     + self.states[1] * query.integral(self.position, hi))

    def nonlinear_objective_derivative(self, target):
        lo, hi = self.domain
        bulk = self.diffuse[0] * (self.states[0] * (self.position - lo) - target.integral(lo, self.position))
        bulk += self.diffuse[1] * (self.states[1] * (hi - self.position) - target.integral(self.position, hi))
        d = target.value(self.position)
        jump = 0.5 * ((self.states[0] - d) ** 2 - (self.states[1] - d) ** 2) * self.motion
        return float(bulk + jump)


class ClassicalFrontRegression:
    """Polynomial regression of region values and front coordinates, not tangents."""
    def __init__(self, features, coefficients, domain):
        self.features, self.coefficients, self.domain = features, coefficients, tuple(domain)

    def predict(self, alpha, observation_time, direction):
        inputs = np.r_[np.asarray(alpha), observation_time]
        values = self.features.values(inputs) @ self.coefficients
        tangent = self.features.jvp(inputs, np.r_[direction, 0.0]) @ self.coefficients
        return FrontPrediction(values[:2], float(values[2]), tangent[:2], float(tangent[2]), self.domain)


class LearnedGridState:
    """Ordinary smooth polynomial state regressor with a real PyTorch JVP."""
    def __init__(self, features, coefficients, domain):
        self.features, self.coefficients, self.domain = features, coefficients, tuple(domain)
        self.resolution = coefficients.shape[1]

    def predict(self, alpha, observation_time):
        return self.features.values(np.r_[alpha, observation_time]) @ self.coefficients

    def jvp(self, alpha, observation_time, direction):
        import torch
        inputs = torch.as_tensor(np.r_[alpha, observation_time], dtype=torch.float64)
        tangent = torch.as_tensor(np.r_[direction, 0.0], dtype=torch.float64)
        weights = torch.as_tensor(self.coefficients, dtype=torch.float64)
        _, derivative = torch.func.jvp(lambda x: self.features.torch_values(x) @ weights,
                                       (inputs,), (tangent,))
        return derivative.detach().numpy()


class DirectGridSensitivity:
    """Linear-in-direction regression of cell-average inviscid measure densities."""
    def __init__(self, features, coefficients, domain):
        self.features, self.coefficients, self.domain = features, coefficients, tuple(domain)
        self.resolution = coefficients.shape[1]

    def predict(self, alpha, observation_time, direction):
        features = (self.features.values(np.r_[alpha, observation_time])[:, None]
                    * np.asarray(direction)[None, :]).reshape(-1)
        return features @ self.coefficients


def density_cell_masses(edges, density, cell_edges):
    """Exact integration of a piecewise-constant diffuse density into cells."""
    edges, density, cell_edges = map(np.asarray, (edges, density, cell_edges))
    mass = np.zeros(len(cell_edges) - 1, dtype=np.float64)
    variation = np.zeros_like(mass)
    for left, right, value in zip(edges[:-1], edges[1:], density):
        overlap = np.maximum(0.0, np.minimum(cell_edges[1:], right) - np.maximum(cell_edges[:-1], left))
        mass += value * overlap
        variation += abs(value) * overlap
    return mass, variation


def measure_cell_masses(measure, n):
    edges = np.linspace(measure.parent.domain[0], measure.parent.domain[1], n + 1)
    masses, _ = density_cell_masses(measure.state.edges, measure.diffuse_density, edges)
    indices = np.searchsorted(edges, measure.atom_positions, side="right") - 1
    for index, weight in zip(indices, measure.atom_weights):
        if index == n:
            index = n - 1
        if 0 <= index < n:
            masses[index] += weight
    return masses


def fit_baselines(dataset, resolution=128, degree=2):
    """Fit all three controls on exactly the declared training parent records."""
    from .reference import ParentInput, reference_sensitivity
    started = time.perf_counter()
    domain = tuple(dataset["domain"])
    parents = [p for p in dataset["parents"] if p["split"] == "train" and p["family"] in {"shock", "constant"}]
    if not parents:
        return {}, {"status": "missing_training_parents"}
    inputs, front_labels, grid_labels, direct_inputs, direct_labels = [], [], [], [], []
    for record in parents:
        for observation_time in record["times"]:
            alpha = np.asarray(record["parameters"], dtype=np.float64)
            parent = ParentInput(record["parent_id"], record["family"], tuple(alpha), observation_time, domain)
            base = reference_sensitivity(parent, np.zeros(3))
            # Constant controls use a conventional zero-jump front with exact same states.
            position = float(alpha[2] + 0.5 * (alpha[0] + alpha[1]) * observation_time)
            inputs.append(np.r_[alpha, observation_time])
            front_labels.append(np.r_[alpha[:2], position])
            grid_labels.append(base.state.cell_averages(resolution))
            for direction in record["directions"]:
                if np.linalg.norm(direction) == 0:
                    continue
                ref = reference_sensitivity(parent, direction)
                if ref.status == "unresolved":
                    continue
                direct_inputs.append((np.r_[alpha, observation_time], np.asarray(direction)))
                direct_labels.append(measure_cell_masses(ref, resolution) * resolution / (domain[1] - domain[0]))
    inputs = np.asarray(inputs)
    features = PolynomialFeatures(inputs, degree)
    design = features.values(inputs)
    front_coefficients, _, rank, _ = np.linalg.lstsq(design, front_labels, rcond=1e-10)
    grid_coefficients, _, grid_rank, _ = np.linalg.lstsq(design, grid_labels, rcond=1e-10)
    controls = {"classical_front_regression": ClassicalFrontRegression(features, front_coefficients, domain),
                "learned_grid_state_autodiff": LearnedGridState(features, grid_coefficients, domain)}
    direct_rank = 0
    if direct_inputs:
        direct_design = np.asarray([(features.values(x)[:, None] * v[None, :]).reshape(-1) for x, v in direct_inputs])
        coefficients, _, direct_rank, _ = np.linalg.lstsq(direct_design, direct_labels, rcond=1e-10)
        controls["direct_grid_sensitivity"] = DirectGridSensitivity(features, coefficients, domain)
    report = {"status": "fitted", "degree": degree, "grid_resolution": resolution,
              "training_parent_ids": [p["parent_id"] for p in parents], "training_parents": len(parents),
              "training_state_examples": len(inputs), "training_direction_examples": len(direct_inputs),
              "feature_count": int(design.shape[1]), "front_rank": int(rank), "state_rank": int(grid_rank),
              "direct_rank": int(direct_rank), "fit_seconds": time.perf_counter() - started,
              "information": "initial parameters and time; exact labels from training parents only",
              "front_target": "two state values and classical front position; degree 2 contains exact Burgers formula",
              "state_target": "exact inviscid cell-average state", "direct_target": "cell-average signed sensitivity measure",
              "fairness": "same training parents; different fitting optimizers and label costs; no matched-budget speed claim"}
    return controls, report


def smoothed_shock(alpha, observation_time, direction, domain, resolution, width):
    """Fixed physical tanh smoothing; derivative belongs to this regularized model.

    This is not a viscous Burgers solve, and shrinking width alone does not prove
    an inviscid PDE sensitivity limit.
    """
    if width <= 0:
        raise ValueError("physical smoothing width must be positive")
    lo, hi = domain
    x = lo + (np.arange(resolution) + 0.5) * (hi - lo) / resolution
    left, right, location = np.asarray(alpha)
    vleft, vright, vlocation = np.asarray(direction)
    position = location + 0.5 * (left + right) * observation_time
    motion = vlocation + 0.5 * (vleft + vright) * observation_time
    transition = np.tanh((x - position) / width)
    state = 0.5 * (left + right) + 0.5 * (right - left) * transition
    derivative = (0.5 * (vleft + vright) + 0.5 * (vright - vleft) * transition
                  + 0.5 * (left - right) * motion * (1.0 - transition ** 2) / width)
    return x, state, derivative
