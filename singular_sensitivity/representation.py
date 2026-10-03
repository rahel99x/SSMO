"""Physical-coordinate state charts and signed sensitivity measures.

The state coefficients have shape [R,2], with u(x)=c0+c1*x in region R.
Diffuse directional densities have shape [R]; atomic arrays have shape [J].
Atomic weights are masses and never receive a grid-cell-width factor.
"""
from dataclasses import dataclass, field
from typing import Any
import numpy as np


def _readonly_array(value, ndim):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"expected {ndim}-dimensional array")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PiecewiseState:
    edges: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self):
        edges = _readonly_array(self.edges, 1)
        coefficients = _readonly_array(self.coefficients, 2)
        if len(edges) < 2 or coefficients.shape != (len(edges) - 1, 2):
            raise ValueError("edges[R+1] and coefficients[R,2] must agree")
        if not np.isfinite(edges).all() or not np.isfinite(coefficients).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("state chart must have finite coefficients and strictly ordered edges")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def domain(self):
        return float(self.edges[0]), float(self.edges[-1])

    def evaluate(self, x):
        """Evaluate with the right-trace convention at an interior jump."""
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all() or np.any(x < self.edges[0]) or np.any(x > self.edges[-1]):
            raise ValueError("evaluation points must lie within the state domain")
        index = np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, len(self.coefficients) - 1)
        result = self.coefficients[index, 0] + self.coefficients[index, 1] * x
        return float(result) if result.ndim == 0 else result

    def integral(self, lo=None, hi=None):
        lo = self.edges[0] if lo is None else float(lo)
        hi = self.edges[-1] if hi is None else float(hi)
        if not np.isfinite([lo, hi]).all() or lo < self.edges[0] or hi > self.edges[-1] or hi < lo:
            raise ValueError("integration bounds must be ordered and inside the domain")
        a = np.maximum(self.edges[:-1], lo)
        b = np.minimum(self.edges[1:], hi)
        b = np.maximum(a, b)
        return float(np.sum(self.coefficients[:, 0] * (b - a) + 0.5 * self.coefficients[:, 1] * (b * b - a * a)))

    def cell_averages(self, n):
        if not isinstance(n, int) or n < 1:
            raise ValueError("cell count must be a positive integer")
        grid = np.linspace(self.edges[0], self.edges[-1], n + 1)
        return np.array([self.integral(a, b) / (b - a) for a, b in zip(grid[:-1], grid[1:])])

    def _check_query(self, query):
        if tuple(query.domain) != self.domain:
            raise ValueError("query and state physical domains must match")

    def linear_query(self, query):
        self._check_query(query)
        a, b = self.edges[:-1], self.edges[1:]
        return float(np.sum(self.coefficients[:, 0] * query.integral(a, b) +
                            self.coefficients[:, 1] * query.first_moment(a, b)))

    def squared_tracking_objective(self, target):
        """Return 0.5 integral (u(x)-target(x))**2 dx."""
        self._check_query(target)
        a, b = self.edges[:-1], self.edges[1:]
        c0, c1 = self.coefficients[:, 0], self.coefficients[:, 1]
        square = c0 * c0 * (b - a) + c0 * c1 * (b * b - a * a) + c1 * c1 * (b ** 3 - a ** 3) / 3.0
        cross = c0 * target.integral(a, b) + c1 * target.first_moment(a, b)
        return float(0.5 * np.sum(square - 2.0 * cross + target.square_integral(a, b)))


@dataclass(frozen=True)
class SensitivityResult:
    parent: Any
    direction: np.ndarray
    state: PiecewiseState
    diffuse_density: np.ndarray
    atom_positions: np.ndarray
    atom_weights: np.ndarray
    left_traces: np.ndarray
    right_traces: np.ndarray
    position_jvps: np.ndarray
    status: str = "regular"
    event_metadata: dict = field(default_factory=dict)
    model_version: str = "burgers-exact-v1"
    precision_and_validation_record: dict = field(default_factory=lambda: {
        "arithmetic": "float64", "reference_tier": 0,
        "claim": "analytic formula evaluated in floating point; not a rigorous certificate"})

    def __post_init__(self):
        if self.status not in {"regular", "near_event", "one_sided", "unresolved"}:
            raise ValueError("invalid derivative status")
        fields = ("direction", "diffuse_density", "atom_positions", "atom_weights",
                  "left_traces", "right_traces", "position_jvps")
        for name in fields:
            object.__setattr__(self, name, _readonly_array(getattr(self, name), 1))
        if self.diffuse_density.shape != (len(self.state.coefficients),):
            raise ValueError("diffuse density must have one value per state region")
        j = len(self.atom_positions)
        if any(getattr(self, name).shape != (j,) for name in ("atom_weights", "left_traces", "right_traces", "position_jvps")):
            raise ValueError("all atomic arrays must have shape [J]")
        if not np.isfinite(self.direction).all() or not np.isfinite(self.atom_positions).all():
            raise ValueError("directions and support positions must be finite")
        if self.status != "unresolved" and any(not np.isfinite(getattr(self, name)).all() for name in fields):
            raise ValueError("resolved derivatives must be finite")
        a, b = self.state.domain
        if np.any(self.atom_positions <= a) or np.any(self.atom_positions >= b):
            raise ValueError("atomic support must lie strictly inside the domain")

    @property
    def parent_id(self):
        return self.parent.parent_id

    @property
    def observation_time(self):
        return self.parent.time

    @property
    def physical_domain(self):
        return self.state.domain

    @property
    def parameter_definition(self):
        return ("uL", "uM", "uR", "a", "b") if len(self.parent.parameters) == 5 else ("uL", "uR", "a")

    @property
    def diffuse_coefficients(self):
        return self.diffuse_density

    @property
    def shock_positions(self):
        return self.atom_positions

    @property
    def total_variation(self):
        self._require_derivative()
        return float(np.sum(np.abs(self.diffuse_density) * np.diff(self.state.edges)) + np.sum(np.abs(self.atom_weights)))

    def _require_derivative(self):
        if self.status == "unresolved":
            raise ValueError("derivative is unresolved at this event; no fabricated gradient is available")

    def linear_pairing(self, query):
        self._require_derivative()
        self.state._check_query(query)
        bulk = np.sum(self.diffuse_density * query.integral(self.state.edges[:-1], self.state.edges[1:]))
        atomic = np.sum(self.atom_weights * query.value(self.atom_positions))
        return float(bulk + atomic)

    def nonlinear_objective_derivative(self, target):
        """Correct derivative of 0.5 integral (u-target)**2.

        The jump term is the objective-density jump times interface motion;
        evaluating (u-target) against the state atom at one trace is incorrect.
        """
        self._require_derivative()
        self.state._check_query(target)
        a, b = self.state.edges[:-1], self.state.edges[1:]
        c0, c1 = self.state.coefficients[:, 0], self.state.coefficients[:, 1]
        bulk = np.sum(self.diffuse_density * (c0 * (b - a) + 0.5 * c1 * (b * b - a * a) - target.integral(a, b)))
        d = target.value(self.atom_positions)
        jump = 0.5 * ((self.left_traces - d) ** 2 - (self.right_traces - d) ** 2)
        return float(bulk + np.sum(jump * self.position_jvps))
