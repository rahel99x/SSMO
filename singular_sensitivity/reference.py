"""Exact Burgers Riemann and two-front references on fixed physical domains.

These formulas describe the whole-line entropy solution restricted to an
interval before any wave reaches its boundary. Boundary interactions and
exact-collision parameter derivatives are deliberately unresolved/unsupported.
Translated steps are a static measure-calculus control, not an entropy claim.
"""
from dataclasses import dataclass
import numpy as np
from .events import collision_event
from .representation import PiecewiseState, SensitivityResult


@dataclass(frozen=True)
class ParentInput:
    parent_id: str
    family: str
    parameters: tuple[float, ...]
    time: float
    domain: tuple[float, float]

    def __post_init__(self):
        object.__setattr__(self, "parameters", tuple(float(x) for x in self.parameters))
        object.__setattr__(self, "domain", tuple(float(x) for x in self.domain))
        if not isinstance(self.parent_id, str) or not self.parent_id:
            raise ValueError("parent_id must be a nonempty string")
        if self.family not in {"step", "translated_step", "shock", "rarefaction", "constant", "collision"}:
            raise ValueError(f"unsupported analytic family: {self.family}")
        expected = 5 if self.family == "collision" else 3
        if len(self.parameters) != expected or not np.isfinite(self.parameters).all():
            raise ValueError(f"family {self.family} requires {expected} finite parameters")
        if len(self.domain) != 2 or not np.isfinite(self.domain).all() or self.domain[0] >= self.domain[1]:
            raise ValueError("domain must be a finite increasing interval")
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError("observation time must be finite and nonnegative")
        if self.family == "collision":
            uL, uM, uR, a, b = self.parameters
            if not (uL > uM > uR and self.domain[0] < a < b < self.domain[1]):
                raise ValueError("collision requires uL>uM>uR and interior initial a<b")
        else:
            uL, uR, a = self.parameters
            if not self.domain[0] < a < self.domain[1]:
                raise ValueError("initial Riemann location must be strictly inside domain")
            if self.family == "shock" and uL < uR:
                raise ValueError("Burgers shock requires uL>=uR")
            if self.family == "rarefaction" and (uL >= uR or self.time <= 0):
                raise ValueError("rarefaction requires uL<uR and t>0")
            if self.family == "constant" and uL != uR:
                raise ValueError("constant control requires equal uL,uR")


def _state(domain, internal_edges, coefficients):
    if any(not domain[0] < s < domain[1] for s in internal_edges):
        raise ValueError("wave reached domain boundary; this no-boundary-interaction reference is unsupported")
    return PiecewiseState(np.asarray([domain[0], *internal_edges, domain[1]]), np.asarray(coefficients))


def reference_state(parent):
    p, t, domain = parent.parameters, parent.time, parent.domain
    if parent.family in {"step", "translated_step"}:
        uL, uR, a = p
        return _state(domain, [a], [(uL, 0.0), (uR, 0.0)])
    if parent.family in {"shock", "constant"}:
        uL, uR, a = p
        s = a + 0.5 * (uL + uR) * t
        return _state(domain, [s], [(uL, 0.0), (uR, 0.0)])
    if parent.family == "rarefaction":
        uL, uR, a = p
        return _state(domain, [a + uL * t, a + uR * t], [(uL, 0.0), (-a / t, 1.0 / t), (uR, 0.0)])
    uL, uM, uR, a, b = p
    event = collision_event(p, t, np.zeros(5))
    c1, c2, c3 = 0.5 * (uL + uM), 0.5 * (uM + uR), 0.5 * (uL + uR)
    if event.phase == "pre_collision":
        return _state(domain, [a + c1 * t, b + c2 * t], [(uL, 0.0), (uM, 0.0), (uR, 0.0)])
    s = a + c1 * event.time + c3 * (t - event.time)
    return _state(domain, [s], [(uL, 0.0), (uR, 0.0)])


def reference_sensitivity(parent, direction, event_tolerance=1e-8, side=None):
    """Assemble a signed measure using exact chart and event derivatives.

    Supports and traces depend on the parent only. Near-event formulas remain
    scoped to their current regular stratum. At exact collision, side requests
    are recorded but remain unresolved, with NaN derivative arrays and methods
    that refuse to return a query gradient.
    """
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (len(parent.parameters),) or not np.isfinite(direction).all():
        raise ValueError("direction must be a finite vector matching parent parameters")
    if not np.isfinite(event_tolerance) or event_tolerance < 0:
        raise ValueError("event tolerance must be finite and nonnegative")
    if side not in {None, "left", "right", "minus", "plus"}:
        raise ValueError("side must be left/right (minus/plus) or None")
    state = reference_state(parent)
    p, t = parent.parameters, parent.time
    status, metadata = "regular", {"event_type": None, "event_distance": None}
    if parent.family in {"step", "translated_step", "shock", "constant"}:
        uL, uR, a = p
        vL, vR, va = direction
        s = state.edges[1]
        ds = va if parent.family in {"step", "translated_step"} else va + 0.5 * t * (vL + vR)
        diffuse = [vL, vR]
        if uL == uR:
            positions, weights, left, right, motions = [], [], [], [], []
        else:
            positions, weights, left, right, motions = [s], [(uL - uR) * ds], [uL], [uR], [ds]
    elif parent.family == "rarefaction":
        vL, vR, va = direction
        diffuse = [vL, -va / t, vR]
        positions, weights, left, right, motions = [], [], [], [], []
    else:
        uL, uM, uR, a, b = p
        vL, vM, vR, va, vb = direction
        event = collision_event(p, t, direction, event_tolerance)
        metadata, status = event.metadata(), event.status
        metadata["requested_parameter_side"] = side
        if event.phase == "pre_collision":
            diffuse = [vL, vM, vR]
            positions = state.edges[1:-1]
            motions = [va + 0.5 * t * (vL + vM), vb + 0.5 * t * (vM + vR)]
            left, right = [uL, uM], [uM, uR]
            weights = np.subtract(left, right) * motions
        else:
            positions, left, right = state.edges[1:-1], [uL], [uR]
            if event.status == "unresolved":
                diffuse, weights, motions = [np.nan, np.nan], [np.nan], [np.nan]
            else:
                c1, c3 = 0.5 * (uL + uM), 0.5 * (uL + uR)
                dc1, dc3 = 0.5 * (vL + vM), 0.5 * (vL + vR)
                ds = va + dc1 * event.time + dc3 * (t - event.time) + (c1 - c3) * event.time_jvp
                diffuse, motions, weights = [vL, vR], [ds], [(uL - uR) * ds]
    return SensitivityResult(parent, direction, state, diffuse, positions, weights, left, right, motions,
                             status=status, event_metadata=metadata)
