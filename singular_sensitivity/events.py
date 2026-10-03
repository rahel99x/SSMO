"""Exact transverse two-shock collision timing and conservative statuses."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CollisionEvent:
    time: float
    time_jvp: float
    distance: float
    phase: str
    status: str

    def metadata(self):
        return {"event_type": "two_shock_collision", "event_time": self.time,
                "event_time_jvp": self.time_jvp, "event_distance": self.distance,
                "phase": self.phase, "transverse": True,
                "exact_event_policy": "unresolved; no analyzed one-sided parameter derivative"}


def collision_event(parameters, time, direction, event_tolerance=1e-8):
    """Differentiate tau=(b-a)/(c1-c2) without denominator clipping."""
    uL, uM, uR, a, b = parameters
    vL, vM, vR, va, vb = direction
    if not (uL > uM > uR and b > a):
        raise ValueError("collision reference requires uL>uM>uR and a<b")
    if not np.isfinite(event_tolerance) or event_tolerance < 0:
        raise ValueError("event tolerance must be finite and nonnegative")
    speed_difference = 0.5 * (uL - uR)
    speed_difference_jvp = 0.5 * (vL - vR)
    tau = (b - a) / speed_difference
    dtau = ((vb - va) * speed_difference - (b - a) * speed_difference_jvp) / speed_difference ** 2
    if not np.isfinite([tau, dtau]).all():
        raise ValueError("collision time or tangent is nonfinite; degenerate event is unsupported")
    distance = abs(float(time) - tau)
    exact_threshold = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(time), abs(tau))
    if distance <= exact_threshold:
        return CollisionEvent(tau, dtau, distance, "collision", "unresolved")
    phase = "pre_collision" if time < tau else "post_collision"
    return CollisionEvent(tau, dtau, distance, phase, "near_event" if distance <= event_tolerance else "regular")
