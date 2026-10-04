"""Scalar FP64 bookkeeping for frozen one-front records, using stdlib only.

The four signed terms use the exact support as anchor and the predicted
coefficients/weight for displacement. This deterministic telescoping order
reconstructs the error; it does not prove a unique causal attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys


RECONSTRUCTION_TOLERANCE = 1e-10


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} numbers")
    return tuple(finite(item, label) for item in value)


def nonnegative(value, label):
    result = finite(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def check_close(actual, expected, label):
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"{label} must remain finite")
    allowed = RECONSTRUCTION_TOLERANCE * max(1.0, abs(actual), abs(expected))
    if abs(actual - expected) > allowed:
        raise ValueError(f"{label} differs from the recorded frozen evidence")


@dataclass(frozen=True)
class ScalarQuery:
    kind: str
    domain: tuple
    degree: int = 1
    frequency: float = 1.0
    center: float = 0.5
    width: float = 0.2

    def __post_init__(self):
        if not isinstance(self.kind, str) or self.kind not in {"constant", "polynomial", "sin", "cos", "bump"}:
            raise ValueError(f"unsupported query kind: {self.kind}")
        domain = vector(self.domain, 2, "query domain")
        if domain[0] >= domain[1]:
            raise ValueError("query domain must be increasing")
        object.__setattr__(self, "domain", domain)
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("query degree must be a nonnegative integer")
        for name in ("frequency", "center", "width"):
            object.__setattr__(self, name, finite(getattr(self, name), f"query {name}"))
        if self.frequency <= 0 or self.width <= 0:
            raise ValueError("query frequency and width must be positive")

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, dict):
            raise ValueError("query must be an object")
        try:
            return cls(**record)
        except TypeError as error:
            raise ValueError("query fields do not match the frozen scalar schema") from error

    @property
    def normalization(self):
        if self.kind == "polynomial":
            return float(max(1, self.degree))
        if self.kind in {"sin", "cos"}:
            return max(1.0, 2.0 * math.pi * self.frequency)
        if self.kind == "bump":
            return max(1.0, (96.0 / (25.0 * math.sqrt(5.0))) / self.width)
        return 1.0

    def value(self, x):
        z = (finite(x, "query coordinate") - self.domain[0]) / (self.domain[1] - self.domain[0])
        if self.kind == "constant":
            return 1.0
        if self.kind == "polynomial":
            return z ** self.degree / self.normalization
        if self.kind in {"sin", "cos"}:
            return getattr(math, self.kind)(2.0 * math.pi * self.frequency * z) / self.normalization
        r = max(-1.0, min(1.0, (z - self.center) / self.width))
        return (1.0 - r * r) ** 3 / self.normalization

    def primitive(self, x):
        length = self.domain[1] - self.domain[0]
        z = (finite(x, "query coordinate") - self.domain[0]) / length
        scale = self.normalization
        if self.kind in {"constant", "polynomial"}:
            degree = 0 if self.kind == "constant" else self.degree
            return length * z ** (degree + 1) / ((degree + 1) * scale)
        if self.kind in {"sin", "cos"}:
            k = 2.0 * math.pi * self.frequency
            basic = -math.cos(k * z) / k if self.kind == "sin" else math.sin(k * z) / k
            return length * basic / scale
        r = max(-1.0, min(1.0, (z - self.center) / self.width))
        return length * self.width * (r - r ** 3 + 3.0 * r ** 5 / 5.0 - r ** 7 / 7.0) / scale

    def integral(self, lo, hi):
        # The boundary term needs an oriented integral when prediction lies left.
        return self.primitive(hi) - self.primitive(lo)


Query = ScalarQuery


def pairing(query, front):
    lo, hi = query.domain
    support = front["position"]
    left, right = front["diffuse"]
    return left * query.integral(lo, support) + right * query.integral(support, hi) + front["weight"] * query.value(support)


def nonlinear_gradient(query, front):
    """Squared tracking derivative with the payoff jump and base traces."""
    lo, hi = query.domain
    support = front["position"]
    left, right = front["states"]
    gleft, gright = front["diffuse"]
    bulk = gleft * (left * (support - lo) - query.integral(lo, support))
    bulk += gright * (right * (hi - support) - query.integral(support, hi))
    target = query.value(support)
    jump = 0.5 * ((left - target) ** 2 - (right - target) ** 2) * front["motion"]
    return bulk + jump


def payoff_jump(left, right, target):
    return 0.5 * ((left - target) ** 2 - (right - target) ** 2)


def attribute_row(row):
    if row.get("family") not in {"shock", "constant"} or row.get("status") != "regular":
        raise ValueError("attribution requires a regular shock/constant record; preserve other statuses")
    domain = vector(row["domain"], 2, "physical domain")
    if domain != (-2.0, 2.0):
        raise ValueError("attribution is frozen on the pilot physical domain [-2,2]")
    left, right, initial = vector(row["parameters"], 3, "initial parameters")
    vleft, vright, vinitial = vector(row["direction"], 3, "direction")
    time = finite(row["time"], "time")
    if time < 0 or not domain[0] < initial < domain[1] or left < right:
        raise ValueError("invalid regular entropy-shock parent")
    if row["family"] == "constant" and left != right:
        raise ValueError("constant parent must have equal states")
    support = initial + 0.5 * (left + right) * time
    motion = vinitial + 0.5 * time * (vleft + vright)
    reference = {"position": support, "motion": motion, "weight": (left - right) * motion,
                 "states": (left, right), "diffuse": (vleft, vright)}
    prediction = {"position": finite(row["predicted_position"], "predicted position"),
                  "motion": finite(row["predicted_motion"], "predicted motion"),
                  "weight": finite(row["predicted_atom_weight"], "predicted atom weight"),
                  "states": vector(row["predicted_states"], 2, "predicted states"),
                  "diffuse": vector(row["predicted_diffuse"], 2, "predicted diffuse")}
    pl, pr = prediction["states"]
    jump_tolerance = 256.0 * sys.float_info.epsilon * max(1.0, abs(pl), abs(pr))
    if pl < pr - jump_tolerance or not all(domain[0] < f["position"] < domain[1] for f in (reference, prediction)):
        raise ValueError("reference or predicted chart is inadmissible")
    expected_weight = 0.0 if abs(pl - pr) <= jump_tolerance else (pl - pr) * prediction["motion"]
    check_close(prediction["weight"], expected_weight, "predicted state/tangent atom weight")
    query = ScalarQuery.from_dict(row["query"])
    if query.domain != domain:
        raise ValueError("query and parent physical domains disagree")
    exact = finite(row["exact_value"], "recorded reference pairing")
    predicted = finite(row["value"], "recorded predicted pairing")
    reconstructed_exact = pairing(query, reference)
    reconstructed_prediction = pairing(query, prediction)
    check_close(reconstructed_exact, exact, "reference pairing reconstruction")
    check_close(reconstructed_prediction, predicted, "prediction pairing reconstruction")
    signed_error = predicted - exact
    check_close(abs(signed_error), nonnegative(row["absolute_error"], "absolute weak error"), "absolute weak error")
    gl, gr = prediction["diffuse"]
    terms = {
        "diffuse_coefficient": (gl - vleft) * query.integral(domain[0], support) + (gr - vright) * query.integral(support, domain[1]),
        "diffuse_support": (gl - gr) * query.integral(support, prediction["position"]),
        "atom_weight": (prediction["weight"] - reference["weight"]) * query.value(support),
        "atom_position": prediction["weight"] * (query.value(prediction["position"]) - query.value(support)),
    }
    total = math.fsum(terms.values())
    check_close(total, signed_error, "signed contribution reconstruction")
    exact_nonlinear = nonlinear_gradient(query, reference)
    predicted_nonlinear = nonlinear_gradient(query, prediction)
    nonlinear_error = predicted_nonlinear - exact_nonlinear
    recorded_nonlinear = nonnegative(row["nonlinear_gradient_error"], "recorded nonlinear error")
    check_close(abs(nonlinear_error), recorded_nonlinear, "nonlinear payoff-jump reconstruction")
    target_reference = query.value(support)
    target_prediction = query.value(prediction["position"])
    nonlinear_terms = {
        "diffuse_coefficient": (gl - vleft) * (left * (support - domain[0]) - query.integral(domain[0], support))
                              + (gr - vright) * (right * (domain[1] - support) - query.integral(support, domain[1])),
        "bulk_state_trace": gl * (pl - left) * (support - domain[0]) + gr * (pr - right) * (domain[1] - support),
        "diffuse_support": (gl * pl - gr * pr) * (prediction["position"] - support)
                           - (gl - gr) * query.integral(support, prediction["position"]),
        "jump_motion": (prediction["motion"] - motion) * payoff_jump(left, right, target_reference),
        "jump_state_trace": prediction["motion"] * (payoff_jump(pl, pr, target_reference) - payoff_jump(left, right, target_reference)),
        "jump_position": prediction["motion"] * (payoff_jump(pl, pr, target_prediction) - payoff_jump(pl, pr, target_reference)),
    }
    nonlinear_total = math.fsum(nonlinear_terms.values())
    check_close(nonlinear_total, nonlinear_error, "nonlinear signed contribution reconstruction")
    return {
        "signed_error": signed_error, "reconstructed_exact_value": reconstructed_exact,
        "reconstructed_prediction_value": reconstructed_prediction, "contributions": terms,
        "reconstruction_residual": total - signed_error,
        "reference_front": reference, "prediction_front": prediction,
        "nonlinear": {"reference_gradient": exact_nonlinear, "predicted_gradient": predicted_nonlinear,
                      "signed_error": nonlinear_error, "recorded_absolute_error": recorded_nonlinear,
                      "absolute_error_residual": abs(nonlinear_error) - recorded_nonlinear,
                      "contributions": nonlinear_terms, "reconstruction_residual": nonlinear_total - nonlinear_error},
    }
