"""Smooth, normalized queries with analytic integrals on a physical interval.

Normalization bounds amplitude and Lipschitz constant in z=(x-x_min)/L,
not in dimensional x. Compact bumps are C2 at both support boundaries.
Torch is optional and imported only by the torch methods.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Query:
    kind: str
    domain: tuple[float, float]
    degree: int = 1
    frequency: float = 1.0
    center: float = 0.5
    width: float = 0.2

    def __post_init__(self):
        if self.kind not in {"constant", "polynomial", "sin", "cos", "bump"}:
            raise ValueError(f"unknown query kind: {self.kind}")
        if len(self.domain) != 2 or not np.isfinite(self.domain).all() or self.domain[1] <= self.domain[0]:
            raise ValueError("query domain must be a finite increasing interval")
        if not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        if not math.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError("frequency must be finite and positive")
        if not math.isfinite(self.center) or not math.isfinite(self.width) or self.width <= 0:
            raise ValueError("bump center and positive width must be finite")

    @property
    def normalization(self):
        if self.kind == "polynomial":
            return float(max(1, self.degree))
        if self.kind in {"sin", "cos"}:
            return max(1.0, 2.0 * math.pi * self.frequency)
        if self.kind == "bump":
            return max(1.0, (96.0 / (25.0 * math.sqrt(5.0))) / self.width)
        return 1.0

    def _value(self, x, backend):
        z = (x - self.domain[0]) / (self.domain[1] - self.domain[0])
        if self.kind == "constant":
            return z * 0.0 + 1.0
        if self.kind == "polynomial":
            return z ** self.degree / self.normalization
        if self.kind in {"sin", "cos"}:
            angle = 2.0 * math.pi * self.frequency * z
            return getattr(backend, self.kind)(angle) / self.normalization
        r = (z - self.center) / self.width
        if backend is np:
            r = np.clip(r, -1.0, 1.0)
        else:
            r = r.clamp(-1.0, 1.0)
        return (1.0 - r * r) ** 3 / self.normalization

    def value(self, x):
        """Evaluate a scalar or NumPy array; units follow the physical domain."""
        result = self._value(np.asarray(x, dtype=np.float64), np)
        return float(result) if result.ndim == 0 else result

    def torch_value(self, x):
        import torch
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float64)
        return self._value(x, torch)

    def _primitive(self, x, backend, moment=0, square=False):
        a, b = self.domain
        length = b - a
        z = (x - a) / length
        scale = self.normalization
        if self.kind in {"constant", "polynomial"}:
            degree = 0 if self.kind == "constant" else self.degree
            if square:
                return length * z ** (2 * degree + 1) / ((2 * degree + 1) * scale ** 2)
            basic = z ** (degree + 1) / (degree + 1)
            if moment:
                return length * (a * basic + length * z ** (degree + 2) / (degree + 2)) / scale
            return length * basic / scale
        if self.kind in {"sin", "cos"}:
            k = 2.0 * math.pi * self.frequency
            if square:
                sign = -1.0 if self.kind == "sin" else 1.0
                return length * (z / 2.0 + sign * backend.sin(2.0 * k * z) / (4.0 * k)) / scale ** 2
            if self.kind == "sin":
                basic = -backend.cos(k * z) / k
                first = -z * backend.cos(k * z) / k + backend.sin(k * z) / (k * k)
            else:
                basic = backend.sin(k * z) / k
                first = z * backend.sin(k * z) / k + backend.cos(k * z) / (k * k)
            return length * (a * basic + length * first) / scale if moment else length * basic / scale
        r = (z - self.center) / self.width
        r = np.clip(r, -1.0, 1.0) if backend is np else r.clamp(-1.0, 1.0)
        if square:
            primitive = r * 0.0
            for j in range(7):
                primitive = primitive + (-1.0) ** j * math.comb(6, j) * r ** (2 * j + 1) / (2 * j + 1)
            return length * self.width * primitive / scale ** 2
        primitive = r - r ** 3 + (3.0 / 5.0) * r ** 5 - r ** 7 / 7.0
        basic = length * self.width * primitive / scale
        if moment:
            first = r * r / 2.0 - 3.0 * r ** 4 / 4.0 + r ** 6 / 2.0 - r ** 8 / 8.0
            return (a + length * self.center) * basic + length ** 2 * self.width ** 2 * first / scale
        return basic

    def _integral_numpy(self, lo, hi, **kwargs):
        lower, upper = np.broadcast_arrays(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
        result = self._primitive(upper, np, **kwargs) - self._primitive(lower, np, **kwargs)
        return float(result) if result.ndim == 0 else result

    def _integral_torch(self, lo, hi, **kwargs):
        import torch
        if isinstance(lo, torch.Tensor):
            hi = torch.as_tensor(hi, dtype=lo.dtype, device=lo.device)
        elif isinstance(hi, torch.Tensor):
            lo = torch.as_tensor(lo, dtype=hi.dtype, device=hi.device)
        else:
            lo, hi = torch.as_tensor(lo, dtype=torch.float64), torch.as_tensor(hi, dtype=torch.float64)
        return self._primitive(hi, torch, **kwargs) - self._primitive(lo, torch, **kwargs)

    def integral(self, lo, hi):
        return self._integral_numpy(lo, hi)

    def first_moment(self, lo, hi):
        """Return integral x*query(x) dx analytically."""
        return self._integral_numpy(lo, hi, moment=1)

    def square_integral(self, lo, hi):
        return self._integral_numpy(lo, hi, square=True)

    def torch_integral(self, lo, hi):
        return self._integral_torch(lo, hi)

    def torch_first_moment(self, lo, hi):
        return self._integral_torch(lo, hi, moment=1)

    def torch_square_integral(self, lo, hi):
        return self._integral_torch(lo, hi, square=True)


def query_bank(domain, held_out=False):
    """Return independent training moments or held-out local C2 queries.

    All bank queries obey amplitude <=1 and Lip_z <=1 on z in [0,1].
    Finite bank accuracy is a diagnostic, not the full bounded-Lipschitz norm.
    """
    domain = tuple(domain)
    if held_out:
        return [Query("bump", domain, center=c, width=w)
                for c, w in ((0.19, 0.13), (0.43, 0.21), (0.73, 0.17), (0.88, 0.08))] + [
                    Query("sin", domain, frequency=2.7), Query("cos", domain, frequency=3.3)]
    return [Query("constant", domain)] + [Query("polynomial", domain, degree=d) for d in (1, 2, 3)] + [
        Query(kind, domain, frequency=f) for f in (1.0, 2.0) for kind in ("sin", "cos")]
