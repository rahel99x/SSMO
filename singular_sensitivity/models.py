"""Small initial-condition-only charts and their explicit measure derivatives.

The model receives only ``(u_left, u_right, initial_position, time)``.  A JVP
of its smooth outputs supplies region tangents and front motion; no hard
indicator is differentiated and no spatial Jacobian is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


class ChartModel(nn.Module):
    """One learned front and two constant regions, anchored at initial time.

    Inputs have shapes alpha[B,3] and time[B]; outputs are states[B,2] and
    positions[B,1]. A vanished initial jump collapses the base chart exactly.
    Front speed is learned from the four input coordinates; the exact Burgers
    speed is deliberately absent from this architecture.
    """

    def __init__(self, width: int = 32, depth: int = 2, domain=(-2.0, 2.0)):
        super().__init__()
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        self.width, self.depth = int(width), int(depth)
        self.domain = tuple(float(x) for x in domain)
        if len(self.domain) != 2 or self.domain[0] >= self.domain[1]:
            raise ValueError("domain must be an increasing pair")
        layers: list[nn.Module] = []
        channels = 4
        for _ in range(depth):
            layers.extend((nn.Linear(channels, width), nn.Tanh()))
            channels = width
        layers.append(nn.Linear(channels, 3))
        self.network = nn.Sequential(*layers)
        # Begin near the static initial chart rather than importing teacher
        # dynamics into the model. Small nonzero weights permit all layers to
        # receive gradients from the first update.
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, alpha: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        if alpha.ndim != 2 or alpha.shape[-1] != 3:
            raise ValueError("alpha must have shape [B,3]")
        if time.ndim == 0:
            time = time.expand(alpha.shape[0])
        if time.shape != alpha.shape[:1]:
            raise ValueError("time must have shape [B]")
        lo, hi = self.domain
        midpoint, length = (lo + hi) / 2.0, hi - lo
        features = torch.stack(
            (alpha[:, 0], alpha[:, 1], (alpha[:, 2] - midpoint) / length, time),
            dim=-1,
        )
        raw = self.network(features)
        jump = alpha[:, 0] - alpha[:, 1]
        states = alpha[:, :2] + (0.1 * time * jump)[:, None] * torch.tanh(raw[:, :2])
        positions = alpha[:, 2:3] + (2.0 * time)[:, None] * torch.tanh(raw[:, 2:3])
        return states, positions


@dataclass
class ChartMeasure:
    """A physical-domain signed measure with direction-independent support.

    Shapes: states[B,2], positions[B,1], diffuse[B,V,2], weights[B,V,1],
    position_jvps[B,V,1], atom_mask[B,1]. Boundary or exterior fronts are
    marked unresolved; query routines refuse to assign ordinary derivatives.
    Boundary front crossings are not a supported training family here.
    """

    states: Tensor
    positions: Tensor
    diffuse: Tensor
    weights: Tensor
    position_jvps: Tensor
    domain: tuple[float, float]
    atom_mask: Tensor
    status: str = "regular"

    @property
    def diffuse_coefficients(self) -> Tensor:
        return self.diffuse

    @property
    def atom_positions(self) -> Tensor:
        return self.positions

    @property
    def atom_weights(self) -> Tensor:
        return self.weights

    @property
    def left_traces(self) -> Tensor:
        return self.states[:, :1]

    @property
    def right_traces(self) -> Tensor:
        return self.states[:, 1:2]


def chart_measure(model: ChartModel, alpha: Tensor, time: Tensor, directions: Tensor) -> ChartMeasure:
    """Compute chart-output JVPs and explicitly assemble moving-front atoms."""
    if directions.ndim != 3 or directions.shape[0] != alpha.shape[0] or directions.shape[-1] != 3:
        raise ValueError("directions must have shape [B,V,3]")
    if directions.shape[1] < 1:
        raise ValueError("at least one direction is required")
    if time.ndim == 0:
        time = time.expand(alpha.shape[0])
    states, positions = model(alpha, time)
    region_tangents, front_tangents = [], []
    # A few physical parameter directions are cheaper than a dense Jacobian.
    # torch.func.jvp retains the mixed parameter/weight derivative graph.
    for direction in directions.unbind(dim=1):
        _, tangent = torch.func.jvp(model, (alpha, time), (direction, torch.zeros_like(time)))
        region_tangents.append(tangent[0])
        front_tangents.append(tangent[1])
    diffuse = torch.stack(region_tangents, dim=1)
    position_jvps = torch.stack(front_tangents, dim=1)
    weights = (states[:, 0] - states[:, 1])[:, None, None] * position_jvps
    lo, hi = model.domain
    atom_mask = (positions > lo) & (positions < hi)
    status = "regular" if bool(atom_mask.all()) else "unresolved"
    return ChartMeasure(states, positions, diffuse, weights, position_jvps, model.domain, atom_mask, status)


def _require_query_domain(query, domain) -> None:
    if tuple(float(x) for x in query.domain) != tuple(domain):
        raise ValueError("query and chart physical domains differ")


def _require_regular_support(positions: Tensor, domain) -> None:
    if not bool(((positions > domain[0]) & (positions < domain[1])).all()):
        raise ValueError("unresolved chart: front lies at or outside the supported physical domain")


def _region_integrals(positions: Tensor, domain: tuple[float, float], query) -> Tensor:
    support = positions[:, 0].clamp(min=domain[0], max=domain[1])
    return torch.stack(
        (query.torch_integral(domain[0], support), query.torch_integral(support, domain[1])), dim=-1
    )


def linear_pairings(measure: ChartMeasure, queries: Sequence) -> Tensor:
    """Return [B,V,Q] without a spatial grid or extra cell-width factor."""
    if not queries:
        raise ValueError("at least one query is required")
    _require_regular_support(measure.positions, measure.domain)
    pairings = []
    for query in queries:
        _require_query_domain(query, measure.domain)
        region = _region_integrals(measure.positions, measure.domain, query)
        bulk = (measure.diffuse * region[:, None, :]).sum(dim=-1)
        atomic = (measure.weights * query.torch_value(measure.positions)[:, None, :]
                  * measure.atom_mask[:, None, :]).sum(dim=-1)
        pairings.append(bulk + atomic)
    return torch.stack(pairings, dim=-1)


def chart_observables(model: ChartModel, alpha: Tensor, time: Tensor, queries: Sequence) -> Tensor:
    """Weak state values [B,Q] computed from the same smooth chart."""
    states, positions = model(alpha, time)
    _require_regular_support(positions, model.domain)
    for query in queries:
        _require_query_domain(query, model.domain)
    if not queries:
        raise ValueError("at least one query is required")
    return torch.stack(
        [(states * _region_integrals(positions, model.domain, query)).sum(dim=-1) for query in queries],
        dim=-1,
    )


def squared_objective_gradient(measure: ChartMeasure, target) -> Tensor:
    """Differentiate 0.5*integral((u-target(x))**2) using the payoff jump.

    ``target`` is a smooth Query or a scalar constant. The bulk term needs
    target integrals; the front term uses both base traces, never an arbitrary
    trace multiplied by the state-derivative atom.
    """
    _require_regular_support(measure.positions, measure.domain)
    lo, hi = measure.domain
    support = measure.positions[:, 0].clamp(min=lo, max=hi)
    lengths = torch.stack((support - lo, hi - support), dim=-1)
    if hasattr(target, "torch_integral") and hasattr(target, "torch_value"):
        _require_query_domain(target, measure.domain)
        target_integrals = _region_integrals(measure.positions, measure.domain, target)
        target_at_front = target.torch_value(measure.positions[:, 0])
    else:
        value = torch.as_tensor(target, dtype=measure.states.dtype, device=measure.states.device)
        if value.numel() != 1:
            raise ValueError("target must be a Query or scalar constant")
        target_integrals = value * lengths
        target_at_front = value.expand_as(support)
    bulk = (measure.diffuse *
            (measure.states * lengths - target_integrals)[:, None, :]).sum(dim=-1)
    left_payoff = 0.5 * (measure.states[:, 0] - target_at_front).square()
    right_payoff = 0.5 * (measure.states[:, 1] - target_at_front).square()
    front = ((left_payoff - right_payoff)[:, None] * measure.position_jvps[:, :, 0]
             * measure.atom_mask[:, 0, None])
    return bulk + front
