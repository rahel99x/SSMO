"""A small independent conservative numerical teacher audit for Burgers.

Godunov finite volumes use exact initial cell averages, constant exterior
states, and deterministic FP64 with adaptive CFL <=0.45. Only a final state
is retained. Finite differences differentiate this discrete solve and remain
tier-2 numerical observations; refinement differences are not certified
uncertainties or evidence of an inviscid sensitivity plateau.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .queries import query_bank
from .reference import ParentInput, reference_state, reference_sensitivity
from .representation import PiecewiseState
from .runtime import atomic_write_json, confined_path


def godunov_flux(left, right):
    """Exact entropy Riemann flux for F(u)=u**2/2, vectorized in FP64."""
    left, right = np.broadcast_arrays(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    fleft, fright = 0.5 * left * left, 0.5 * right * right
    rarefaction = np.where(left >= 0, fleft, np.where(right <= 0, fright, 0.0))
    shock = np.where(left + right >= 0, fleft, fright)
    return np.where(left <= right, rarefaction, shock)


def _godunov_run(parent, resolution, cfl=0.45):
    if parent.family not in {"shock", "rarefaction", "constant"} or len(parent.parameters) != 3:
        raise ValueError("Numerical teacher supports Burgers Riemann shock/rarefaction/constant inputs only")
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 4:
        raise ValueError("Finite-volume resolution must be an integer >=4")
    if not np.isfinite(cfl) or not 0 < cfl <= 0.45:
        raise ValueError("Conservative audit requires 0<CFL<=0.45")
    # Establish the no-boundary-interaction scope independently of FV evolution.
    reference_state(parent)
    started = time.perf_counter()
    uleft, uright, location = parent.parameters
    edges = np.linspace(*parent.domain, resolution + 1, dtype=np.float64)
    dx = (parent.domain[1] - parent.domain[0]) / resolution
    left_fraction = np.clip((location - edges[:-1]) / dx, 0.0, 1.0)
    values = uleft * left_fraction + uright * (1.0 - left_fraction)
    initial_mass = float(np.sum(values) * dx)
    boundary_mass = 0.0
    elapsed, steps = 0.0, 0
    lower, upper = min(uleft, uright), max(uleft, uright)
    max_range_violation = 0.0
    while elapsed < parent.time:
        max_speed = max(float(np.max(np.abs(values))), abs(uleft), abs(uright))
        remaining = parent.time - elapsed
        dt = min(remaining, cfl * dx / max_speed) if max_speed > 0 else remaining
        if dt <= 0 or elapsed + dt == elapsed:
            raise RuntimeError("Floating-point timestep stopped progressing")
        extended = np.r_[uleft, values, uright]
        flux = godunov_flux(extended[:-1], extended[1:])
        values = values - (dt / dx) * np.diff(flux)
        boundary_mass += dt * float(flux[0] - flux[-1])
        elapsed = parent.time if dt == remaining else elapsed + dt
        steps += 1
        max_range_violation = max(max_range_violation, float(np.max(values) - upper), float(lower - np.min(values)))
        if not np.isfinite(values).all() or max_range_violation > 1e-11 * max(1.0, abs(lower), abs(upper)):
            raise RuntimeError("Monotone numerical teacher violated finiteness or the maximum principle")
    final_mass = float(np.sum(values) * dx)
    coefficients = np.column_stack((values, np.zeros(resolution)))
    state = PiecewiseState(edges, coefficients)
    record = {"scheme": "first-order conservative Godunov finite volume", "flux": "Burgers u^2/2",
              "boundary_conditions": "fixed constant exterior states", "initialization": "exact initial cell averages",
              "precision": "float64", "resolution": resolution, "dx": dx, "cfl": cfl, "steps": steps,
              "initial_mass": initial_mass, "final_mass": final_mass, "integrated_boundary_mass": boundary_mass,
              "conservation_residual": final_mass - initial_mass - boundary_mass,
              "max_range_violation": max_range_violation, "minimum": float(np.min(values)), "maximum": float(np.max(values)),
              "wall_seconds": time.perf_counter() - started, "reference_tier": 2,
              "derivative_claim": "no atoms or inviscid tangents inferred from numerical cell interfaces"}
    return state, record


def godunov_state(parent, resolution, cfl=0.45):
    """Return only the final piecewise-constant finite-volume state.

    This chart's cell boundaries are numerical cells, not physical shock
    supports. Do not assemble derivative atoms from them.
    """
    return _godunov_run(parent, resolution, cfl)[0]


def _perturbed_entropy_parent(parent, direction, step):
    parameters = tuple(np.asarray(parent.parameters) + step * np.asarray(direction))
    family = "shock" if parameters[0] >= parameters[1] else "rarefaction"
    return ParentInput(parent.parent_id, family, parameters, parent.time, parent.domain)


def _atomic_jsonl(path, records):
    destination = confined_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def numerical_audit(config: dict, run_dir: Path):
    """Write a bounded CPU audit, raw forward/FD errors, and explicit gates.

    Defaults are declared protocol tolerances, not fitted error estimates.
    Passing these small checks permits this implementation's pilot use; it
    does not make numerical directional labels inviscid ground truth.
    """
    started = time.perf_counter()
    run_dir = confined_path(run_dir)
    options = config.get("numerical", {})
    resolutions = options.get("resolutions", [128, 256, 512])
    steps = options.get("fd_steps", [0.05, 0.02, 0.01])
    if (not isinstance(resolutions, list) or not 2 <= len(resolutions) <= 4 or
            any(not isinstance(n, int) or isinstance(n, bool) or not 16 <= n <= 2048 for n in resolutions) or
            resolutions != sorted(set(resolutions))):
        raise ValueError("Numerical audit requires 2-4 distinct increasing integer resolutions in [16,2048]")
    if (not isinstance(steps, list) or not 2 <= len(steps) <= 4 or
            any(not np.isfinite(h) or not 1e-4 <= h <= 0.1 for h in steps) or
            steps != sorted(set(steps), reverse=True)):
        raise ValueError("Numerical audit requires 2-4 distinct decreasing FD steps in [1e-4,.1]")
    cfl = float(options.get("cfl", 0.45))
    forward_tolerance = float(options.get("forward_tolerance", 0.02))
    fd_tolerance = float(options.get("fd_tolerance", 0.03))
    conservation_tolerance = float(options.get("conservation_tolerance", 1e-11))
    if not np.isfinite([forward_tolerance, fd_tolerance, conservation_tolerance]).all() or min(forward_tolerance, fd_tolerance, conservation_tolerance) <= 0:
        raise ValueError("Numerical audit tolerances must be finite and positive")
    domain = tuple(config.get("domain", [-2.0, 2.0]))
    parents = [ParentInput("numeric-shock", "shock", (2.0, -0.5, 0.1), 0.4, domain),
               ParentInput("numeric-rarefaction", "rarefaction", (-0.8, 1.2, 0.1), 0.4, domain)]
    directions = [[0.0, 0.0, 1.0], [0.2, -0.3, 0.4], [0.5, 0.1, -0.2]]
    train_queries, novel_queries = query_bank(domain), query_bank(domain, held_out=True)
    queries = [(f"training-{i}", q) for i, q in enumerate(train_queries[:3])] + [
        (f"held-out-{i}", novel_queries[i]) for i in (0, 1, len(novel_queries) - 1)]
    records, solver_records, summaries = [], [], []
    for parent in parents:
        exact_state = reference_state(parent)
        forward_by_resolution, fd_by_resolution = {}, {}
        previous_fd = {}
        for n in resolutions:
            state, solver = _godunov_run(parent, n, cfl)
            solver_records.append(solver)
            records.append({"record_type": "forward_state", "parent": asdict(parent), "solver": solver,
                            "final_cell_averages": state.coefficients[:, 0].tolist(),
                            "retention": "final cells only; no time trajectory"})
            query_errors = []
            for query_id, query in queries:
                exact, numerical = exact_state.linear_query(query), state.linear_query(query)
                query_errors.append(abs(numerical - exact))
                records.append({"record_type": "forward_query", "parent_id": parent.parent_id, "family": parent.family,
                                "resolution": n, "query_id": query_id, "query": asdict(query),
                                "exact_query": exact, "numerical_query": numerical,
                                "signed_error": numerical - exact, "absolute_error": abs(numerical - exact)})
            forward_by_resolution[str(n)] = max(query_errors)
            fd_errors = {str(h): [] for h in steps}
            for direction_index, direction in enumerate(directions):
                reference = reference_sensitivity(parent, direction)
                previous_h_values = {}
                for h in steps:
                    plus_parent, minus_parent = _perturbed_entropy_parent(parent, direction, h), _perturbed_entropy_parent(parent, direction, -h)
                    plus, plus_solver = _godunov_run(plus_parent, n, cfl)
                    minus, minus_solver = _godunov_run(minus_parent, n, cfl)
                    solver_records.extend([plus_solver, minus_solver])
                    exact_plus, exact_minus = reference_state(plus_parent), reference_state(minus_parent)
                    for query_id, query in queries:
                        p, m = plus.linear_query(query), minus.linear_query(query)
                        ep, em = exact_plus.linear_query(query), exact_minus.linear_query(query)
                        numerical_fd, exact_fd = (p - m) / (2.0 * h), (ep - em) / (2.0 * h)
                        derivative = reference.linear_pairing(query)
                        error = abs(numerical_fd - derivative)
                        fd_errors[str(h)].append(error)
                        key = (direction_index, h, query_id)
                        previous = previous_fd.get(key)
                        previous_h = previous_h_values.get(query_id)
                        records.append({"record_type": "finite_difference", "parent_id": parent.parent_id,
                            "family": parent.family, "parameters": list(parent.parameters), "time": parent.time,
                            "resolution": n, "h": h, "direction_index": direction_index, "direction": direction,
                            "direction_norm": float(np.linalg.norm(direction)), "query_id": query_id,
                            "query": asdict(query), "reference_tier": 2, "derivative_status": "numerical_fd_observation",
                            "reference_derivative": derivative, "numerical_fd": numerical_fd, "exact_fd": exact_fd,
                            "absolute_error": error, "signed_error": numerical_fd - derivative,
                            "finite_step_bias": abs(exact_fd - derivative),
                            "discrete_forward_difference_error": abs(numerical_fd - exact_fd),
                            "forward_plus_signed_error": p - ep, "forward_minus_signed_error": m - em,
                            "observed_amplified_forward_error": (abs(p - ep) + abs(m - em)) / (2 * h),
                            "resolution_difference": None if previous is None else numerical_fd - previous,
                            "h_difference": None if previous_h is None else numerical_fd - previous_h,
                            "uncertainty_claim": "observed errors and differences only; not a certified error bound"})
                        previous_fd[key], previous_h_values[query_id] = numerical_fd, numerical_fd
            fd_by_resolution[str(n)] = {h: max(errors) for h, errors in fd_errors.items()}
        forward_values = list(forward_by_resolution.values())
        decreasing = all(b <= a + 1e-13 for a, b in zip(forward_values[:-1], forward_values[1:]))
        finest_forward = forward_values[-1]
        finest_fd = max(fd_by_resolution[str(resolutions[-1])].values())
        fd_maxima = [max(fd_by_resolution[str(n)].values()) for n in resolutions]
        summaries.append({"family": parent.family, "parent": asdict(parent),
            "max_forward_query_absolute_error_by_resolution": forward_by_resolution,
            "max_fd_absolute_error_by_resolution_and_h": fd_by_resolution,
            "forward_errors_decrease_over_selected_meshes": decreasing,
            "fd_max_errors_decrease_over_selected_meshes": all(b <= a + 1e-13 for a, b in zip(fd_maxima[:-1], fd_maxima[1:])),
            "finest_forward_error": finest_forward, "finest_fd_error_all_steps": finest_fd,
            "science_gate_status": "passed" if decreasing and finest_forward <= forward_tolerance and finest_fd <= fd_tolerance else "failed"})
    max_conservation = max(abs(s["conservation_residual"]) for s in solver_records)
    max_range = max(s["max_range_violation"] for s in solver_records)
    setup_passed = max_conservation <= conservation_tolerance and max_range <= 1e-11
    science_passed = setup_passed and all(case["science_gate_status"] == "passed" for case in summaries)
    raw_path, summary_path = run_dir / "numeric_teacher_audit.jsonl", run_dir / "numeric_teacher_audit.json"
    summary = {"schema_version": 1, "status": "passed" if science_passed else "failed",
        "setup_gate_status": "passed" if setup_passed else "failed", "science_gate_status": "passed" if science_passed else "failed",
        "protocol": {"reference_tier": 2, "scheme": "first-order Godunov", "precision": "float64", "cfl": cfl,
            "resolutions": resolutions, "fd_steps": steps, "directions": directions,
            "query_count": len(queries), "domain": domain, "observation_time": 0.4,
            "required_gates": "conservative monotone finite solve; declared finest-mesh observable tolerances; decreasing forward errors",
            "forward_absolute_tolerance": forward_tolerance, "fd_absolute_tolerance": fd_tolerance,
            "conservation_absolute_tolerance": conservation_tolerance},
        "cases": summaries, "solve_count": len(solver_records), "raw_record_count": len(records),
        "max_conservation_residual": max_conservation, "max_range_violation": max_range,
        "raw_path": str(raw_path), "wall_seconds": time.perf_counter() - started,
        "limitations": ["No convergence plateau is claimed from these three meshes or perturbations.",
            "Numerical FD errors are observed against analytic formulas, not certified bounds.",
            "No numerical labels are promoted to inviscid training ground truth.",
            "Boundary interactions, events, arbitrary initial fields, and larger numerical training campaigns remain deferred."]}
    _atomic_jsonl(raw_path, records)
    summary["wall_seconds"] = time.perf_counter() - started
    atomic_write_json(summary_path, summary)
    return summary
