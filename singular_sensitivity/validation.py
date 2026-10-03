"""Bounded, auditable weak diagnostics for the exact single-shock pilot.

All raw differences survive in JSONL. Statistical units are physical parents;
query counts, times, and directions never inflate the number of independent
examples. Numerical LP diagnostics are explicitly not rigorous certificates.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .baselines import FrontPrediction, density_cell_masses, fit_baselines, smoothed_shock
from .queries import query_bank
from .reference import ParentInput, reference_state, reference_sensitivity


def bounded_lipschitz_lp(positions, masses, domain, discretization_radius_bound=0.0):
    """Exact discrete BL formulation, solved numerically on normalized coordinates.

    A separately justified diffuse-cell relocation bound can be added to the
    numerical discrete value. It does not turn floating-point LP output into a
    certificate. Coordinates are normalized; physical atom masses are retained.
    """
    from scipy.optimize import linprog
    positions, masses = np.asarray(positions, dtype=np.float64), np.asarray(masses, dtype=np.float64)
    lo, hi = map(float, domain)
    if hi <= lo or positions.shape != masses.shape or positions.ndim != 1:
        raise ValueError("invalid signed support or physical domain")
    if not np.isfinite(np.r_[positions, masses, lo, hi, discretization_radius_bound]).all():
        raise ValueError("BL inputs must be finite")
    if np.any(positions < lo) or np.any(positions > hi) or discretization_radius_bound < 0:
        raise ValueError("support outside domain or negative relocation bound")
    support, inverse = np.unique(positions, return_inverse=True)
    combined = np.bincount(inverse, weights=masses, minlength=len(support))
    keep = combined != 0.0
    support, combined = support[keep], combined[keep]
    if not len(support):
        return {"status": "exact_zero_discrete_measure", "value": 0.0, "primal_feasibility": 0.0,
                "dual_feasibility": 0.0, "complementarity": 0.0, "duality_gap": 0.0,
                "residual_checks_passed": True, "discretization_radius_bound": float(discretization_radius_bound),
                "value_plus_discretization_bound": float(discretization_radius_bound), "rigorous_certificate": False}
    coordinates = (support - lo) / (hi - lo)
    count = len(support)
    constraints = np.zeros((2 * max(0, count - 1), count))
    gaps = np.repeat(np.diff(coordinates), 2)
    for index in range(count - 1):
        constraints[2 * index, index:index + 2] = (-1, 1)
        constraints[2 * index + 1, index:index + 2] = (1, -1)
    solved = linprog(-combined, A_ub=constraints if len(gaps) else None,
                     b_ub=gaps if len(gaps) else None, bounds=(-1.0, 1.0), method="highs",
                     options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    report = {"status": str(solved.message), "solver_status": int(solved.status),
              "support_count": count, "normalized_domain": [0.0, 1.0], "rigorous_certificate": False,
              "discretization_radius_bound": float(discretization_radius_bound)}
    if not solved.success:
        return report | {"value": None, "residual_checks_passed": False}
    x = solved.x
    inequalities = np.asarray(solved.ineqlin.marginals)
    lower, upper = np.asarray(solved.lower.marginals), np.asarray(solved.upper.marginals)
    slack = gaps - constraints @ x
    primal = max(float(np.max(np.abs(x) - 1.0)), float(np.max(-slack, initial=0.0)), 0.0)
    stationarity = -combined - constraints.T @ inequalities - lower - upper
    dual = max(float(np.max(np.abs(stationarity))), float(np.max(inequalities, initial=0.0)),
               float(np.max(-lower)), float(np.max(upper)), 0.0)
    complementarity = max(float(np.max(np.abs(inequalities * slack), initial=0.0)),
                          float(np.max(np.abs(lower * (x + 1.0)))), float(np.max(np.abs(upper * (1.0 - x)))))
    dual_objective = float(gaps @ inequalities - lower.sum() + upper.sum())
    gap = abs(float(solved.fun) - dual_objective)
    value = max(0.0, -float(solved.fun))
    tolerance = 1e-7 * max(1.0, float(np.abs(combined).sum()))
    return report | {"value": value, "primal_feasibility": primal, "dual_feasibility": dual,
                     "complementarity": complementarity, "duality_gap": gap,
                     "residual_tolerance": tolerance,
                     "residual_checks_passed": max(primal, dual, complementarity, gap) <= tolerance,
                     "value_plus_discretization_bound": value + float(discretization_radius_bound)}


def _components(measure):
    if isinstance(measure, FrontPrediction):
        return (np.asarray([*measure.domain[:1], measure.position, measure.domain[1]]), measure.diffuse,
                np.asarray([measure.position]), np.asarray([measure.weight]), measure.domain)
    return (measure.state.edges, measure.diffuse_density, measure.atom_positions,
            measure.atom_weights, measure.parent.domain)


def discretize_measure(measure, resolution=128):
    """Exact diffuse-cell masses plus unmodified atoms and a TV radius bound."""
    edges, density, positions, weights, domain = _components(measure)
    cell_edges = np.linspace(*domain, resolution + 1)
    centers = 0.5 * (cell_edges[1:] + cell_edges[:-1])
    masses, variation = density_cell_masses(edges, density, cell_edges)
    radius = np.diff(cell_edges) / (2.0 * (domain[1] - domain[0]))
    return np.r_[centers, positions], np.r_[masses, weights], float(radius @ variation)


def measure_bl_diagnostic(reference, prediction, resolution=128):
    x, m, radius = discretize_measure(reference, resolution)
    y, n, other_radius = discretize_measure(prediction, resolution)
    return bounded_lipschitz_lp(np.r_[x, y], np.r_[m, -n], reference.parent.domain, radius + other_radius)


def _perturbed(parent, direction, step):
    parameters = np.asarray(parent.parameters) + step * np.asarray(direction)
    family = parent.family
    if family in {"constant", "shock", "rarefaction"}:
        family = "shock" if parameters[0] >= parameters[1] else "rarefaction"
    return ParentInput(parent.parent_id, family, tuple(parameters), parent.time, parent.domain)


def representation_rows(parent, direction, queries, resolutions=(32, 64, 128),
                        fd_steps=(1e-2, 1e-3, 1e-4), smoothing_widths=(0.16, 0.08, 0.04)):
    """Coupled n/h ladders, exact forward controls, and physical smoothing ladders."""
    direction = np.asarray(direction, dtype=np.float64)
    reference = reference_sensitivity(parent, direction)
    base = {"parent_id": parent.parent_id, "family": parent.family, "time": parent.time,
            "direction": direction.tolist(), "direction_norm": float(np.linalg.norm(direction)),
            "status": reference.status, "event_distance": reference.event_metadata.get("event_distance"),
            "reference_tier": "exact_analytic_burgers"}
    if reference.status == "unresolved":
        return [base | {"method": "representation_audit", "status": "unresolved", "reason": "no validated derivative at exact event"}]
    rows = []
    lo, hi = parent.domain
    for step in fd_steps:
        if step <= 0:
            raise ValueError("finite difference steps must be positive")
        try:
            plus_parent, minus_parent = _perturbed(parent, direction, step), _perturbed(parent, direction, -step)
            plus, minus = reference_state(plus_parent), reference_state(minus_parent)
            phases = [reference_sensitivity(p, np.zeros(len(direction))).event_metadata.get("phase")
                      for p in (plus_parent, minus_parent)]
        except ValueError as error:
            rows.append(base | {"method": "grid_finite_difference", "h": float(step),
                                "status": "unsupported_perturbation", "reason": str(error)})
            continue
        for resolution in resolutions:
            dx = (hi - lo) / resolution
            x = lo + (np.arange(resolution) + 0.5) * dx
            cell_plus, cell_minus = plus.cell_averages(resolution), minus.cell_averages(resolution)
            cell_density = (cell_plus - cell_minus) / (2 * step)
            point_density = (plus.evaluate(x) - minus.evaluate(x)) / (2 * step)
            for query_index, query in enumerate(queries):
                exact = reference.linear_pairing(query)
                exact_plus, exact_minus = plus.linear_query(query), minus.linear_query(query)
                exact_fd = (exact_plus - exact_minus) / (2 * step)
                qx = query.value(x)
                quadrature_plus, quadrature_minus = dx * float(qx @ cell_plus), dx * float(qx @ cell_minus)
                quadrature_error_bound = (abs(quadrature_plus - exact_plus) + abs(quadrature_minus - exact_minus)) / (2 * step)
                common = base | {"query_index": query_index, "query": asdict(query), "resolution": int(resolution),
                                 "dx": dx, "h": float(step), "dx_over_h": dx / step,
                                 "event_straddled": phases[0] != phases[1], "exact_value": exact,
                                 "exact_forward_fd": exact_fd, "exact_forward_fd_error": abs(exact_fd - exact),
                                 "reference_uncertainty": "analytic FP64, no rigorous roundoff enclosure"}
                for name, density in (("grid_cell_average_fd", cell_density), ("grid_point_sample_fd", point_density)):
                    value = dx * float(qx @ density)
                    rows.append(common | {"method": name, "value": value, "absolute_error": abs(value - exact),
                                          "density_l2": float(np.sqrt(dx * (density @ density))),
                                          "density_linf": float(np.max(np.abs(density))),
                                          "forward_quadrature_quotient_error_bound": quadrature_error_bound if name == "grid_cell_average_fd" else None})
    if parent.family in {"shock", "constant"}:
        for width in smoothing_widths:
            for resolution in resolutions:
                x, state, derivative = smoothed_shock(parent.parameters, parent.time, direction, parent.domain, resolution, width)
                dx = (hi - lo) / resolution
                for query_index, query in enumerate(queries):
                    value, exact = dx * float(query.value(x) @ derivative), reference.linear_pairing(query)
                    rows.append(base | {"method": "fixed_physical_tanh_regularization", "query_index": query_index,
                                        "query": asdict(query), "resolution": int(resolution), "dx": dx,
                                        "physical_width": float(width), "dx_over_width": dx / width,
                                        "value": value, "exact_value": exact, "absolute_error": abs(value - exact),
                                        "regularization": "smoothed chart derivative; not a viscous Burgers solve",
                                        "inviscid_limit_established": False})
    return rows


def representation_study(config, run_dir):
    """Standalone WP2 audit on fixed analytic controls; writes fresh artifacts."""
    from .runtime import atomic_write_json, confined_path
    started = time.perf_counter()
    run_dir = confined_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output = confined_path(run_dir / "representation.jsonl")
    summary_path = confined_path(run_dir / "representation.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("representation artifacts already exist; use a fresh run directory")
    domain = tuple(config.get("domain", (-2.0, 2.0)))
    queries = query_bank(domain, held_out=True)
    options = config.get("evaluation", {})
    examples = [ParentInput("wp2-single-shock", "shock", (2.0, -0.5, 0.1), 0.6, domain),
                ParentInput("wp2-constant", "constant", (0.2, 0.2, 0.0), 0.4, domain),
                ParentInput("wp2-rarefaction", "rarefaction", (-0.5, 0.7, 0.0), 0.4, domain),
                ParentInput("wp2-pre-collision", "collision", (1.0, 0.0, -1.0, -0.4, 0.4), 0.4, domain),
                ParentInput("wp2-exact-collision", "collision", (1.0, 0.0, -1.0, -0.4, 0.4), 0.8, domain),
                ParentInput("wp2-post-collision", "collision", (1.0, 0.0, -1.0, -0.4, 0.4), 1.1, domain)]
    count, errors, unresolved = 0, {}, 0
    with output.open("x") as stream:
        for parent in examples:
            direction = np.asarray((0.2, -0.3, 0.4) if len(parent.parameters) == 3 else (0.2, -0.1, -0.3, 0.3, -0.2))
            direction /= np.linalg.norm(direction)
            for row in representation_rows(parent, direction, queries,
                    options.get("resolutions", [32, 64, 128]), options.get("fd_steps", [0.01, 0.001, 0.0001]),
                    options.get("smoothing_widths", [0.16, 0.08, 0.04])):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                count += 1
                unresolved += row.get("status") == "unresolved"
                if "absolute_error" in row:
                    errors.setdefault(row["method"], []).append(row["absolute_error"])
    report = {"status": "completed_scoped_representation_audit", "physical_parents": len(examples), "raw_rows": count,
              "unresolved_rows": unresolved, "raw_artifact": str(output), "wall_seconds": time.perf_counter() - started,
              "diagnostic_errors": {m: {"mean": float(np.mean(e)), "worst": float(np.max(e))} for m, e in errors.items()},
              "limitations": "n/h ladder summaries are diagnostics within fixed controls, not independent-parent statistics or learned-method gains; smoothing is a regularized chart, not a viscous PDE solve"}
    atomic_write_json(summary_path, report)
    return report


def _piecewise_l1(edges1, values1, edges2, values2):
    partition = np.unique(np.r_[edges1, edges2])
    midpoints = 0.5 * (partition[1:] + partition[:-1])
    first = np.asarray(values1)[np.searchsorted(edges1, midpoints, side="right") - 1]
    second = np.asarray(values2)[np.searchsorted(edges2, midpoints, side="right") - 1]
    return float(np.diff(partition) @ np.abs(first - second))


def front_errors(reference, prediction):
    """Exact piecewise-constant errors; used only for learned shock/constant charts."""
    predicted_edges = np.asarray([reference.parent.domain[0], prediction.position, reference.parent.domain[1]])
    return {"state_l1_error": _piecewise_l1(reference.state.edges, reference.state.coefficients[:, 0], predicted_edges, prediction.states),
            "diffuse_l1_error": _piecewise_l1(reference.state.edges, reference.diffuse_density, predicted_edges, prediction.diffuse),
            "support_error": float(abs(reference.atom_positions[0] - prediction.position)) if len(reference.atom_positions) else None,
            "weight_error": float(abs(reference.atom_weights.sum() - prediction.weight)),
            "predicted_total_variation": float(np.diff(predicted_edges) @ np.abs(prediction.diffuse) + abs(prediction.weight)),
            "reference_total_variation": float(np.diff(reference.state.edges) @ np.abs(reference.diffuse_density)
                                               + np.abs(reference.atom_weights).sum()),
            "predicted_position": prediction.position, "predicted_atom_weight": prediction.weight,
            "predicted_motion": prediction.motion, "predicted_states": prediction.states.tolist(),
            "predicted_diffuse": prediction.diffuse.tolist()}


def _model_predictions(model, alpha, observation_time, directions, device):
    import torch
    from .models import chart_measure
    dtype = next(model.parameters()).dtype
    a = torch.as_tensor(np.asarray(alpha)[None, :], dtype=dtype, device=device)
    t = torch.as_tensor([observation_time], dtype=dtype, device=device)
    v = torch.as_tensor(np.asarray(directions)[None, :, :], dtype=dtype, device=device)
    measure = chart_measure(model, a, t, v)
    states = measure.states[0].detach().cpu().numpy().astype(np.float64)
    position = float(measure.positions[0, 0].detach().cpu())
    diffuse = measure.diffuse[0].detach().cpu().numpy().astype(np.float64)
    motions = measure.position_jvps[0, :, 0].detach().cpu().numpy().astype(np.float64)
    return [FrontPrediction(states, position, g, float(motion), model.domain) for g, motion in zip(diffuse, motions)]


def _model_prediction(model, alpha, observation_time, direction, device):
    return _model_predictions(model, alpha, observation_time, [direction], device)[0]


def _model_state(model, alpha, observation_time, device):
    import torch
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        states, position = model(torch.as_tensor(np.asarray(alpha)[None, :], dtype=dtype, device=device),
                                 torch.as_tensor([observation_time], dtype=dtype, device=device))
    return states.detach().cpu().numpy(), position.detach().cpu().numpy()


def _exact_front(alpha, observation_time, direction, domain):
    alpha, direction = np.asarray(alpha), np.asarray(direction)
    return FrontPrediction(alpha[:2], float(alpha[2] + observation_time * (alpha[0] + alpha[1]) / 2),
                           direction[:2], float(direction[2] + observation_time * (direction[0] + direction[1]) / 2), tuple(domain))


def _timed(function, repeats=3, warmup=1, device="cpu"):
    def synchronize():
        if str(device).startswith("cuda"):
            import torch
            torch.cuda.synchronize()
    synchronize()
    started = time.perf_counter()
    function()
    synchronize()
    cold = time.perf_counter() - started
    for _ in range(warmup):
        function()
    durations = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        function()
        synchronize()
        durations.append(time.perf_counter() - started)
    return {"cold_call_seconds": cold, "seconds": durations, "median_seconds": float(np.median(durations)),
            "warmup": warmup, "repeats": repeats, "cuda_synchronized": str(device).startswith("cuda")}


def _parent_statistics(records):
    metrics = ("absolute_error", "relative_error", "nonlinear_gradient_error", "state_l1_error",
               "diffuse_l1_error", "support_error", "weight_error", "predicted_total_variation")
    result = {}
    for metric in metrics:
        values = [r[metric] for r in records if r.get(metric) is not None and np.isfinite(r[metric])]
        if values:
            result[metric + "_mean"] = float(np.mean(values))
            result[metric + "_max"] = float(np.max(values))
    return result


def _aggregate(parents):
    keys = sorted({(p["method"], p["split"], p["family"]) for p in parents})
    summaries = []
    for method, split, family in keys:
        group = [p for p in parents if (p["method"], p["split"], p["family"]) == (method, split, family)]
        fields = set().union(*(p["metrics"] for p in group))
        metrics = {}
        for name in sorted(fields):
            values = [p["metrics"][name] for p in group if name in p["metrics"]]
            if values:
                metrics[name] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                                 "q90": float(np.quantile(values, 0.9)), "worst": float(np.max(values)),
                                 "physical_parents": len(values)}
        strata = {}
        for stratum in ("zero_direction", "near_event", "zero_reference_observable"):
            stratified = [p.get("strata", {}).get(stratum, {}) for p in group]
            names = set().union(*stratified)
            strata[stratum] = {}
            for name in sorted(names):
                values = [p[name] for p in stratified if name in p]
                if values:
                    strata[stratum][name] = {"median": float(np.median(values)), "q90": float(np.quantile(values, 0.9)),
                                             "worst": float(np.max(values)), "physical_parents": len(values)}
        summaries.append({"method": method, "split": split, "family": family,
                          "physical_parents": len(group), "metrics": metrics, "control_strata": strata,
                          "accuracy_gate_parents_passed": sum(p["accuracy_gate"]["passed"] for p in group),
                          "accuracy_gate_parents_failed": sum(not p["accuracy_gate"]["passed"] for p in group),
                          "invalid_chart_rows": sum(p["invalid_chart_rows"] for p in group)})
    return summaries


def evaluate(config: dict, dataset: dict, run_dir: Path, checkpoint: Path | None = None, device="cpu",
             state_only_checkpoint: Path | None = None):
    """Run bounded independent-parent/query audits; return and save JSON summary."""
    from .inverse import inverse_tracking
    from .runtime import atomic_write_json, confined_path, content_hash, memory_record
    started = time.perf_counter()
    run_dir = confined_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: confined_path(run_dir / name) for name in (
        "query_errors.jsonl", "representation.jsonl", "evaluation_parents.jsonl", "evaluation.json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("evaluation artifacts already exist; use a fresh run directory")
    evaluation = config.get("evaluation", config.get("eval", {}))
    max_parents = int(evaluation.get("max_parents", 16))
    lp_parents = int(evaluation.get("lp_parents", 4))
    resolutions = tuple(int(n) for n in evaluation.get("resolutions", [32, 64, 128]))
    steps = tuple(float(h) for h in evaluation.get("fd_steps", [1e-2, 1e-3, 1e-4]))
    widths = tuple(float(w) for w in evaluation.get("smoothing_widths", [0.16, 0.08, 0.04]))
    if max_parents < 1 or lp_parents < 0 or not resolutions or any(n < 2 for n in resolutions):
        raise ValueError("invalid bounded evaluation workload")
    domain = tuple(dataset["domain"])
    if tuple(config.get("domain", domain)) != domain:
        raise ValueError("configuration physical domain disagrees with dataset")
    queries = query_bank(domain, held_out=True)
    models, checkpoint_records = {}, {}
    candidates = [checkpoint, state_only_checkpoint or evaluation.get("state_only_checkpoint")]
    for path in candidates:
        if path is None:
            continue
        from .training import load_model
        path = confined_path(path)
        model, payload = load_model(path, device=device)
        if payload.get("dataset_hash") != content_hash(dataset):
            raise ValueError("checkpoint parent manifest hash disagrees with evaluation dataset")
        # CLI method, single model seed and execution device are legitimate
        # per-run overrides; all scientific/data/training settings must match.
        import copy
        def comparison_config(value):
            result = copy.deepcopy(value)
            result.pop("method", None)
            result.pop("device", None)
            result.get("training", {}).pop("seed", None)
            result.get("evaluation", {}).pop("state_only_checkpoint", None)
            return result
        if comparison_config(payload.get("config", {})) != comparison_config(config):
            raise ValueError("checkpoint scientific/training configuration disagrees with evaluation configuration")
        if tuple(model.domain) != domain:
            raise ValueError("checkpoint physical domain disagrees with dataset")
        method = payload["method"]
        if method not in {"measure", "state_only"}:
            raise ValueError("unknown checkpoint method")
        models[method] = model
        checkpoint_records[method] = {"path": str(path), "dataset_hash": payload.get("dataset_hash"),
                                      "step": payload.get("step"), "best_step": payload.get("best_step")}
    controls, baseline_fit = fit_baselines(dataset, resolution=max(resolutions), degree=2)
    # Reserve the fixed analytic family/event controls, then fill held-out data.
    # A cap smaller than the audit bank is explicitly reported as incomplete.
    audits = [p for p in dataset["parents"] if p["split"] == "audit"]
    selected = audits[:max_parents]
    pools = [[p for p in dataset["parents"] if p["split"] == split] for split in ("test", "ood")]
    for index in range(max(map(len, pools), default=0)):
        for pool in pools:
            if index < len(pool) and len(selected) < max_parents:
                selected.append(pool[index])
    if not selected:
        raise ValueError("dataset has no held-out or audit parents")
    training_ids = set(baseline_fit.get("training_parent_ids", []))
    if training_ids.intersection(p["parent_id"] for p in selected):
        raise ValueError("physical training/evaluation parent leakage")
    floor = float(evaluation.get("relative_floor", 0.01))
    tolerance = float(evaluation.get("absolute_tolerance", 0.01))
    if floor <= 0:
        raise ValueError("relative error floor must be positive")
    if tolerance <= 0:
        raise ValueError("predeclared absolute tolerance must be positive")
    parents, timings, consistency, bl_records, unresolved = [], [], [], [], []
    lp_done = set()
    with paths["query_errors.jsonl"].open("x") as raw, paths["representation.jsonl"].open("x") as representations:
        for record in selected:
            per_method = {}
            for observation_time in record["times"]:
                parent = ParentInput(record["parent_id"], record["family"], tuple(record["parameters"]), observation_time, domain)
                batched_predictions = {method: _model_predictions(model, parent.parameters, observation_time, record["directions"], device)
                                       for method, model in models.items()} if parent.family in {"shock", "constant"} else {}
                for direction_index, direction in enumerate(record["directions"]):
                    ref = reference_sensitivity(parent, direction)
                    metadata = {"parent_id": parent.parent_id, "split": record["split"], "family": parent.family,
                                "time": observation_time, "parameters": list(parent.parameters), "domain": list(domain),
                                "direction_index": direction_index, "direction": list(direction),
                                "direction_norm": float(np.linalg.norm(direction)), "status": ref.status,
                                "event_distance": ref.event_metadata.get("event_distance"),
                                "reference_tier": "exact_analytic_burgers", "query_capabilities": "smooth normalized held-out queries"}
                    if ref.status == "unresolved":
                        entry = metadata | {"method": "exact_front", "reason": "exact event derivative deliberately unresolved"}
                        raw.write(json.dumps(entry, allow_nan=False) + "\n")
                        unresolved.append(entry)
                        continue
                    # The n/h ladder is restricted to the first nonzero direction per time.
                    nonzero_indices = [i for i, v in enumerate(record["directions"]) if np.linalg.norm(v) > 0]
                    if nonzero_indices and direction_index == nonzero_indices[0]:
                        for row in representation_rows(parent, direction, queries, resolutions, steps, widths):
                            representations.write(json.dumps(row | {"split": record["split"]}, allow_nan=False) + "\n")
                    predictions = {}
                    if parent.family in {"shock", "constant"}:
                        predictions["exact_front"] = _exact_front(parent.parameters, observation_time, direction, domain)
                        if "classical_front_regression" in controls:
                            predictions["classical_front_regression"] = controls["classical_front_regression"].predict(
                                parent.parameters, observation_time, direction)
                        for method, model in models.items():
                            predictions[method + "_chart"] = batched_predictions[method][direction_index]
                    else:
                        predictions["exact_front"] = ref
                    for method, prediction in predictions.items():
                        if isinstance(prediction, FrontPrediction) and not prediction.admissible:
                            entry = metadata | {"method": method, "status": "invalid_predicted_chart",
                                                "position": prediction.position, "states": prediction.states.tolist()}
                            raw.write(json.dumps(entry, allow_nan=False) + "\n")
                            per_method.setdefault(method, []).append(entry)
                            continue
                        metrics = front_errors(ref, prediction) if isinstance(prediction, FrontPrediction) else {
                            "state_l1_error": 0.0, "diffuse_l1_error": 0.0, "weight_error": 0.0,
                            "reference_total_variation": float(np.diff(ref.state.edges) @ np.abs(ref.diffuse_density) + np.abs(ref.atom_weights).sum())}
                        for query_index, query in enumerate(queries):
                            exact, predicted = ref.linear_pairing(query), prediction.linear_pairing(query)
                            nonlinear_error = abs(prediction.nonlinear_objective_derivative(query) - ref.nonlinear_objective_derivative(query))
                            entry = metadata | metrics | {"method": method, "query_index": query_index, "query": asdict(query),
                                    "exact_value": exact, "value": predicted, "absolute_error": abs(predicted - exact),
                                    "relative_error": abs(predicted - exact) / max(floor, abs(exact)),
                                    "nonlinear_gradient_error": nonlinear_error}
                            raw.write(json.dumps(entry, allow_nan=False) + "\n")
                            per_method.setdefault(method, []).append(entry)
                        if (isinstance(prediction, FrontPrediction) and method != "exact_front"
                                and parent.parent_id not in lp_done and len(lp_done) < lp_parents
                                and observation_time == record["times"][0]
                                and nonzero_indices and direction_index == nonzero_indices[0]):
                            bl_records.append(metadata | {"method": method, "diagnostic": measure_bl_diagnostic(ref, prediction, max(resolutions))})
                    if parent.family in {"shock", "constant"}:
                        grid_resolution = max(resolutions)
                        dx = (domain[1] - domain[0]) / grid_resolution
                        grid = domain[0] + (np.arange(grid_resolution) + 0.5) * dx
                        for method in ("learned_grid_state_autodiff", "direct_grid_sensitivity"):
                            if method not in controls:
                                continue
                            control = controls[method]
                            density = (control.jvp(parent.parameters, observation_time, direction) if method == "learned_grid_state_autodiff"
                                       else control.predict(parent.parameters, observation_time, direction))
                            associated_state = controls["learned_grid_state_autodiff"].predict(parent.parameters, observation_time)
                            state_error = float(dx * np.abs(associated_state - ref.state.cell_averages(grid_resolution)).sum())
                            for query_index, query in enumerate(queries):
                                exact = ref.linear_pairing(query)
                                predicted = float(dx * query.value(grid) @ density)
                                nonlinear_prediction = float(dx * (associated_state - query.value(grid)) @ density)
                                entry = metadata | {"method": method, "query_index": query_index, "query": asdict(query),
                                        "resolution": grid_resolution, "exact_value": exact, "value": predicted,
                                        "absolute_error": abs(predicted - exact), "relative_error": abs(predicted - exact) / max(floor, abs(exact)),
                                        "predicted_total_variation": float(dx * np.abs(density).sum()), "state_l1_error": state_error,
                                        "nonlinear_gradient_error": abs(nonlinear_prediction - ref.nonlinear_objective_derivative(query)),
                                        "nonlinear_semantics": "smooth-grid squared objective chain rule; direct tangent paired with ordinary grid state, integrability not assumed",
                                        "state_error_semantics": "L1 of cell-average fields" if state_error is not None else None}
                                raw.write(json.dumps(entry, allow_nan=False) + "\n")
                                per_method.setdefault(method, []).append(entry)
                    if (nonzero_indices and direction_index == nonzero_indices[0] and observation_time == record["times"][0]
                            and len(lp_done) < lp_parents and parent.family in {"shock", "constant"}):
                        lp_done.add(parent.parent_id)
            for method, rows in per_method.items():
                nonzero = [r for r in rows if r["direction_norm"] > 0]
                invalid_count = sum(r["status"] == "invalid_predicted_chart" for r in rows)
                metrics = _parent_statistics(nonzero)
                passed = bool(nonzero) and invalid_count == 0 and metrics.get("absolute_error_max", float("inf")) <= tolerance
                parents.append({"parent_id": record["parent_id"], "split": record["split"], "family": record["family"],
                                "method": method, "evaluated_query_rows": len(rows),
                                "invalid_chart_rows": invalid_count, "metrics": metrics,
                                "accuracy_gate": {"passed": passed, "absolute_tolerance": tolerance,
                                                  "criterion": "every nonzero-direction held-out query meets absolute tolerance; any invalid chart fails"},
                                "strata": {"zero_direction": _parent_statistics([r for r in rows if r["direction_norm"] == 0]),
                                           "near_event": _parent_statistics([r for r in nonzero if r["status"] == "near_event"]),
                                           "zero_reference_observable": _parent_statistics([r for r in nonzero if abs(r.get("exact_value", float("inf"))) <= floor])}})
            # Only regular one-front initial-data comparisons share learned input information.
            if record["family"] in {"shock", "constant"}:
                alpha, observation_time = record["parameters"], record["times"][0]
                directions = [v for v in record["directions"] if np.linalg.norm(v) > 0] or [[0.0, 0.0, 0.0]]
                builders = {"exact_front": lambda v: _exact_front(alpha, observation_time, v, domain)}
                multi_builders = {"exact_front": lambda vv: [_exact_front(alpha, observation_time, v, domain) for v in vv]}
                state_functions = {"exact_front": lambda: (np.asarray(alpha[:2]), alpha[2] + 0.5 * observation_time * (alpha[0] + alpha[1]))}
                if "classical_front_regression" in controls:
                    builders["classical_front_regression"] = lambda v: controls["classical_front_regression"].predict(alpha, observation_time, v)
                    multi_builders["classical_front_regression"] = lambda vv: [controls["classical_front_regression"].predict(alpha, observation_time, v) for v in vv]
                    state_functions["classical_front_regression"] = lambda: (controls["classical_front_regression"].features.values(np.r_[alpha, observation_time])
                                                                            @ controls["classical_front_regression"].coefficients)
                for method, model in models.items():
                    builders[method + "_chart"] = lambda v, m=model: _model_prediction(m, alpha, observation_time, v, device)
                    multi_builders[method + "_chart"] = lambda vv, m=model: _model_predictions(m, alpha, observation_time, vv, device)
                    state_functions[method + "_chart"] = lambda m=model: _model_state(m, alpha, observation_time, device)
                repeats, warmup = int(evaluation.get("timing_repeats", 3)), int(evaluation.get("warmup", 1))
                if repeats < 1 or warmup < 0:
                    raise ValueError("invalid timing protocol")
                for method, builder in builders.items():
                    for ds, qs in ((directions[:1], queries[:1]), (directions, queries)):
                        # Construct one measure per direction before looping over queries.
                        def endpoint(b=multi_builders[method], vv=ds, qq=qs):
                            measures = b(vv)
                            return [[m.linear_pairing(q) for q in qq] for m in measures]
                        timings.append({"parent_id": record["parent_id"], "method": method, "directions": len(ds), "queries": len(qs),
                                        "endpoint": "initial inputs -> forward chart -> directional assembly -> all requested weak queries; includes host transfer",
                                        "measurement": _timed(endpoint, repeats, warmup, device if method.endswith("_chart") else "cpu")})
                    representation = multi_builders[method](directions)
                    timing_device = device if method.endswith("_chart") else "cpu"
                    timings.append({"parent_id": record["parent_id"], "method": method, "kind": "components",
                                    "forward_state": _timed(state_functions[method], repeats, warmup, timing_device),
                                    "state_direction_and_representation": _timed(lambda b=multi_builders[method]: b(directions), repeats, warmup, timing_device),
                                    "query_only": _timed(lambda mm=representation: [[m.linear_pairing(q) for q in queries] for m in mm], repeats, warmup, "cpu"),
                                    "accounting": "components overlap; do not sum them. Total endpoint above is measured directly. Query-only operates on host arrays."})
                    # State/tangent agreement uses the associated chart's own finite variations.
                    v = np.asarray(directions[0])
                    h = 1e-3 if method.endswith("_chart") else 1e-5
                    original = builder(v)
                    if original.admissible:
                        if method == "exact_front":
                            altered = lambda a: _exact_front(a, observation_time, np.zeros(3), domain)
                        elif method == "classical_front_regression":
                            altered = lambda a: controls[method].predict(a, observation_time, np.zeros(3))
                        else:
                            learned = models[method.removesuffix("_chart")]
                            altered = lambda a: _model_prediction(learned, a, observation_time, np.zeros(3), device)
                        plus, minus = altered(np.asarray(alpha) + h * v), altered(np.asarray(alpha) - h * v)
                        # The chart's own parameter variation is a calculus
                        # check, including equal-trace perturbations with an
                        # increasing chart. Only its physical support needs to
                        # stay interior. Trusted entropy FD dispatches fans.
                        chart_variation_interior = all(domain[0] < p.position < domain[1]
                                                       and np.isfinite(np.r_[p.states, p.position]).all() for p in (plus, minus))
                        if chart_variation_interior:
                            errors = [abs((plus.linear_state_query(q) - minus.linear_state_query(q)) / (2 * h) - original.linear_pairing(q)) for q in queries]
                            first = builder(v)
                            second = builder(2 * v)
                            zero = builder(np.zeros(3))
                            linearity = max(abs(second.linear_pairing(q) - 2 * first.linear_pairing(q)) for q in queries)
                            zero_error = max(abs(zero.linear_pairing(q)) for q in queries)
                            other_v = np.asarray(directions[-1])
                            summed, other = builder(v + other_v), builder(other_v)
                            additivity = max(abs(summed.linear_pairing(q) - first.linear_pairing(q) - other.linear_pairing(q)) for q in queries)
                            consistency.append({"parent_id": record["parent_id"], "method": method, "h": h,
                                                "chart_finite_variation_max_error": max(errors), "scaling_max_error": linearity,
                                                "additivity_max_error": additivity,
                                                "zero_direction_max_error": zero_error,
                                                "status": "numerical_chart_consistency_diagnostic"})
                        else:
                            consistency.append({"parent_id": record["parent_id"], "method": method,
                                                "status": "unresolved_chart_finite_variation",
                                                "reason": "perturbed chart support leaves the physical domain"})
                    else:
                        consistency.append({"parent_id": record["parent_id"], "method": method,
                                            "status": "unresolved_chart_finite_variation", "reason": "inadmissible base chart"})
                # Grid controls include their associated state prediction in
                # the full endpoint. The direct tangent is paired explicitly
                # with the same ordinary grid-state regressor.
                for method in ("learned_grid_state_autodiff", "direct_grid_sensitivity"):
                    if method not in controls:
                        continue
                    grid_control = controls[method]
                    state_control = controls["learned_grid_state_autodiff"]
                    n = grid_control.resolution
                    dx = (domain[1] - domain[0]) / n
                    x = domain[0] + (np.arange(n) + 0.5) * dx
                    def grid_direction(v, m=method, c=grid_control):
                        return c.jvp(alpha, observation_time, v) if m == "learned_grid_state_autodiff" else c.predict(alpha, observation_time, v)
                    for ds, qs in ((directions[:1], queries[:1]), (directions, queries)):
                        def grid_endpoint(vv=ds, qq=qs):
                            state_control.predict(alpha, observation_time)
                            densities = [grid_direction(v) for v in vv]
                            return [[float(dx * q.value(x) @ density) for q in qq] for density in densities]
                        timings.append({"parent_id": record["parent_id"], "method": method, "directions": len(ds), "queries": len(qs),
                                        "endpoint": "initial inputs -> ordinary grid state -> grid directional output -> all weak queries",
                                        "associated_state": "learned_grid_state_autodiff", "measurement": _timed(grid_endpoint, repeats, warmup, "cpu")})
                    v = np.asarray(directions[0])
                    h = 1e-5
                    plus = state_control.predict(np.asarray(alpha) + h * v, observation_time)
                    minus = state_control.predict(np.asarray(alpha) - h * v, observation_time)
                    finite_density = (plus - minus) / (2 * h)
                    predicted_density = grid_direction(v)
                    variation_error = max(abs(float(dx * q.value(x) @ (finite_density - predicted_density))) for q in queries)
                    other = np.asarray(directions[-1])
                    additivity = max(abs(float(dx * q.value(x) @ (grid_direction(v + other) - predicted_density - grid_direction(other)))) for q in queries)
                    scaling = max(abs(float(dx * q.value(x) @ (grid_direction(2 * v) - 2 * predicted_density))) for q in queries)
                    zero = max(abs(float(dx * q.value(x) @ grid_direction(np.zeros(3)))) for q in queries)
                    consistency.append({"parent_id": record["parent_id"], "method": method, "h": h,
                                        "chart_finite_variation_max_error": variation_error, "additivity_max_error": additivity,
                                        "scaling_max_error": scaling, "zero_direction_max_error": zero,
                                        "status": "numerical_grid_state_tangent_consistency_diagnostic",
                                        "associated_state": "learned_grid_state_autodiff", "integrability_claim": False})
    parent_path = paths["evaluation_parents.jsonl"]
    with parent_path.open("x") as stream:
        for row in parents:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    inverse_config = config.get("inverse", {})
    inverse_results = {"exact_front": inverse_tracking(domain, int(inverse_config.get("steps", 10)), float(inverse_config.get("lr", 0.1)))}
    for method, model in models.items():
        def gradient_provider(alpha, observation_time, target, m=model):
            predictions = [_model_prediction(m, alpha, observation_time, v, device) for v in np.eye(3)]
            if not all(p.admissible for p in predictions):
                raise ValueError("invalid predicted inverse chart")
            return [p.nonlinear_objective_derivative(target) for p in predictions]
        inverse_results[method + "_chart"] = inverse_tracking(domain, int(inverse_config.get("steps", 10)),
                                                              float(inverse_config.get("lr", 0.1)), gradient_provider=gradient_provider)
    summary = {"schema_version": 1, "status": "completed_scoped_pilot", "domain": list(domain),
               "physical_parents": len(selected), "selected_parent_ids": [p["parent_id"] for p in selected],
               "independent_unit": "physical parent; times/directions/queries averaged within parent",
               "held_out_query_bank": [asdict(q) for q in queries], "baseline_training": baseline_fit,
               "checkpoints": checkpoint_records, "parent_aggregates": _aggregate(parents),
               "direction_and_chart_consistency": consistency, "bounded_lipschitz_diagnostics": bl_records,
               "unresolved_records": unresolved, "timings": timings, "inverse": inverse_results,
            "memory": memory_record(device), "wall_seconds": time.perf_counter() - started,
            "fixed_audit_coverage": {"available": len(audits), "evaluated": sum(p["split"] == "audit" for p in selected),
                                     "complete": all(p in selected for p in audits)},
            "accuracy_gate": {"absolute_tolerance": tolerance, "criterion": "all nonzero held-out weak-query errors <= tolerance, no invalid chart",
                              "parent_method_failures": [{"parent_id": p["parent_id"], "method": p["method"], "split": p["split"],
                                                           "invalid_chart_rows": p["invalid_chart_rows"], "worst_absolute_error": p["metrics"].get("absolute_error_max")}
                                                          for p in parents if not p["accuracy_gate"]["passed"]],
                              "research_advantage_established": False},
               "artifacts": {"query_errors": str(run_dir / "query_errors.jsonl"), "representation": str(run_dir / "representation.jsonl"),
                             "parents": str(parent_path)},
               "evidence": {"WP1": "exact analytic reference tests and event audits", "WP2": "raw coupled resolution/step and fixed physical smoothing ladders",
                            "WP3": "single-shock learned charts when checkpoints supplied; train-only classical and grid controls",
                            "WP4": "held-out parents, smooth query families, and parameter range controls only; learned multi-shock transfer incomplete",
                            "WP5": "trusted-objective fixed-budget tracking pilot; collision learning/one-sided targets incomplete",
                            "WP6": "not implemented", "WP7": "scoped actual inference timings; matched training-budget/CARC performance/novelty audits incomplete"},
               "limitations": ["finite held-out query bank is not the full BL norm", "LP residual checks are numerical diagnostics, not rigorous certificates",
                               "tanh smoothing is a declared regularized chart, not a viscous PDE solve or proved inviscid limit",
                               "polynomial controls and neural charts use different fitting optimizers; no matched-budget computational advantage claim",
                               "grid regressors use one declared resolution; resolution transfer is audited by exact n/h controls",
                               "local timing endpoints include host transfer and per-direction chart inference; batched GPU performance is unestablished",
                               "single-shock interpolation controls are expected to be strong; toy results do not establish research novelty"]}
    atomic_write_json(run_dir / "evaluation.json", summary)
    return summary
