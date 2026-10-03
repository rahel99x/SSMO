"""Bounded pilot optimization with trusted objective acceptance at every step."""
import time

import numpy as np


def inverse_tracking(domain=(-2.0, 2.0), steps=10, learning_rate=0.1,
                     gradient_provider=None, initial=(1.1, -0.9, 0.25),
                     bounds=((0.2, 2.0), (-2.0, -0.2), (-0.7, 0.7)),
                     observation_time=0.4, max_backtracks=10):
    """A fixed-budget smooth tracking task; no evaluation examples tune this task.

    The field objective is nonlinear and uses the moving-boundary payoff jump.
    Learned gradients may propose a step, but only exact entropy-state objective
    evaluations accept it. An exhausted learned line search gets one explicitly
    counted exact-gradient fallback. This toy target need not identify parameters.
    """
    from .queries import Query
    from .reference import ParentInput, reference_sensitivity
    if steps < 0 or learning_rate <= 0 or max_backtracks < 1:
        raise ValueError("invalid fixed optimization budget")
    domain = tuple(domain)
    bound_array = np.asarray(bounds, dtype=np.float64)
    parameters = np.clip(np.asarray(initial, dtype=np.float64), bound_array[:, 0], bound_array[:, 1])
    if parameters.shape != (3,) or bound_array.shape != (3, 2) or np.any(bound_array[:, 0] >= bound_array[:, 1]):
        raise ValueError("inverse task needs three initial-data parameters and increasing bounds")
    target = Query("sin", domain, frequency=1.0)
    objective_calls = 0
    exact_gradient_calls = 0
    started = time.perf_counter()

    def make_parent(alpha):
        return ParentInput("inverse-fixed-initialization", "shock", tuple(alpha), observation_time, domain)

    def objective(alpha):
        nonlocal objective_calls
        objective_calls += 1
        ref = reference_sensitivity(make_parent(alpha), np.zeros(3))
        if ref.status == "unresolved":
            raise ValueError("inverse objective encountered an unresolved reference")
        return float(ref.state.squared_tracking_objective(target))

    def exact_gradient(alpha):
        nonlocal exact_gradient_calls
        exact_gradient_calls += 3
        return np.asarray([reference_sensitivity(make_parent(alpha), v).nonlinear_objective_derivative(target)
                           for v in np.eye(3)], dtype=np.float64)

    current = objective(parameters)
    initial_objective = current
    history = [{"step": 0, "parameters": parameters.tolist(), "trusted_objective": current}]
    fallbacks, failure, gradient_errors = 0, None, []
    for step in range(1, steps + 1):
        trusted_gradient = exact_gradient(parameters)
        source = "exact_payoff_jump" if gradient_provider is None else "learned_chart_payoff_jump"
        try:
            gradient = trusted_gradient if gradient_provider is None else np.asarray(
                gradient_provider(parameters, observation_time, target), dtype=np.float64)
            if gradient.shape != (3,) or not np.isfinite(gradient).all():
                raise ValueError("nonfinite or invalid learned inverse gradient")
        except (ValueError, RuntimeError, FloatingPointError):
            gradient = trusted_gradient
            source = "exact_gradient_fallback_invalid_chart"
            fallbacks += 1
        gradient_errors.append(float(np.linalg.norm(gradient - trusted_gradient)))
        if np.linalg.norm(trusted_gradient) < 1e-10:
            failure = "stationary_within_declared_tolerance"
            break

        accepted = False
        evaluations_this_step = 0
        attempts = [gradient] if gradient_provider is None or source.startswith("exact") else [gradient, trusted_gradient]
        for attempt_index, candidate_gradient in enumerate(attempts):
            if attempt_index:
                source = "exact_gradient_fallback_failed_line_search"
                fallbacks += 1
            for backtrack in range(max_backtracks):
                scale = learning_rate * 0.5 ** backtrack
                proposed = np.clip(parameters - scale * candidate_gradient, bound_array[:, 0], bound_array[:, 1])
                if np.linalg.norm(proposed - parameters) < 1e-14:
                    continue
                # Keep the declared entropy shock and no boundary interactions.
                position = proposed[2] + observation_time * (proposed[0] + proposed[1]) / 2.0
                if proposed[0] <= proposed[1] or not domain[0] < position < domain[1]:
                    continue
                proposed_objective = objective(proposed)
                evaluations_this_step += 1
                expected = float(np.dot(trusted_gradient, proposed - parameters))
                if proposed_objective < current and proposed_objective <= current + 1e-4 * min(0.0, expected):
                    parameters, current, accepted = proposed, proposed_objective, True
                    history.append({"step": step, "parameters": parameters.tolist(), "trusted_objective": current,
                                    "gradient_source": source, "gradient": candidate_gradient.tolist(),
                                    "gradient_error_l2": gradient_errors[-1], "backtracks": backtrack,
                                    "objective_evaluations": evaluations_this_step})
                    break
            if accepted:
                break
        if not accepted:
            failure = "no_trusted_descent_within_fixed_line_search_budget"
            break
    # Reevaluate the final candidate independently, even after the final accepted step.
    final = objective(parameters)
    return {"status": "completed" if failure is None else failure,
            "target": {"kind": "sin", "frequency": 1.0, "domain": list(domain)},
            "objective": "0.5 integral (u-target)^2; exact payoff-jump derivative",
            "initial_parameters": list(initial), "bounds": bound_array.tolist(),
            "observation_time": observation_time, "fixed_steps": steps, "learning_rate": learning_rate,
            "max_backtracks": max_backtracks, "accepted_steps": len(history) - 1,
            "initial_trusted_objective": initial_objective, "final_trusted_objective": final,
            "final_parameters": parameters.tolist(), "trusted_objective_evaluations": objective_calls,
            "trusted_gradient_reference_evaluations": exact_gradient_calls,
            "exact_gradient_fallbacks": fallbacks,
            "gradient_error_l2_max": max(gradient_errors, default=0.0),
            "identifiability": "not established for this smooth tracking target; no recovery claim",
            "total_seconds": time.perf_counter() - started, "history": history}
