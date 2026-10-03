"""Tiny FP64 calculus checks and FP32 deterministic checkpoint tests."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import tempfile

import numpy as np
import unittest
from unittest import mock
import torch
from torch import nn

from singular_sensitivity.data import generate_manifest
from singular_sensitivity.models import (ChartModel, chart_measure, chart_observables,
                                         linear_pairings, squared_objective_gradient)
from singular_sensitivity.queries import Query, query_bank
from singular_sensitivity.runtime import project_root
from singular_sensitivity import training

DOMAIN = (-2.0, 2.0)




class ExactShockChart(nn.Module):
    domain = DOMAIN

    def forward(self, alpha, time):
        return alpha[:, :2], alpha[:, 2:3] + 0.5 * time[:, None] * (alpha[:, :1] + alpha[:, 1:2])














def _tiny_config():
    return {"schema_version": 1, "domain": list(DOMAIN), "method": "measure",
            "data": {"seed": 1729, "train_parents": 8, "validation_parents": 3,
                     "test_parents": 2, "ood_parents": 0, "times": [0.2, 0.4]},
            "training": {"seeds": [17], "steps": 8, "batch_size": 4, "directions_per_parent": 2,
                         "lr": 0.001, "width": 8, "depth": 1, "eval_every": 4,
                         "checkpoint_every": 2, "patience": 0, "max_seconds": 60,
                         "loss_weights": {"weak": 1.0, "state": 1.0}}}


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right







class ModelTests(unittest.TestCase):
    def setUp(self):
        root = project_root() / ".cache" / "tmp"
        root.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(prefix="model-tests-", dir=root)
        self.addCleanup(directory.cleanup)
        self.artifact_directory = Path(directory.name)

    def test_initial_anchor_zero_jump_and_input_only_invocation(self):
        torch.manual_seed(3)
        model = ChartModel(width=8, depth=1).double()
        alpha = torch.tensor([[1.0, -0.5, 0.1], [0.3, 0.3, -0.2]], dtype=torch.float64)
        states, positions = model(alpha, torch.zeros(2, dtype=torch.float64))
        assert torch.equal(states, alpha[:, :2])
        assert torch.equal(positions, alpha[:, 2:3])
        measure = chart_measure(model, alpha[1:], torch.tensor([0.6], dtype=torch.float64),
                                torch.tensor([[[0.2, -0.3, 0.4]]], dtype=torch.float64))
        assert torch.equal(measure.states, alpha[1:, :2])
        assert torch.equal(measure.weights, torch.zeros_like(measure.weights))
        # The supported invocation contains only initial coefficients and time.
        # No label container, future support, or reference solver is available.
        result = model(alpha.clone(), torch.tensor([0.2, 0.4], dtype=torch.float64))
        assert result[0].shape == (2, 2) and result[1].shape == (2, 1)

    def test_signed_atoms_physical_pairings_and_payoff_jump(self):
        chart = ExactShockChart()
        alpha = torch.tensor([[1.0, -0.5, 0.2]], dtype=torch.float64)
        time = torch.tensor([0.4], dtype=torch.float64)
        directions = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]], dtype=torch.float64)
        measure = chart_measure(chart, alpha, time, directions)
        assert measure.positions.shape == (1, 1)
        assert torch.equal(measure.weights[0, :, 0], torch.tensor([1.5, -1.5, 0.0], dtype=torch.float64))
        constant = Query("constant", DOMAIN)
        # Pure position variation has no cell-width or probability normalization.
        assert torch.equal(linear_pairings(measure, [constant])[0, :, 0], measure.weights[0, :, 0])
        # With target zero, objective jump=(1^2-(-.5)^2)/2=.375, whereas
        # taking the left trace times the state atom would incorrectly give1.5.
        gradient = squared_objective_gradient(measure, 0.0)
        torch.testing.assert_close(gradient[0], torch.tensor([0.375, -0.375, 0.0], dtype=torch.float64), rtol=0, atol=0)

    def test_direction_additivity_scaling_and_shared_support(self):
        torch.manual_seed(7)
        model = ChartModel(width=8, depth=1).double()
        alpha = torch.tensor([[0.9, -0.6, 0.1]], dtype=torch.float64)
        time = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)
        w = torch.tensor([-0.1, 0.2, 0.3], dtype=torch.float64)
        directions = torch.stack((v, w, v + w, -2 * v, 0 * v))[None]
        measure = chart_measure(model, alpha, time, directions)
        pairings = linear_pairings(measure, query_bank(DOMAIN))
        torch.testing.assert_close(pairings[:, 2], pairings[:, 0] + pairings[:, 1], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(pairings[:, 3], -2 * pairings[:, 0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(pairings[:, 4], torch.zeros_like(pairings[:, 4]), rtol=0, atol=0)
        assert measure.positions.shape == (1, 1), "support must not gain a direction axis"

    def test_chart_jvp_matches_weak_state_finite_variation_and_nonlinear_gradient(self):
        torch.manual_seed(11)
        model = ChartModel(width=8, depth=1).double()
        alpha = torch.tensor([[0.9, -0.6, 0.1]], dtype=torch.float64)
        time = torch.tensor([0.5], dtype=torch.float64)
        directions = torch.tensor([[[0.2, -0.3, 0.4]]], dtype=torch.float64)
        queries = query_bank(DOMAIN)
        measure = chart_measure(model, alpha, time, directions)
        h = 1e-5
        plus, minus = alpha + h * directions[:, 0], alpha - h * directions[:, 0]
        finite_difference = (chart_observables(model, plus, time, queries) - chart_observables(model, minus, time, queries)) / (2 * h)
        torch.testing.assert_close(linear_pairings(measure, queries)[:, 0], finite_difference, rtol=1e-8, atol=1e-9)
        target = Query("sin", DOMAIN, frequency=1.3)

        def objective(parameters):
            states, positions = model(parameters, time)
            front = positions[:, 0]
            lengths = torch.stack((front - DOMAIN[0], DOMAIN[1] - front), dim=-1)
            target_integrals = torch.stack((target.torch_integral(DOMAIN[0], front),
                                            target.torch_integral(front, DOMAIN[1])), dim=-1)
            return (0.5 * (states.square() * lengths - 2 * states * target_integrals).sum(-1)
                    + 0.5 * target.torch_square_integral(DOMAIN[0], DOMAIN[1]))

        finite_difference = (objective(plus) - objective(minus)) / (2 * h)
        torch.testing.assert_close(squared_objective_gradient(measure, target)[:, 0], finite_difference, rtol=1e-8, atol=1e-9)

    def test_mixed_parameter_weight_derivative_against_centered_difference(self):
        torch.manual_seed(13)
        model = ChartModel(width=4, depth=1).double()
        alpha = torch.tensor([[0.8, -0.6, 0.1]], dtype=torch.float64)
        time = torch.tensor([0.3], dtype=torch.float64)
        directions = torch.tensor([[[0.3, -0.2, 0.4]]], dtype=torch.float64)
        queries = [Query("constant", DOMAIN), Query("sin", DOMAIN, frequency=1.3)]

        def loss():
            pairings = linear_pairings(chart_measure(model, alpha, time, directions), queries)
            return (pairings - 0.07).square().sum()

        value = loss()
        gradients = torch.autograd.grad(value, tuple(model.parameters()))
        for parameter, gradient in ((model.network[0].weight, gradients[0]),
                                    (model.network[-1].weight, gradients[-2])):
            index = (0, 0)
            original = float(parameter[index].detach())
            h = 1e-6
            with torch.no_grad():
                parameter[index] = original + h
            plus = float(loss().detach())
            with torch.no_grad():
                parameter[index] = original - h
            minus = float(loss().detach())
            with torch.no_grad():
                parameter[index] = original
            numerical = (plus - minus) / (2 * h)
            assert abs(float(gradient[index]) - numerical) < 1e-7 + 1e-5 * abs(numerical)

    def test_wrong_query_domain_and_boundary_chart_are_refused(self):
        model = ExactShockChart()
        alpha = torch.tensor([[1.0, -0.5, 0.1]], dtype=torch.float64)
        time = torch.tensor([0.4], dtype=torch.float64)
        directions = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
        measure = chart_measure(model, alpha, time, directions)
        incompatible = Query("constant", (-1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "domains differ"):
            linear_pairings(measure, [incompatible])
        with self.assertRaisesRegex(ValueError, "domains differ"):
            chart_observables(model, alpha, time, [incompatible])
        with self.assertRaisesRegex(ValueError, "domains differ"):
            squared_objective_gradient(measure, incompatible)
        alpha[:, 2] = 2.1
        measure = chart_measure(model, alpha, time, directions)
        assert measure.status == "unresolved"
        with self.assertRaisesRegex(ValueError, "unresolved chart"):
            linear_pairings(measure, [Query("constant", DOMAIN)])

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(1)
        config = _tiny_config()
        manifest = generate_manifest(config)
        full_dir, resumed_dir = self.artifact_directory / "full", self.artifact_directory / "resumed"
        full = training.train(config, manifest, full_dir)
        original = torch.optim.Adam.step
        counter = 0

        def pause_after_three(optimizer, *args, **kwargs):
            nonlocal counter
            result = original(optimizer, *args, **kwargs)
            counter += 1
            if counter == 3:
                os.kill(os.getpid(), signal.SIGUSR1)
            return result

        with mock.patch.object(torch.optim.Adam, "step", pause_after_three):
            paused = training.train(config, manifest, resumed_dir)
        assert paused["status"] == "paused" and paused["steps_completed"] == 3
        assert paused["stopped_reason"] == f"signal_{signal.SIGUSR1}"
        with (resumed_dir / "training.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": 4, "interrupted_attempt": True}) + "\n")
            stream.write("{interrupted_final_row")
        resumed = training.train(config, manifest, resumed_dir, resume=resumed_dir / "last.pt")
        assert Path(resumed["recovery_log"]).is_file()
        assert full["status"] == resumed["status"] == "complete"
        full_payload = torch.load(full_dir / "last.pt", weights_only=True)
        resumed_payload = torch.load(resumed_dir / "last.pt", weights_only=True)
        for key in ("model_state", "optimizer_state", "rng", "step", "best_step", "best_score", "history"):
            _assert_nested_equal(full_payload[key], resumed_payload[key])
        assert (resumed_dir / "last.previous.pt").is_file()
        training_query_specs = full_payload["query_protocol"]["training"]
        validation_query_specs = full_payload["query_protocol"]["validation"]
        assert not any(query in training_query_specs for query in validation_query_specs)
        test_specs = [vars(query) for query in query_bank(DOMAIN, held_out=True)]
        assert not any(query in test_specs for query in validation_query_specs)
        logs = [json.loads(line) for line in (resumed_dir / "training.jsonl").read_text().splitlines()]
        assert [record["step"] for record in logs] == list(range(1, 9))
        assert all(record["complete_step_seconds"] > 0 and record["teacher_seconds"] >= 0 for record in logs)

    def test_resume_new_directory_retains_prior_best_and_hash_guard(self):
        config = _tiny_config()
        config["training"]["min_delta"] = 1e6  # Initial model stays the global best.
        config["training"]["steps"] = 3
        manifest = generate_manifest(config)
        original = torch.optim.Adam.step

        def pause_once(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            training.request_pause("unit_test")
            return result

        source = self.artifact_directory / "source"
        destination = self.artifact_directory / "new-directory"
        with mock.patch.object(torch.optim.Adam, "step", pause_once):
            training.train(config, manifest, source)
        result = training.train(config, manifest, destination, resume=source / "last.pt")
        assert result["best_step"] == 0
        assert (destination / "best.pt").is_file()
        source_model, _ = training.load_model(source / "best.pt")
        destination_model, _ = training.load_model(destination / "best.pt")
        _assert_nested_equal(source_model.state_dict(), destination_model.state_dict())
        changed = deepcopy(config)
        changed["training"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "identical configuration"):
            training.train(changed, manifest, self.artifact_directory / "bad", resume=source / "last.pt")

    def test_training_never_accesses_test_parent_labels(self):
        from singular_sensitivity import reference
        config = _tiny_config()
        config["training"]["steps"] = 1
        config["training"]["eval_every"] = 1
        manifest = generate_manifest(config)
        seen = []
        original_state, original_sensitivity = reference.reference_state, reference.reference_sensitivity

        def guarded_state(parent):
            assert parent.parent_id.startswith(("train-", "validation-"))
            seen.append(parent.parent_id)
            return original_state(parent)

        def guarded_sensitivity(parent, direction, **kwargs):
            assert parent.parent_id.startswith(("train-", "validation-"))
            return original_sensitivity(parent, direction, **kwargs)

        patch_state = mock.patch.object(reference, "reference_state", guarded_state)
        patch_state.start()
        self.addCleanup(patch_state.stop)
        patch_sensitivity = mock.patch.object(reference, "reference_sensitivity", guarded_sensitivity)
        patch_sensitivity.start()
        self.addCleanup(patch_sensitivity.stop)
        result = training.train(config, manifest, self.artifact_directory / "sealed")
        assert result["status"] == "complete" and seen
        before = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(FileExistsError):
            training.train(config, manifest, self.artifact_directory / "sealed")
        assert signal.getsignal(signal.SIGTERM) == before
