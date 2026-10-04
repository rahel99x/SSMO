"""Bounded FP32 chart training with parent-separated labels and safe resume.

No test-parent parameters, directions, times, or labels are read by this module.
Reference labels are assembled separately and only input tensors reach the model.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import threading
import tempfile
import time
from typing import Any

import numpy as np
import torch

from .models import ChartModel, chart_measure, chart_observables, linear_pairings
from .runtime import confined_path
from .tower_reporting import emit_event

DIRECTION_CONVENTION = "physical-alpha=(u_left,u_right,initial_position); supplied-v; fixed-time; atom=(left-right)*Ds[v]"
_CHECKPOINT_SCHEMA = 1
_PAUSE_REQUESTED = False
_PAUSE_REASON = "requested"


def request_pause(reason: str = "requested") -> None:
    """Request a checkpoint at the next completed optimizer boundary."""
    global _PAUSE_REQUESTED, _PAUSE_REASON
    _PAUSE_REQUESTED, _PAUSE_REASON = True, reason


def _signal_pause(signum, frame) -> None:
    # No I/O, torch operations, or locks are performed in a signal handler.
    global _PAUSE_REQUESTED, _PAUSE_REASON
    _PAUSE_REQUESTED, _PAUSE_REASON = True, f"signal_{signum}"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path = confined_path(path)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = confined_path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    """Install a fully written checkpoint while retaining a known-good copy."""
    path = confined_path(path)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = confined_path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.replace(path, confined_path(path.with_name(path.stem + ".previous" + path.suffix)))
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_resume_log(path: Path, step: int) -> str | None:
    """Preserve crash-tail rows while keeping resumed optimizer steps unique."""
    path = confined_path(path)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    retained, needs_recovery = [], False
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("training log has a malformed interior record")
            needs_recovery = True  # Interrupted final write is quarantined.
            continue
        if int(record["step"]) <= step:
            retained.append(line if line.endswith("\n") else line + "\n")
        else:
            needs_recovery = True
    if not needs_recovery:
        return None
    archive = confined_path(path.with_name(f"training.recovery.{time.time_ns()}.jsonl"))
    descriptor, name = tempfile.mkstemp(prefix=".training-recovery.", suffix=".tmp", dir=path.parent)
    temporary = confined_path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.writelines(retained)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(path, archive)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(archive)


def _rng_state(sampler: np.random.Generator, device="cpu") -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": numpy_state[2], "has_gauss": numpy_state[3],
                  "cached_gaussian": numpy_state[4]},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else [],
        "sampler": sampler.bit_generator.state,
    }


def _restore_rng(state: dict, sampler: np.random.Generator) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint requires CUDA to restore its RNG state")
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])
    sampler.bit_generator.state = state["sampler"]


def _read_checkpoint(path: Path, device="cpu") -> dict:
    path = confined_path(path)
    if not path.exists():
        previous = confined_path(path.with_name(path.stem + ".previous" + path.suffix))
        if previous.exists():
            path = previous
    # Only tensors and primitive containers are serialized. Avoid unsafe pickle
    # fallback so checkpoints cannot silently execute arbitrary Python objects.
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("checkpoint_schema_version") != _CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("direction_convention") != DIRECTION_CONVENTION:
        raise ValueError("checkpoint direction convention does not match")
    return payload


def load_model(checkpoint: str | Path, device="cpu") -> tuple[ChartModel, dict]:
    """Load an evaluation model and its frozen experiment metadata."""
    payload = _read_checkpoint(Path(checkpoint), device="cpu")
    specification = payload["model_specification"]
    model = ChartModel(**specification).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def _reference_batch(records: list[dict], selected_times: list[float], direction_indices: list[list[int]],
                     domain: tuple[float, float], queries, device) -> dict[str, torch.Tensor]:
    from .reference import ParentInput, reference_state, reference_sensitivity

    inputs, directions, state_targets, query_targets = [], [], [], []
    support_targets, atom_targets, active = [], [], []
    for record, observation_time, indices in zip(records, selected_times, direction_indices):
        parameters = tuple(float(x) for x in record["parameters"])
        parent = ParentInput(parent_id=record["parent_id"], family=record["family"],
                             parameters=parameters, time=observation_time, domain=domain)
        state = reference_state(parent)
        state_targets.append([state.linear_query(query) for query in queries])
        inputs.append(parameters)
        parent_directions, parent_queries, parent_weights = [], [], []
        for index in indices:
            direction = [float(x) for x in record["directions"][index]]
            result = reference_sensitivity(parent, direction)
            if result.status != "regular":
                raise ValueError(f"training reference is not regular: {record['parent_id']}")
            parent_directions.append(direction)
            parent_queries.append([result.linear_pairing(query) for query in queries])
            parent_weights.append(float(result.atom_weights[0]) if len(result.atom_weights) else 0.0)
        directions.append(parent_directions)
        query_targets.append(parent_queries)
        atom_targets.append(parent_weights)
        support_targets.append(float(state.edges[1]) if len(state.edges) > 2 else float(parameters[2]))
        active.append(record["family"] == "shock" and parameters[0] != parameters[1])
    return {
        "alpha": torch.tensor(inputs, dtype=torch.float32, device=device),
        "time": torch.tensor(selected_times, dtype=torch.float32, device=device),
        "directions": torch.tensor(directions, dtype=torch.float32, device=device),
        "state_targets": torch.tensor(state_targets, dtype=torch.float32, device=device),
        "query_targets": torch.tensor(query_targets, dtype=torch.float32, device=device),
        "support_targets": torch.tensor(support_targets, dtype=torch.float32, device=device),
        "atom_targets": torch.tensor(atom_targets, dtype=torch.float32, device=device),
        "active": torch.tensor(active, dtype=torch.bool, device=device),
    }


def _sample_batch(parents, batch_size: int, direction_count: int, sampler, domain, queries, device):
    chosen = [parents[int(index)] for index in sampler.integers(0, len(parents), size=batch_size)]
    times, directions = [], []
    for parent in chosen:
        if not parent["times"] or not parent["directions"]:
            raise ValueError("training parent needs times and directions")
        times.append(float(parent["times"][int(sampler.integers(len(parent["times"]))) ]))
        directions.append(sampler.choice(len(parent["directions"]), size=direction_count,
                                         replace=direction_count > len(parent["directions"])).tolist())
    return _reference_batch(chosen, times, directions, domain, queries, device)


def _validation_batches(parents, batch_size, direction_count, domain, queries, device):
    # Last observation time and first manifest directions are a deterministic,
    # fixed validation protocol. Held-out queries are reserved for evaluation.
    batches = []
    for start in range(0, len(parents), batch_size):
        records = parents[start:start + batch_size]
        times = [float(record["times"][-1]) for record in records]
        indices = [[index % len(record["directions"]) for index in range(direction_count)] for record in records]
        batches.append(_reference_batch(records, times, indices, domain, queries, device))
    return batches


def _loss_terms(model, batch, queries, method, weights):
    prediction = chart_observables(model, batch["alpha"], batch["time"], queries)
    state_loss = (prediction - batch["state_targets"]).square().mean()
    terms = {"state": state_loss}
    total = weights.get("state", 1.0) * state_loss
    if method == "measure":
        measure = chart_measure(model, batch["alpha"], batch["time"], batch["directions"])
        weak_loss = (linear_pairings(measure, queries) - batch["query_targets"]).square().mean()
        terms["weak"] = weak_loss
        total = total + weights.get("weak", 1.0) * weak_loss
        if weights.get("support", 0.0):
            error = (measure.positions[:, 0] - batch["support_targets"]).square() / (model.domain[1] - model.domain[0]) ** 2
            support_loss = (error * batch["active"]).sum() / batch["active"].sum().clamp_min(1)
            terms["support"] = support_loss
            total = total + weights["support"] * support_loss
        if weights.get("atom", weights.get("weights", 0.0)):
            atom_loss = (measure.weights[:, :, 0] - batch["atom_targets"]).square().mean()
            terms["atom"] = atom_loss
            total = total + weights.get("atom", weights.get("weights", 0.0)) * atom_loss
        if weights.get("tv", 0.0):
            lo, hi = model.domain
            support = measure.positions[:, 0].clamp(min=lo, max=hi)
            lengths = torch.stack((support - lo, hi - support), dim=-1)
            variation = ((measure.diffuse.abs() * lengths[:, None, :]).sum(-1)
                         + (measure.weights.abs() * measure.atom_mask[:, None, :]).sum(-1)).mean()
            terms["tv"] = variation
            total = total + weights["tv"] * variation
    return total, terms


def _validate(model, batches, queries):
    totals = {"state_mse": 0.0, "weak_mse": 0.0}
    examples = 0
    model.eval()
    # torch.func.jvp works with no_grad and retains the forward tangents.
    with torch.no_grad():
        for batch in batches:
            count = batch["alpha"].shape[0]
            prediction = chart_observables(model, batch["alpha"], batch["time"], queries)
            measure = chart_measure(model, batch["alpha"], batch["time"], batch["directions"])
            totals["state_mse"] += count * float((prediction - batch["state_targets"]).square().mean())
            totals["weak_mse"] += count * float((linear_pairings(measure, queries) - batch["query_targets"]).square().mean())
            examples += count
    model.train()
    values = {name: value / examples for name, value in totals.items()}
    values["score"] = values["state_mse"] + values["weak_mse"]
    return values


def _hardware(device):
    record = {"device": str(device), "torch_version": str(torch.__version__), "precision": "float32", "compiled": False}
    if torch.device(device).type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory)
    return record


def train(config: dict, dataset: dict, run_dir: str | Path, device="cpu", resume=None) -> dict:
    """Train one seed/method within its step and invocation-time budgets.

    ``status=paused`` requires a nonzero CLI exit so Slurm ``afterok`` cannot
    begin dependent evaluation. The same config and dataset hashes, model,
    optimizer, random generators, and sampler are restored when resuming.
    """
    global _PAUSE_REQUESTED, _PAUSE_REASON
    _PAUSE_REQUESTED, _PAUSE_REASON = False, "requested"
    from .queries import Query, query_bank

    run_dir = confined_path(run_dir)
    log_path = confined_path(run_dir / "training.jsonl")
    if resume is None and any(confined_path(run_dir / name).exists() for name in
                              ("training.jsonl", "best.pt", "last.pt", "training_summary.json")):
        raise FileExistsError("fresh training would overwrite an existing run; use resume or a new run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    options = config.get("training", {})
    method = config.get("method", "measure")
    if method not in {"measure", "state_only"}:
        raise ValueError("method must be measure or state_only")
    if dataset.get("schema_version") != 1:
        raise ValueError("unsupported dataset schema")
    # Metadata filtering is the only contact with sealed test records. Their
    # parameters, times, directions, and labels are never accessed.
    parents = {split: [record for record in dataset["parents"] if record["split"] == split]
               for split in ("train", "validation")}
    if not parents["train"] or not parents["validation"]:
        raise ValueError("training requires nonempty train and validation splits")
    if any(record["family"] not in {"shock", "constant"} for split in parents.values() for record in split):
        raise ValueError("only shock and constant parents are supported for this training stage")
    domain = tuple(float(x) for x in dataset.get("domain", config.get("domain", (-2, 2))))
    if domain != tuple(float(x) for x in config.get("domain", domain)):
        raise ValueError("config and dataset domains differ")
    queries = query_bank(domain, held_out=False)
    # Local bump locations and noninteger frequencies are disjoint from both
    # the training moments and the sealed final-evaluation query bank.
    validation_queries = [Query("bump", domain, center=c, width=w)
                          for c, w in ((0.29, 0.12), (0.57, 0.18), (0.81, 0.09))] + [
                              Query("sin", domain, frequency=1.7), Query("cos", domain, frequency=2.3)]
    query_protocol = {"training": [asdict(query) for query in queries],
                      "validation": [asdict(query) for query in validation_queries],
                      "checkpoint_score": "state_mse + weak_mse on fixed validation bank"}
    config_hash, dataset_hash = _hash(config), _hash(dataset)
    seed = int(options.get("seed", options.get("seeds", [config.get("seed", 0)])[0]))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA training is unavailable")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    sampler = np.random.default_rng(seed)
    model_specification = {"width": int(options.get("width", 32)), "depth": int(options.get("depth", 2)), "domain": list(domain)}
    model = ChartModel(**model_specification).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options.get("lr", 0.001)))
    step, best_score, best_step, stale_evaluations = 0, math.inf, 0, 0
    cumulative_seconds, history = 0.0, []
    resume_payload = None
    best_payload = None
    if resume is not None:
        resume_path = confined_path(resume)
        payload = _read_checkpoint(resume_path, device="cpu")
        resume_payload = payload
        if payload["config_hash"] != config_hash or payload["dataset_hash"] != dataset_hash:
            raise ValueError("resume requires identical configuration and parent manifest hashes")
        if payload.get("query_protocol") != query_protocol:
            raise ValueError("resume validation query/checkpoint protocol differs")
        if payload["method"] != method or payload["model_specification"] != model_specification:
            raise ValueError("resume method or architecture differs")
        if torch.device(payload["hardware"]["device"]).type != torch.device(device).type:
            raise ValueError("resume requires the same CPU/CUDA device class")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(device)
        step, best_score, best_step = payload["step"], payload["best_score"], payload["best_step"]
        stale_evaluations = payload["stale_evaluations"]
        cumulative_seconds, history = payload["elapsed_seconds"], payload["history"]
        _restore_rng(payload["rng"], sampler)
        existing_best = confined_path(run_dir / "best.pt")
        best_source = existing_best if existing_best.exists() else confined_path(resume_path.parent / "best.pt")
        if payload["step"] == payload["best_step"] and payload["best_score"] == payload["history"][-1]["validation"]["score"]:
            best_payload = payload
        elif best_source.exists():
            best_payload = _read_checkpoint(best_source, device="cpu")
        else:
            raise ValueError("resume requires the earlier best.pt to preserve global checkpoint selection")
        if (best_payload["config_hash"] != config_hash or best_payload["dataset_hash"] != dataset_hash
                or best_payload["step"] != best_step or best_payload["best_score"] != best_score):
            raise ValueError("earlier best checkpoint does not match resume history")
        if not existing_best.exists():
            _atomic_checkpoint(existing_best, best_payload)
    batch_size = int(options.get("batch_size", 32))
    direction_count = int(options.get("directions_per_parent", 4))
    step_limit = int(options.get("steps", 1000))
    eval_every = int(options.get("eval_every", 100))
    checkpoint_every = int(options.get("checkpoint_every", eval_every))
    patience = int(options.get("patience", 20))
    max_seconds = float(options.get("max_seconds", 3600))
    min_delta = float(options.get("min_delta", 0.0))
    if min(batch_size, direction_count, step_limit, eval_every, checkpoint_every) < 1 or max_seconds <= 0:
        raise ValueError("training budgets must be positive")
    recovery_log = _prepare_resume_log(log_path, step) if resume is not None else None
    loss_weights = dict(options.get("loss_weights", {"state": 1.0, "weak": 1.0}))
    hardware = _hardware(device)
    # Validation teachers are generated once into small direction/query tables.
    # They never enter model inference and are reused to avoid teacher overhead.
    started = time.monotonic()
    teacher_started = time.monotonic()
    validation = _validation_batches(parents["validation"], batch_size, direction_count, domain, validation_queries, device)
    validation_teacher_seconds = time.monotonic() - teacher_started
    training_teacher_seconds = float(resume_payload.get("training_teacher_seconds", 0.0)) if resume_payload else 0.0
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            previous_handlers[signum] = signal.signal(signum, _signal_pause)
    status, stopped_reason = "complete", "step_budget"
    tower_last_written = -math.inf
    last_terms = {}
    validation_values = history[-1]["validation"] if history else None

    def snapshot():
        return {
            "checkpoint_schema_version": _CHECKPOINT_SCHEMA,
            "direction_convention": DIRECTION_CONVENTION,
            "config": config, "config_hash": config_hash, "query_protocol": query_protocol,
            "dataset_hash": dataset_hash, "method": method,
            "model_specification": model_specification,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "step": step, "best_score": best_score, "best_step": best_step,
            "stale_evaluations": stale_evaluations,
            "elapsed_seconds": cumulative_seconds + time.monotonic() - started,
            "training_teacher_seconds": training_teacher_seconds,
            "history": history, "rng": _rng_state(sampler, device), "hardware": hardware,
            "sampler_metadata": {"with_replacement_parents": True, "direction_count": direction_count,
                                 "validation_time_rule": "last_manifest_time", "step": step},
        }

    try:
        model.train()
        if validation_values is None:
            validation_values = _validate(model, validation, validation_queries)
            best_score, best_step = validation_values["score"], step
            history.append({"step": step, "validation": validation_values})
            _atomic_checkpoint(run_dir / "best.pt", snapshot())
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            while step < step_limit:
                if _PAUSE_REQUESTED or time.monotonic() - started >= max_seconds:
                    status = "paused"
                    stopped_reason = _PAUSE_REASON if _PAUSE_REQUESTED else "wall_time_budget"
                    break
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                step_started = time.monotonic()
                teacher_started = time.monotonic()
                batch = _sample_batch(parents["train"], batch_size, direction_count, sampler, domain, queries, device)
                teacher_seconds = time.monotonic() - teacher_started
                training_teacher_seconds += teacher_seconds
                forward_started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                loss, terms = _loss_terms(model, batch, queries, method, loss_weights)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training loss; previous checkpoint remains usable")
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                forward_jvp_seconds = time.monotonic() - forward_started
                backward_started = time.monotonic()
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in model.parameters()):
                    raise FloatingPointError("nonfinite weight gradient; previous checkpoint remains usable")
                optimizer.step()
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                backward_optimizer_seconds = time.monotonic() - backward_started
                complete_step_seconds = time.monotonic() - step_started
                step += 1
                last_terms = {key: float(value.detach()) for key, value in terms.items()}
                entry = {"step": step, "loss": float(loss.detach()), "loss_terms": last_terms,
                         "teacher_seconds": teacher_seconds, "forward_and_jvp_seconds": forward_jvp_seconds,
                         "backward_and_optimizer_seconds": backward_optimizer_seconds,
                         "complete_step_seconds": complete_step_seconds,
                         "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
                if step % eval_every == 0 or step == step_limit:
                    validation_values = _validate(model, validation, validation_queries)
                    entry["validation"] = validation_values
                    history.append({"step": step, "validation": validation_values})
                    if validation_values["score"] < best_score - min_delta:
                        best_score, best_step, stale_evaluations = validation_values["score"], step, 0
                        _atomic_checkpoint(run_dir / "best.pt", snapshot())
                    else:
                        stale_evaluations += 1
                entry["elapsed_seconds"] = cumulative_seconds + time.monotonic() - started
                if torch.device(device).type == "cuda":
                    entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    entry["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                    if entry["peak_reserved_bytes"] > 0.8 * hardware["gpu_total_bytes"]:
                        request_pause("gpu_memory_budget")
                log.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
                # Keep every raw step in training.jsonl. Tower receives the
                # current measured step at validation/checkpoint boundaries or
                # about every two seconds, avoiding metadata I/O at every update.
                if (os.environ.get("SSMO_TOWER_ACTIVE_COMMAND") == "train"
                        and ("validation" in entry or step % checkpoint_every == 0
                             or time.monotonic() - tower_last_written >= 2.0)):
                    observations = {key: entry[key] for key in (
                        "loss", "teacher_seconds", "forward_and_jvp_seconds", "backward_and_optimizer_seconds",
                        "complete_step_seconds", "host_peak_rss_bytes", "peak_allocated_bytes", "peak_reserved_bytes")
                        if key in entry}
                    observations["training_cumulative_seconds"] = entry["elapsed_seconds"]
                    observations.update({"loss_" + key: value for key, value in last_terms.items()})
                    if "validation" in entry:
                        observations.update({"validation_" + key: value for key, value in entry["validation"].items()
                                             if isinstance(value, (int, float)) and not isinstance(value, bool)})
                    emit_event("train", observations, step=step, completed=step, total=step_limit, unit="optimizer steps")
                    tower_last_written = time.monotonic()
                if step % checkpoint_every == 0:
                    _atomic_checkpoint(run_dir / "last.pt", snapshot())
                if patience > 0 and stale_evaluations >= patience:
                    stopped_reason = "validation_patience"
                    break
                if _PAUSE_REQUESTED:
                    status, stopped_reason = "paused", _PAUSE_REASON
                    break
        # A paused optimizer boundary is saved without inserting an extra
        # validation call, preserving the checkpoint schedule on exact resume.
        _atomic_checkpoint(run_dir / "last.pt", snapshot())
        summary = {
            "status": status, "stopped_reason": stopped_reason, "method": method, "seed": seed,
            "steps_completed": step, "best_step": best_step, "best_validation_score": best_score,
            "validation": validation_values, "loss_terms": last_terms,
            "elapsed_seconds": cumulative_seconds + time.monotonic() - started,
            "config_hash": config_hash, "dataset_hash": dataset_hash,
            "direction_convention": DIRECTION_CONVENTION, "query_protocol": query_protocol,
            "best_checkpoint": str(run_dir / "best.pt"), "last_checkpoint": str(run_dir / "last.pt"),
            "hardware": hardware,
            "training_teacher_seconds": training_teacher_seconds,
            "validation_teacher_seconds": validation_teacher_seconds,
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "recovery_log": recovery_log,
        }
        if torch.device(device).type == "cuda":
            summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        _atomic_json(run_dir / "training_summary.json", summary)
        return summary
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
