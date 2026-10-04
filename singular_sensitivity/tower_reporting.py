"""Bounded, dependency-free SSMO reports for Tower's version-1 interchange.

One batch coordinator owns lifecycle files; one application process appends
metrics at meaningful boundaries. No scheduler queries, polling, or scientific
calculations occur here. Unknown allocations and measurements remain unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import statistics
import stat
import sys
import tempfile
import time

from .runtime import project_root

MAX_JSON = 1 << 20
MAX_LOG_JSON = 256 << 10
MAX_METRIC_LINE = 65535
MAX_METRIC_FILE = 4 << 20
MAX_SMALL_JSON = 256 << 10
_EVENT_STATE = {}
RESOURCE_KEYS = {"partition", "account", "qos", "cpus", "nodes", "gpus",
                 "gpu_type", "mem_bytes", "time_seconds"}
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                   "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
                   "UNKNOWN", "INTERRUPTED"}
STATES = TERMINAL_STATES | {"CREATED", "SUBMITTED", "PENDING", "RUNNING", "COMPLETING",
    "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "RESIZING", "SIGNALING", "SPECIAL_EXIT",
    "STAGE_OUT", "STOPPED", "SUSPENDED"}
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SUMMARY_KEYS = RESOURCE_KEYS | {"runtime_seconds", "cpu_seconds", "memory_bytes", "memory_scope",
    "start", "end", "submit", "input_size", "script_sha256", "parameters", "results", "metadata"}


def _text(value, name, limit=128, *, empty=False):
    if (not isinstance(value, str) or not empty and not value or len(value) > limit
            or value and not value.isprintable()):
        raise ValueError(f"{name} must be printable text of at most {limit} characters")
    return value


def _number(value, name, *, minimum=0, maximum=1e100, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        valid = math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    if not valid or integer and not isinstance(value, int):
        raise ValueError(f"{name} is outside its numeric bounds")
    return value


def _encoded(value, limit=MAX_JSON):
    stack, count = [(value, 0)], 0
    while stack:
        current, depth = stack.pop()
        count += 1
        if depth > 24 or count > 30000:
            raise ValueError("Report exceeds JSON depth/value bounds")
        if isinstance(current, dict):
            for key, item in current.items():
                _text(key, "JSON key", 4096)
                stack.append((item, depth + 1))
        elif isinstance(current, (list, tuple)):
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            # General scientific JSON may contain escaped multiline evidence,
            # such as nvidia-smi output. Printable restrictions belong to the
            # specified path/identity/label/metric/parameter fields.
            if len(current) > MAX_JSON:
                raise ValueError("JSON text exceeds its byte budget")
        elif current is None or isinstance(current, bool):
            pass
        else:
            _number(current, "JSON value", minimum=-1e300, maximum=1e300)
    raw = (json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")) + "\n").encode("utf-8")
    if len(raw) > limit:
        raise ValueError(f"Report exceeds {limit} bytes")
    return raw


def _path(value, *, directory=False, allow_root=False):
    root = project_root()
    raw = str(value)
    _text(raw, "path", 4096)
    if "\\" in raw or any(c in raw for c in "*?[]") or any(p in {".", "..", ""} for p in raw.lstrip("/").split("/")):
        raise ValueError("Reporting paths must be exact, with no traversal or globs")
    target = Path(value)
    target = target if target.is_absolute() else root / target
    if target == root and not allow_root or not target.is_relative_to(root):
        raise ValueError("Reporting paths must stay strictly inside SSMO_PROJECT_ROOT")
    for entry in (*reversed(target.parents), target):
        if os.path.lexists(entry):
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("Reporting paths cannot contain symlinks")
            if entry != target or directory:
                if not stat.S_ISDIR(mode):
                    raise ValueError("Reporting ancestor must be a real directory")
            elif not stat.S_ISREG(mode) or entry.stat().st_nlink != 1:
                raise ValueError("Reporting file must be regular and have no hardlink aliases")
    return target


def _read(path, limit=MAX_JSON):
    path = _path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise ValueError("Reporting JSON must be a bounded regular file without aliases")
        chunks, remaining = [], limit + 1
        while remaining:
            block = os.read(fd, min(65536, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after, named = os.fstat(fd), path.stat(follow_symlinks=False)
        signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if len(raw) > limit or len(raw) != before.st_size or signature(before) != signature(after) or signature(after) != signature(named):
            raise ValueError("Reporting JSON changed during the read")
    finally:
        os.close(fd)
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate JSON keys are refused")
            result[key] = item
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    _encoded(value, limit)
    return value


def _atomic(path, value, *, exclusive=False, limit=MAX_JSON):
    destination = _path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = _encoded(value, limit)
    fd, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
        if exclusive:
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _parameters(value):
    if not isinstance(value, dict) or len(value) > 64:
        raise ValueError("parameters must be an object with at most 64 keys")
    stack, count = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > 4 or count > 128:
            raise ValueError("parameters exceed four levels or 128 values")
        if isinstance(item, dict):
            for key, child in item.items():
                _text(key, "parameter key", 64)
                stack.append((child, depth + 1))
        elif isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            _text(item, "parameter value", 1024, empty=True)
    _encoded(value)
    return value


def _identity_hash(value):
    """Match runtime.content_hash's canonical protocol, without a newline."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _recorded_digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"Recorded {name} must be an actual SHA256 digest")
    return value.lower()


def _resources(value):
    if not isinstance(value, dict) or set(value) - RESOURCE_KEYS:
        raise ValueError("Only Tower resource request fields may be declared")
    result = {}
    for key, item in value.items():
        if item is None:
            continue
        if key in {"cpus", "nodes", "gpus", "mem_bytes"}:
            result[key] = _number(item, key, minimum=0 if key == "gpus" else 1,
                                  maximum=(1 << 63) - 1 if key == "mem_bytes" else 1_000_000_000, integer=True)
        elif key == "time_seconds":
            result[key] = _number(item, key, minimum=1e-9, maximum=3_162_240_000)
        else:
            result[key] = _text(item, key, empty=key == "gpu_type")
    return result


def _identifier(value):
    value = _text(value, "identifier", 128)
    if not ID.fullmatch(value):
        raise ValueError("Identifier must contain only ASCII letters/digits/._-")
    return value


def _attempt(value=None):
    selected = value if value is not None else os.environ.get("SSMO_TOWER_RUN_DIR")
    return _path(selected, directory=True) if selected else None


def _job_id():
    array = os.environ.get("SLURM_ARRAY_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{array}_{task}" if array and task else os.environ.get("SLURM_JOB_ID")


def start_attempt(attempt_dir, *, pipeline_dir=None, stage, method="measure", seed=None,
                  profile=None, config_path=None, source_dir=None, job_id=None,
                  resources=None, scheduler_logs=None, imported=False, state="RUNNING", metadata=None):
    """Create one exclusive attempt; requests are known values, never inferred host counts."""
    target = _path(attempt_dir, directory=True)
    stage, method = _text(stage, "stage", 64), _text(method, "method", 64)
    if state not in STATES:
        raise ValueError("Unknown lifecycle state")
    pipeline = _path(pipeline_dir, directory=True) if pipeline_dir else None
    config = _path(config_path) if config_path else None
    source = _path(source_dir, directory=True, allow_root=True) if source_dir else None
    params = {"stage": stage, "method": method}
    if seed is not None:
        params["seed"] = _number(seed, "seed", maximum=(1 << 63) - 1, integer=True)
    if profile is not None:
        params["hardware_profile"] = _text(profile, "profile", 64)
    preset = config.stem if config else "unspecified"
    if config:
        if not config.is_file() or config.stat().st_size > MAX_JSON:
            raise ValueError("Config must be a small regular file")
        params["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        params["preset"] = preset
    name = _text(f"SSMO/{stage}/{method}/{preset}", "workload name", 128)
    resource_record = _resources(resources or {})
    details = dict(metadata or {})
    details.update(stage=stage, method=method, imported_only=bool(imported))
    if pipeline:
        details["pipeline_directory"] = pipeline.relative_to(project_root()).as_posix()
    stable_identity = (details.get("source_identity") or details.get("source_directory") or target.name
                       if imported else target.relative_to(project_root()).as_posix())
    run_hash = hashlib.sha256(str(stable_identity).encode()).hexdigest()[:16]
    base = re.sub(r"[^A-Za-z0-9_.-]", "-", target.name).strip("-._") or "attempt"
    run_id = _identifier(f"SSMO-{base[:105]}-{run_hash}")
    manifest = {"schema": "tower.run/v1", "project_id": "SSMO", "run_id": run_id,
                "experiment_id": _text(pipeline.name if pipeline else base, "experiment_id"),
                "attempt": 1, "name": name, "state": state, "parameters": _parameters(params),
                "resources": resource_record, "metadata": details,
                "paths": {"metrics": "metrics.jsonl", "summary": "summary.json", "log_index": "logs.json",
                    "outputs": "outputs", "logs": "logs", "passports": "passports",
                    "stdout": "logs/application.stdout.log", "stderr": "logs/application.stderr.log"}}
    actual_job = job_id if job_id is not None else None if imported else _job_id()
    if actual_job:
        manifest["job_id"] = _text(actual_job, "job_id")
    if not imported:
        manifest["start"] = time.time()
        details["started_monotonic_ns"] = time.monotonic_ns()
    if source:
        entrypoint = _text(details.get("execution_entrypoint", "slurm/stage.sbatch"), "execution entrypoint", 4096)
        if Path(entrypoint).is_absolute():
            raise ValueError("Execution entrypoint must be source-relative")
        script = _path(source / entrypoint)
        if script.is_file():
            if script.stat().st_size > 8 << 20:
                raise ValueError("Executed entry point exceeds hash budget")
            manifest["provenance"] = {"script": script.relative_to(project_root()).as_posix(),
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    logs = [{"id": "application.stdout", "path": "logs/application.stdout.log",
             "label": "Application stdout", "group": "Application"},
            {"id": "application.stderr", "path": "logs/application.stderr.log",
             "label": "Application stderr", "group": "Application"}]
    for kind, value in (scheduler_logs or {}).items():
        if kind not in {"stdout", "stderr"}:
            raise ValueError("Scheduler logs require stdout/stderr keys")
        path = _path(value)
        logs.append({"id": f"scheduler.{kind}", "path": Path(os.path.relpath(path, target)).as_posix(),
                     "label": f"Slurm {kind}", "group": "Scheduler"})
    trace_status = os.environ.get("SSMO_TOWER_GPU_TRACE_STATUS") if not imported else None
    if trace_status:
        trace = {"status": _text(trace_status, "GPU trace status", 64), "cadence_seconds": 60,
                 "scope": "single explicit physical GPU on one verified allocated node"}
        for key, variable in (("reason", "SSMO_TOWER_GPU_TRACE_REASON"), ("selector", "SSMO_TOWER_GPU_TRACE_SELECTOR")):
            if os.environ.get(variable):
                trace[key] = _text(os.environ[variable], "GPU trace " + key, 160)
        details["gpu_trace"] = trace
        if trace_status == "eligible" and os.environ.get("SLURM_JOB_ID"):
            runtime_job = _text(os.environ["SLURM_JOB_ID"], "GPU trace job_id")
            trace_path = _path(target / f"logs/gpu-util-{runtime_job}.csv")
            logs.extend([{"id": "gpu.trace", "path": trace_path.relative_to(target).as_posix(), "label": "Allocated GPU utilization", "group": "GPU",
                          "description": "nvidia-smi timestamp,index,utilization percent,memory used MiB; one explicit physical GPU every 60 seconds."},
                         {"id": "gpu.trace.stderr", "path": "logs/gpu-trace.stderr.log", "label": "GPU trace diagnostics", "group": "GPU"}])
    normalized_logs = [os.path.normpath(target / row["path"]) for row in logs]
    if len(normalized_logs) != len(set(normalized_logs)):
        raise ValueError("Duplicate normalized log paths are refused")
    _encoded(manifest, 64 << 10)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Existing attempt content must be preserved; choose a fresh attempt directory")
    target.mkdir(parents=True, exist_ok=True)
    _atomic(target / "run.json", manifest, exclusive=True)
    for directory in ("logs", "outputs", "passports"):
        (target / directory).mkdir()
    for kind in ("stdout", "stderr"):
        (target / f"logs/application.{kind}.log").touch(exist_ok=False)
    (target / "metrics.jsonl").touch(exist_ok=False)
    index = {"schema": "tower.logs/v1", "run_id": run_id, "logs": logs}
    if actual_job:
        index["job_id"] = actual_job
    _atomic(target / "logs.json", index, exclusive=True, limit=MAX_LOG_JSON)
    _atomic(target / "outputs/analytics.json", {"schema": "ssmo.tower.analytics/v1", "run_id": run_id,
        "phases": [], "results": {}, "artifact_references": [], "limits": ["Each seed is a separate initialization of shared physical parents.",
        "Overlapping timing components must not be summed.", "GPU allocator peaks are not host/allocation memory."]}, exclusive=True)
    if not imported:
        emit_event("lifecycle", {}, attempt_dir=target)
    return manifest


def register_log(attempt_dir, log_id, path, *, label=None, group=None, description=None):
    """Index one exact durable location without reading it or recursively scanning."""
    target = _attempt(attempt_dir)
    index = _read(target / "logs.json", MAX_LOG_JSON)
    log_id = _identifier(log_id)
    given = Path(path)
    location = _path(given if given.is_absolute() else target / given)
    relative = Path(os.path.relpath(location, target)).as_posix()
    if len(index["logs"]) >= 256:
        raise ValueError("Log index exceeds 256 entries")
    for entry in index["logs"]:
        existing = (target / entry["path"]).absolute()
        if entry["id"] == log_id or os.path.normpath(existing) == os.path.normpath(location):
            raise ValueError("Duplicate log identity or normalized path")
    entry = {"id": log_id, "path": relative}
    for key, value, limit in (("label", label, 160), ("group", group, 160), ("description", description, 512)):
        if value is not None:
            entry[key] = _text(value, key, limit, empty=key == "description")
    index["logs"].append(entry)
    _atomic(target / "logs.json", index, limit=MAX_LOG_JSON)
    return entry


def emit_event(phase, metrics, step=None, completed=None, total=None, unit=None, *, attempt_dir=None):
    """Append a finite numeric JSONL record, or do nothing when reporting is disabled."""
    target = _attempt(attempt_dir)
    if target is None:
        return None
    destination = _path(target / "metrics.jsonl")
    cache_key = str(target)
    if cache_key not in _EVENT_STATE:
        if len(_EVENT_STATE) >= 256:
            _EVENT_STATE.clear()
        state = {"names": set(), "phases": {}, "last_epoch": -1.0}
        if destination.stat().st_size > MAX_METRIC_FILE:
            raise ValueError("Metric file exceeds 4 MiB")
        with destination.open("rb") as stream:
            for line in stream:
                if len(line) > MAX_METRIC_LINE or not line.endswith(b"\n"):
                    raise ValueError("Metric stream has an incomplete or oversized line")
                previous = json.loads(line)
                state["names"].update(previous.get("metrics", {}))
                state["phases"][previous.get("phase", "")] = previous
                state["last_epoch"] = previous["t"]
        _EVENT_STATE[cache_key] = state
    state = _EVENT_STATE[cache_key]
    if not isinstance(metrics, dict) or len(metrics) > 64:
        raise ValueError("Metrics require at most 64 numeric observations")
    record = {"t": _number(time.time(), "metric epoch"), "phase": _text(phase, "phase", 160, empty=True),
              "metrics": {_text(key, "metric name", 96): _number(value, key, minimum=-1e300, maximum=1e300)
                          for key, value in metrics.items()}}
    if step is not None:
        record["step"] = _number(step, "step", maximum=(1 << 63) - 1, integer=True)
    if (completed is None) != (total is None):
        raise ValueError("Progress needs both completed and total")
    if completed is not None:
        _number(total, "total", minimum=1e-300)
        record["progress"] = {"completed": _number(completed, "completed", maximum=total), "total": total}
        if unit is not None:
            record["progress"]["unit"] = _text(unit, "progress unit", 64, empty=True)
    elif unit is not None:
        raise ValueError("Progress unit requires completed/total")
    names = state["names"] | set(metrics)
    if len(names) > 64:
        raise ValueError("Attempt exceeds 64 distinct metric series")
    previous = state["phases"].get(phase, {})
    if step is not None and previous.get("step", -1) > step:
        raise ValueError("Metric steps must increase within a phase")
    if "progress" in record and "progress" in previous:
        before, after = previous["progress"], record["progress"]
        if before["total"] != after["total"] or before.get("unit") != after.get("unit") or before["completed"] > after["completed"]:
            raise ValueError("Progress must increase within a stable phase")
    record["t"] = max(record["t"], state["last_epoch"] + 0.000001)
    raw = _encoded(record, MAX_METRIC_LINE)
    if destination.stat().st_size + len(raw) > MAX_METRIC_FILE:
        raise ValueError("Metric file exceeds 4 MiB")
    with destination.open("ab") as stream:
        stream.write(raw)
        stream.flush()
    state["names"], state["last_epoch"] = names, record["t"]
    state["phases"][phase] = record
    return record


def _compact(value, depth=0):
    """Bound summaries; retain raw bulk outputs as references instead of copying them."""
    if depth > 6:
        return {"omitted": "depth bound"}
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:80]:
            if key in {"history", "timings", "parent_aggregates", "selected_parent_ids", "held_out_query_bank",
                       "artifacts", "bounded_lipschitz_diagnostics", "direction_and_chart_consistency"}:
                result[key] = {"record_count": len(item) if isinstance(item, (dict, list)) else None,
                               "detail": "original artifact reference"}
            else:
                result[key] = _compact(item, depth + 1)
        if len(value) > 80:
            result["omitted_keys"] = len(value) - 80
        return result
    if isinstance(value, (list, tuple)):
        return {"record_count": len(value), "detail": "original artifact reference"} if len(value) > 16 else [_compact(v, depth + 1) for v in value]
    return value


def _scientific_summary(result):
    compact = _compact(result)
    cohorts = result.get("parent_aggregates")
    if isinstance(cohorts, list):
        if len(cohorts) > 128:
            raise ValueError("Evaluation exceeds 128 separate scientific cohorts")
        records = []
        for cohort in cohorts:
            record = {key: cohort[key] for key in ("method", "split", "family", "physical_parents",
                "accuracy_gate_parents_passed", "accuracy_gate_parents_failed", "invalid_chart_rows") if key in cohort}
            record["error_statistics"] = {key: value for key, value in cohort.get("metrics", {}).items()
                if key in {"absolute_error_max", "relative_error_max", "nonlinear_gradient_error_max",
                           "state_l1_error_max", "diffuse_l1_error_max", "support_error_max",
                           "weight_error_max", "predicted_total_variation_max"}}
            records.append(record)
        compact["evaluation_cohorts"] = records
        compact["independent_unit"] = "physical parent within each method/split/family; seeds and controls remain separate"
    timings = result.get("timings")
    if isinstance(timings, list):
        groups = {}
        for row in timings:
            if "measurement" not in row or row.get("kind") == "components":
                continue
            measurement = row["measurement"]
            key = (row.get("method"), row.get("directions"), row.get("queries"), row.get("endpoint"),
                   measurement.get("warmup"), measurement.get("repeats"), measurement.get("cuda_synchronized"))
            group = groups.setdefault(key, {"method": row.get("method"), "directions": row.get("directions"),
                "queries": row.get("queries"), "endpoint": row.get("endpoint"), "warmup": measurement.get("warmup"),
                "repeats": measurement.get("repeats"), "cuda_synchronized": measurement.get("cuda_synchronized"), "parents": set(), "warm": [], "cold": []})
            if row.get("parent_id") in group["parents"]:
                raise ValueError("Duplicate physical parent in one endpoint workload")
            group["parents"].add(row.get("parent_id"))
            for field, destination in (("median_seconds", "warm"), ("cold_call_seconds", "cold")):
                value = measurement.get(field)
                if value is not None:
                    group[destination].append(_number(value, field, minimum=1e-300))
        if len(groups) > 64:
            raise ValueError("Evaluation exceeds 64 endpoint workload protocols")
        endpoints = []
        for group in groups.values():
            record = {key: value for key, value in group.items() if key not in {"parents", "warm", "cold"}}
            record["physical_parents"] = len(group["parents"])
            for kind in ("warm", "cold"):
                record[kind + "_parent_median_seconds"] = statistics.median(group[kind]) if len(group[kind]) == len(group["parents"]) and group[kind] else None
            endpoints.append(record)
        compact["endpoint_workloads"] = endpoints
        compact["endpoint_timing_scope"] = "Per-method workload/protocol summaries of recorded endpoints; CPU controls and synchronized GPU endpoints stay separate. Cold means first endpoint call, not process/model startup. No summed components or accuracy-qualified speedup."
    if "inverse" in result:
        compact["inverse_scope"] = "Recorded trusted audit; exact diagnostic gradients every iteration and exact-objective acceptance included. Not autonomous learned-optimizer speedup."
    return compact


def report_phase(command, result, artifact_dir, config=None, dataset=None, elapsed_seconds=None,
                 provenance_record=None, *, attempt_dir=None):
    """Attach compact scientific results and exact references; never recalculate evidence."""
    target = _attempt(attempt_dir)
    if target is None:
        return None
    command = _text(command, "command", 64)
    artifact = _path(artifact_dir, directory=True)
    if not isinstance(result, dict):
        raise ValueError("Scientific phase results must be an object")
    manifest = _read(target / "run.json")
    analytics = _read(target / "outputs/analytics.json")
    if len(analytics["phases"]) >= 32:
        raise ValueError("Too many phases for one execution attempt")
    phase = {"command": command, "results": _scientific_summary(result)}
    if elapsed_seconds is not None:
        phase["runtime_seconds"] = _number(elapsed_seconds, "phase runtime")
    analytics["phases"].append(phase)
    analytics["results"][command] = phase["results"]
    known_names = ("training.jsonl", "training.jsonl.gz", "training_summary.json", "evaluation.json",
        "parents.json", "query_errors.jsonl", "query_errors.jsonl.gz", "representation.jsonl",
        "representation.jsonl.gz", "parents_metrics.jsonl", "parents_metrics.jsonl.gz", "representation.json",
        "audit.json", "numerical.json", "gpu_audit.json", "hardware.json", "inverse.json", "report.json",
        "provenance.json", "failure.json", "tests.log", "evaluation_parents.jsonl", "evaluation_parents.jsonl.gz",
        "numeric_teacher_audit.json", "numeric_teacher_audit.jsonl", "numeric_teacher_audit.jsonl.gz",
        "log-compaction.json", "summary.json", "summary.txt", "failures.jsonl",
        "SHA256SUMS", "source-checksums.sha256", "source-index.tsv")
    index = _read(target / "logs.json", MAX_LOG_JSON)
    for name in known_names:
        path = artifact / name
        if not path.exists():
            continue
        path = _path(path)
        reference = {"path": Path(os.path.relpath(path, target)).as_posix(), "bytes": path.stat().st_size,
                     "kind": "scientific_artifact", "command": command}
        if not any(item["path"] == reference["path"] for item in analytics["artifact_references"]):
            analytics["artifact_references"].append(reference)
        log_id = "science." + re.sub(r"[^A-Za-z0-9_.-]", "-", command) + "." + name.replace(".jsonl.gz", ".jsonl-compressed")
        known_paths = {os.path.normpath(target / entry["path"]) for entry in index["logs"]}
        if not name.endswith(".gz") and os.path.normpath(path) not in known_paths:
            register_log(target, log_id, path, label=name, group="Scientific evidence",
                         description="Original artifact; compressed JSONL remains compressed." if name.endswith(".gz") else "Original artifact; no scientific recalculation.")
            index = _read(target / "logs.json", MAX_LOG_JSON)
    parameters = manifest["parameters"]
    recorded = provenance_record or {}
    if recorded.get("config_sha256") is not None:
        parameters["effective_config_sha256"] = _recorded_digest(recorded["config_sha256"], "config identity")
    elif config is not None:
        parameters["effective_config_sha256"] = _identity_hash(config)
    if recorded.get("manifest_sha256") is not None:
        parameters["dataset_sha256"] = _recorded_digest(recorded["manifest_sha256"], "manifest identity")
    elif dataset is not None:
        parameters["dataset_sha256"] = _identity_hash(dataset)
    if dataset is not None:
        if isinstance(dataset, dict) and isinstance(dataset.get("parents"), list):
            manifest["input_size"] = len(dataset["parents"])
            manifest["metadata"]["input_size_definition"] = "physical parents in declared stage manifest; shared across initialization seeds"
    if provenance_record:
        source = provenance_record.get("source", {})
        if source.get("source_sha256"):
            parameters["source_sha256"] = source["source_sha256"]
        if source.get("git_commit"):
            manifest.setdefault("provenance", {})["git_commit"] = source["git_commit"]
    software_identity = {}
    if recorded.get("python"):
        software_identity["python"] = _text(recorded["python"], "recorded Python version", 128)
    if isinstance(recorded.get("software"), dict):
        versions = {name: value for name, value in recorded["software"].items() if value is not None}
        if versions:
            software_identity["software"] = versions
    if software_identity:
        parameters["software_sha256"] = _identity_hash(_parameters(software_identity))
    hardware = result.get("hardware")
    if not isinstance(hardware, dict) and command == "gpu-audit":
        hardware = result
    if not isinstance(hardware, dict):
        for name in ("hardware.json", "gpu_audit.json"):
            path = artifact / name
            if path.exists():
                hardware = _read(path)
                break
    if isinstance(hardware, dict):
        hardware_identity = {key: hardware[key] for key in ("device", "model", "capability", "gpu_name",
            "total_memory_bytes", "gpu_total_bytes") if hardware.get(key) is not None}
        if hardware_identity:
            parameters["hardware_sha256"] = _identity_hash(_parameters(hardware_identity))
    if not manifest["metadata"].get("imported_only"):
        phase_rss = (provenance_record or {}).get("memory", {}).get("host_peak_rss_bytes")
        if phase_rss is None:
            phase_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        phase_rss = _number(phase_rss, "application process RSS")
        previous = manifest["metadata"].get("application_observations", {})
        observation = {"memory_bytes": max(previous.get("memory_bytes", 0), phase_rss),
                       "memory_scope": "max_task_rss", "cpu_scope": "sum of explicitly measured application command windows; excludes children and batch launcher",
                       "memory_source": "getrusage(RUSAGE_SELF).ru_maxrss; single application process peak"}
        phase_cpu = (provenance_record or {}).get("process_cpu_seconds")
        if phase_cpu is not None:
            observation["cpu_seconds"] = previous.get("cpu_seconds", 0) + _number(phase_cpu, "application command CPU seconds")
        elif "cpu_seconds" in previous:
            observation["cpu_seconds"] = previous["cpu_seconds"]
        manifest["metadata"]["application_observations"] = observation
        metrics = {"application_process_peak_rss_bytes": phase_rss}
        if phase_cpu is not None:
            metrics["application_command_cpu_seconds"] = phase_cpu
        if elapsed_seconds is not None:
            metrics["phase_runtime_seconds"] = elapsed_seconds
        for key in ("steps_completed", "best_step", "best_validation_score", "training_teacher_seconds", "validation_teacher_seconds",
                    "peak_allocated_bytes", "peak_reserved_bytes", "host_peak_rss_bytes", "physical_parents", "tests_executed", "failures", "errors"):
            item = result.get(key)
            if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
                metrics[key] = item
        emit_event(command, metrics, attempt_dir=target)
    manifest["parameters"] = _parameters(parameters)
    _atomic(target / "outputs/analytics.json", analytics, limit=MAX_SMALL_JSON)
    _atomic(target / "run.json", manifest, limit=64 << 10)
    return phase


def finish_attempt(attempt_dir, exit_code, *, state=None, summary_fields=None):
    """Finalize observed process status; a signal alone never proves timeout/OOM."""
    target = _attempt(attempt_dir)
    manifest = _read(target / "run.json")
    if (target / "summary.json").exists():
        raise FileExistsError("Attempt already finalized; preserve its summary")
    if exit_code is not None:
        _number(exit_code, "exit_code", minimum=-(1 << 31), maximum=(1 << 31) - 1, integer=True)
    elif state is None:
        raise ValueError("An unknown exit code requires an explicitly evidenced state")
    final_state = state or ("COMPLETED" if exit_code == 0 else "INTERRUPTED" if exit_code in {75, 129, 130, 143} else "FAILED")
    allowed_states = STATES if manifest["metadata"].get("imported_only") else TERMINAL_STATES
    if final_state not in allowed_states:
        raise ValueError("Final state must be terminal and explicitly evidenced")
    fields = dict(summary_fields or {})
    if set(fields) - SUMMARY_KEYS:
        raise ValueError("Unknown final-summary fields")
    summary = {"schema": "tower.summary/v1", "id": manifest["run_id"], "project_id": "SSMO",
               "name": manifest["name"], "state": final_state, "experiment_id": manifest["experiment_id"],
               "attempt": manifest["attempt"], "parameters": manifest["parameters"],
               "metadata": manifest["metadata"], "results": {}, **manifest["resources"]}
    if exit_code is not None:
        summary["exit_code"] = exit_code
    for key in ("job_id", "start", "end", "submit", "input_size"):
        if key in manifest:
            summary[key] = manifest[key]
    if manifest.get("provenance", {}).get("script_sha256"):
        summary["script_sha256"] = manifest["provenance"]["script_sha256"]
    if not manifest["metadata"].get("imported_only"):
        summary["end"] = time.time()
        started = manifest["metadata"].get("started_monotonic_ns")
        if started is not None:
            summary["runtime_seconds"] = max(0, (time.monotonic_ns() - started) / 1e9)
        observation = manifest["metadata"].get("application_observations", {})
        # The batch launcher's own RSS/CPU is never substituted for the child task.
        for key in ("cpu_seconds", "memory_bytes", "memory_scope"):
            if key in observation:
                summary[key] = observation[key]
    analytics = _read(target / "outputs/analytics.json")
    for phase in analytics["phases"]:
        summary["results"][phase["command"]] = phase["results"]
    summary.update(fields)
    _resources({key: summary[key] for key in RESOURCE_KEYS if key in summary})
    for key in ("runtime_seconds", "cpu_seconds", "memory_bytes", "input_size"):
        if summary.get(key) is not None:
            _number(summary[key], key)
    for key in ("start", "end", "submit"):
        if summary.get(key) is not None:
            _number(summary[key], key, maximum=253402300799)
    if summary.get("memory_scope") not in {None, "job_peak", "per_node_peak", "max_task_rss"}:
        raise ValueError("Unknown observed-memory scope")
    if summary.get("memory_bytes") is not None and not summary.get("memory_scope"):
        raise ValueError("Observed memory requires an explicit measurement scope")
    _parameters(summary["parameters"])
    if summary.get("script_sha256") and not re.fullmatch(r"[0-9a-fA-F]{64}", summary["script_sha256"]):
        raise ValueError("Script digest must be actual SHA256")
    _encoded(summary, MAX_SMALL_JSON)
    manifest["state"] = final_state
    if "end" in summary:
        manifest["end"] = summary["end"]
    manifest["results"] = {"analytics": "outputs/analytics.json", "phases_recorded": len(analytics["phases"])}
    analytics["state"] = final_state
    analytics["exit_code"] = exit_code
    if not manifest["metadata"].get("imported_only"):
        emit_event("lifecycle", {"exit_code": exit_code} if exit_code is not None else {}, attempt_dir=target)
    _atomic(target / "outputs/analytics.json", analytics, limit=MAX_SMALL_JSON)
    _atomic(target / "summary.json", summary, exclusive=True, limit=MAX_SMALL_JSON)
    _atomic(target / "run.json", manifest, limit=64 << 10)
    return summary


def _walltime(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if not match:
        raise ValueError("Requested time must be [days-]HH:MM:SS")
    days, hours, minutes, seconds = (int(v or 0) for v in match.groups())
    if minutes >= 60 or seconds >= 60 or days and hours >= 24:
        raise ValueError("Invalid Slurm walltime")
    return 86400 * days + 3600 * hours + 60 * minutes + seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start")
    start.add_argument("--attempt-dir", required=True)
    start.add_argument("--pipeline-dir")
    start.add_argument("--stage", required=True)
    start.add_argument("--method", default="measure")
    start.add_argument("--seed", type=int)
    start.add_argument("--profile")
    start.add_argument("--config")
    start.add_argument("--source-dir")
    start.add_argument("--job-id")
    for key in ("cpus", "nodes", "gpus", "mem-gb"):
        start.add_argument("--request-" + key, type=float if key == "mem-gb" else int)
    for key in ("time", "gpu-type", "partition", "account", "qos"):
        start.add_argument("--request-" + key)
    start.add_argument("--scheduler-out")
    start.add_argument("--scheduler-err")
    finish = sub.add_parser("finish")
    finish.add_argument("--attempt-dir", required=True)
    finish.add_argument("--exit-code", type=int, required=True)
    event = sub.add_parser("event")
    event.add_argument("--attempt-dir", required=True)
    event.add_argument("--phase", required=True)
    event.add_argument("--metrics", default="{}")
    args = parser.parse_args(argv)
    try:
        if args.action == "start":
            resources = {}
            defaults = {"cpus": "SSMO_TOWER_CPUS", "gpus": "SSMO_TOWER_GPU_COUNT", "mem_gb": "SSMO_TOWER_MEM_GB",
                        "time": "SSMO_TOWER_WALLTIME", "gpu_type": "SSMO_TOWER_GPU_TYPE"}
            for key in ("cpus", "nodes", "gpus", "mem_gb", "time", "gpu_type", "partition", "account", "qos"):
                value = getattr(args, "request_" + key)
                if value is None and key in defaults:
                    value = os.environ.get(defaults[key])
                if value is None and key in {"partition", "account", "qos"}:
                    value = os.environ.get({"partition": "SLURM_JOB_PARTITION", "account": "SLURM_JOB_ACCOUNT", "qos": "SLURM_JOB_QOS"}[key])
                if value is not None:
                    if key in {"cpus", "nodes", "gpus"}:
                        resources[key] = int(value)
                    elif key == "mem_gb":
                        resources["mem_bytes"] = int(_number(float(value), "request memory", minimum=1e-9) * 2**30)
                    elif key == "time":
                        resources["time_seconds"] = _walltime(value)
                    else:
                        resources[key] = value
            scheduler = {}
            for kind, selected in (("stdout", args.scheduler_out or os.environ.get("SSMO_TOWER_SCHEDULER_OUT")),
                                   ("stderr", args.scheduler_err or os.environ.get("SSMO_TOWER_SCHEDULER_ERR"))):
                if selected:
                    scheduler[kind] = selected
            start_attempt(args.attempt_dir, pipeline_dir=args.pipeline_dir, stage=args.stage, method=args.method,
                seed=args.seed, profile=args.profile, config_path=args.config, source_dir=args.source_dir,
                job_id=args.job_id, resources=resources, scheduler_logs=scheduler)
        elif args.action == "finish":
            finish_attempt(args.attempt_dir, args.exit_code)
        else:
            emit_event(args.phase, json.loads(args.metrics), attempt_dir=args.attempt_dir)
        return 0
    except (ValueError, OSError, TypeError) as error:
        print(f"SSMO Tower reporting: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
