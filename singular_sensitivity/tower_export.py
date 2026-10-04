"""Explicit, read-only imports and bounded planning exports for Tower.

This producer does not import Tower or scientific dependencies. Historical
artifacts remain untouched; import time is never an experiment start time.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import statistics
import subprocess
import time

from .runtime import project_root

MAX_JSON = 32 << 20  # Existing project summaries, not Tower interchange files.
MAX_PLANNING = 1 << 20
MAX_ATTEMPTS = 256
JOB_ID = re.compile(r"[0-9]+(?:_[0-9]+)?")
IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
STATES = {"CREATED", "SUBMITTED", "PENDING", "RUNNING", "COMPLETING", "COMPLETED",
          "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
          "BOOT_FAIL", "DEADLINE", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD",
          "RESIZING", "REVOKED", "SIGNALING", "SPECIAL_EXIT", "STAGE_OUT", "STOPPED",
          "SUSPENDED", "UNKNOWN", "INTERRUPTED"}
QUERY_KEYS = {"name", "partition", "account", "qos", "cpus", "nodes", "gpus", "gpu_type",
              "mem_bytes", "time_seconds", "script_sha256", "input_size", "parameters", "memory_scope"}


def plain_path(root: Path, value: str | Path, *, exists=False) -> Path:
    raw = str(value)
    if (not raw or not raw.isprintable() or "\\" in raw or any(c in raw for c in "*?[]")
            or any(piece in {".", ".."} for piece in raw.split("/")) or "//" in raw):
        raise ValueError("choose an exact project path without traversal or globs")
    path = Path(raw)
    path = path if path.is_absolute() else root / path
    if path == root or not path.is_relative_to(root):
        raise ValueError("path must be strictly inside SSMO")
    for component in reversed((path, *path.parents)):
        if component.exists() or component.is_symlink():
            mode = component.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink refused: {component}")
            if component != path and not stat.S_ISDIR(mode):
                raise ValueError(f"non-directory ancestor: {component}")
    if exists and not path.exists():
        raise FileNotFoundError(path)
    return path


def encoded(value, limit=MAX_PLANNING):
    stack, count = [(value, 0)], 0
    while stack:
        child, depth = stack.pop()
        count += 1
        if depth > 32 or count > 100000:
            raise ValueError("JSON exceeds depth/value budget")
        if isinstance(child, dict):
            stack.extend((v, depth + 1) for v in child.values())
        elif isinstance(child, list):
            stack.extend((v, depth + 1) for v in child)
        elif isinstance(child, int) and not isinstance(child, bool) and child.bit_length() > 256:
            raise ValueError("JSON integer exceeds bounded reader profile")
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > limit:
        raise ValueError(f"JSON exceeds {limit} byte budget")
    return raw


def read_bytes(root, value, *, limit=MAX_JSON):
    path = plain_path(root, value, exists=True)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError(f"bounded regular file required: {path}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        after = os.fstat(fd)
        named = path.lstat()
        if len(raw) != before.st_size or len(raw) > limit or signature(before) != signature(after) or signature(after) != signature(named):
            raise ValueError(f"source changed during read: {path}")
    finally:
        os.close(fd)
    return raw


def read_json(root, value, *, limit=MAX_JSON, receipts=None):
    def pairs(items):
        obj = {}
        for key, child in items:
            if key in obj:
                raise ValueError("duplicate JSON keys")
            obj[key] = child
        return obj
    raw = read_bytes(root, value, limit=limit)
    if receipts is not None:
        receipts[str(plain_path(root, value).relative_to(root))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    obj = json.loads(raw, object_pairs_hook=pairs,
                     parse_constant=lambda token: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    encoded(obj, limit)
    if not isinstance(obj, dict):
        raise ValueError("JSON object required")
    return obj


def write_json(root, path, value, *, replace=False):
    path = plain_path(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plain_path(root, path.parent, exists=True)
    if path.exists() and (not replace or not stat.S_ISREG(path.lstat().st_mode)):
        raise FileExistsError(path)
    raw = encoded(value)
    temporary = path.parent / ("." + path.name + "." + str(os.getpid()) + ".new")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path)
        temporary.unlink()


def job_rows(root, run, receipts=None):
    raw = read_bytes(root, run / "jobs.tsv", limit=256 << 10)
    if receipts is not None:
        receipts[str((run / "jobs.tsv").relative_to(root))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t"))
    if not rows or len(rows) > MAX_ATTEMPTS:
        raise ValueError("jobs.tsv must contain 1..256 actual stage attempts")
    seen = set()
    for row in rows:
        if set(row) != {"stage", "method", "seed", "job_id", "dependency"}:
            raise ValueError("unsupported jobs.tsv fields")
        if not JOB_ID.fullmatch(row["job_id"]) or row["job_id"] in seen:
            raise ValueError("jobs.tsv must contain distinct actual Slurm job IDs")
        if not IDENT.fullmatch(row["stage"]) or not IDENT.fullmatch(row["method"]) or not row["seed"].isdigit():
            raise ValueError("invalid stage/method/seed identity")
        seen.add(row["job_id"])
    return rows


def duration(value):
    if not value or value in {"Unknown", "N/A", "UNLIMITED"}:
        return None
    days, sep, remainder = value.partition("-")
    if not sep:
        remainder, days = days, "0"
    pieces = remainder.split(":")
    if len(pieces) == 1:
        return float(value)
    if len(pieces) not in {2, 3}:
        raise ValueError("invalid scheduler duration")
    seconds = float(pieces[-1]) + 60 * int(pieces[-2])
    if len(pieces) == 3:
        seconds += 3600 * int(pieces[0])
    return seconds + int(days) * 86400


def memory_bytes(value):
    if not value or value in {"Unknown", "N/A"}:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)(?:B)?", value, re.I)
    if not match:
        raise ValueError("invalid scheduler memory quantity")
    value = float(match[1]) * 1024 ** ("KMGTPE".index(match[2].upper()) + 1 if match[2] else 0)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError("scheduler memory must normalize to finite whole bytes")
    return int(value)


def parse_accounting(raw: str):
    """Read real parsable2 sacct rows; MaxRSS is a maximum task, never job peak."""
    utc = raw.startswith("# ssmo-sacct-timezone=UTC\n")
    if utc:
        raw = raw.split("\n", 1)[1]
    rows = list(csv.DictReader(io.StringIO(raw), delimiter="|"))
    records, steps = {}, {}
    for row in rows:
        if "JobIDRaw" not in row or "State" not in row:
            raise ValueError("accounting requires JobIDRaw and State headers")
        ident = row["JobIDRaw"]
        if "." in ident:
            base = ident.split(".", 1)[0]
            if not JOB_ID.fullmatch(base):
                raise ValueError("invalid scheduler step identity")
            rss = memory_bytes(row.get("MaxRSS", ""))
            if rss is not None:
                steps.setdefault(base, []).append(rss)
            continue
        if not JOB_ID.fullmatch(ident) or ident in records:
            raise ValueError("duplicate/invalid scheduler parent identity")
        state = row["State"].split(" ", 1)[0].rstrip("+")
        if state not in STATES:
            raise ValueError(f"unsupported actual scheduler state: {state}")
        fields = {}
        for key, source in (("partition", "Partition"), ("account", "Account"), ("qos", "QOS")):
            if row.get(source) and row[source] not in {"Unknown", "N/A"}:
                fields[key] = row[source]
        for key, source in (("cpus", "AllocCPUS"), ("nodes", "NNodes")):
            if row.get(source):
                value = int(row[source])
                if value > 0:
                    fields[key] = value
        for key, source in (("runtime_seconds", "ElapsedRaw"), ("cpu_seconds", "TotalCPU")):
            value = duration(row.get(source, ""))
            if value is not None:
                fields[key] = value
        wall = row.get("TimelimitRaw", "")
        if wall.isdigit():
            fields["time_seconds"] = int(wall) * 60
        tres = dict(piece.split("=", 1) for piece in row.get("AllocTRES", "").split(",") if "=" in piece)
        if "gres/gpu" in tres:
            fields["gpus"] = int(tres["gres/gpu"])
        elif tres.get("cpu") and not any(k.startswith("gres/gpu") for k in tres):
            fields["gpus"] = 0
        types = [key.removeprefix("gres/gpu:") for key in tres if key.startswith("gres/gpu:")]
        if len(types) == 1:
            fields["gpu_type"] = types[0]
            fields.setdefault("gpus", int(tres["gres/gpu:" + types[0]]))
        request = row.get("ReqMem", "")
        if request and request[-1:] in {"n", "c"}:
            factor = fields.get("nodes" if request[-1] == "n" else "cpus")
            if factor is not None:
                value = memory_bytes(request[:-1])
                if value is not None and value > 0:
                    fields["mem_bytes"] = value * factor
        if utc:
            for key, source in (("start", "Start"), ("end", "End"), ("submit", "Submit")):
                value = row.get(source, "")
                if value and value not in {"Unknown", "N/A", "None"}:
                    fields[key] = datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
        exit_raw = row.get("ExitCode", "")
        exit_code = None
        if re.fullmatch(r"[0-9]+:[0-9]+", exit_raw):
            code, signal = map(int, exit_raw.split(":"))
            if signal == 0:
                exit_code = code
            if state == "COMPLETED" and (code or signal):
                raise ValueError("scheduler COMPLETED contradicts ExitCode")
        rss = memory_bytes(row.get("MaxRSS", ""))
        if rss is not None:
            steps.setdefault(ident, []).append(rss)
        encoded(fields)
        records[ident] = {"state": state, "exit_code": exit_code, "fields": fields, "slurm_exit_code": exit_raw}
    for ident, record in records.items():
        if steps.get(ident):
            record["fields"].update(memory_bytes=max(steps[ident]), memory_scope="max_task_rss")
    return records


def artifact_directory(run, row):
    stage = row["stage"]
    if stage in {"train", "evaluate"}:
        return run / "artifacts" / f"{stage}-{row['method']}-seed{row['seed']}"
    return run / "artifacts" / ({"report": "", "install": ""}.get(stage, stage))


def phase_files(stage):
    return {"validate": [("audit", "audit.json")], "data": [("generate", "parents.json")],
            "gpu-smoke": [("gpu-audit", "gpu_audit.json")], "train": [("train", "training_summary.json")],
            "evaluate": [("evaluate", "evaluation.json")],
            "report": [("representation", "representation/representation.json"),
                       ("numerical", "numerical/numeric_teacher_audit.json"),
                       ("inverse", "inverse/inverse.json"), ("compaction", "log-compaction.json"),
                       ("report", "report.json")],
            "install": []}.get(stage, [])


def compact_result(command, result):
    """Keep aggregate evidence visible without copying individual query logs."""
    if command == "evaluate":
        compact = {key: result[key] for key in ("status", "physical_parents", "independent_unit",
                   "wall_seconds", "accuracy_gate", "fixed_audit_coverage", "baseline_training", "memory", "limitations") if key in result}
        cohorts = {}
        for row in result.get("parent_aggregates", []):
            method = row["method"]
            identity = row["split"] + "/" + row["family"]
            record = {key: row[key] for key in ("physical_parents", "accuracy_gate_parents_passed",
                      "accuracy_gate_parents_failed", "invalid_chart_rows") if key in row}
            for metric, measurement in row.get("metrics", {}).items():
                for statistic in ("mean", "median", "q90", "worst"):
                    if statistic in measurement:
                        record[metric + "_" + statistic] = measurement[statistic]
            cohorts.setdefault(method, {})[identity] = record
        compact["method_cohorts"] = cohorts
        diagnostics = {}
        for row in result.get("direction_and_chart_consistency", []):
            method = row.get("method", "unspecified")
            record = diagnostics.setdefault(method, {"records": 0})
            record["records"] += 1
            for key, value in row.items():
                if key in {"parent_id", "method", "time", "direction_index", "h"}:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    record[key + "_max"] = max(record.get(key + "_max", value), value)
        compact["chart_consistency_maxima"] = diagnostics
        timing_groups = {}
        for row in result.get("timings", []):
            if "measurement" not in row:
                continue
            measurement = row["measurement"]
            key = (row["method"], row["directions"], row["queries"], row.get("endpoint"),
                   measurement.get("warmup"), measurement.get("repeats"), measurement.get("cuda_synchronized"))
            timing_groups.setdefault(key, []).append(row)
        timings = {}
        for (method, directions, queries, endpoint, warmup, repeats, synced), rows in timing_groups.items():
            measured = [row["measurement"] for row in rows]
            if len({row["parent_id"] for row in rows}) != len(rows):
                raise ValueError("duplicate physical parent in one endpoint workload/protocol")
            modes = {m.get("cuda_synchronized") for m in measured}
            record = {"physical_parents": len({row["parent_id"] for row in rows}),
                      "records": len(rows), "directions": directions, "queries": queries,
                      "device_scope": "CUDA synchronized endpoint" if modes == {True} else "CPU endpoint" if modes == {False} else "mixed/unknown",
                      "endpoint": endpoint, "warmup": warmup, "repeats": repeats,
                      "scope": "descriptive per-parent endpoint medians; not paired speedup or added components"}
            for name, source in (("warm_seconds", "median_seconds"), ("first_call_seconds", "cold_call_seconds")):
                values = [m[source] for m in measured if source in m and isinstance(m[source], (int, float)) and not isinstance(m[source], bool)]
                if values:
                    record[name + "_median"] = statistics.median(values)
                    record[name + "_minimum"] = min(values)
                    record[name + "_maximum"] = max(values)
                    record[name + "_record_count"] = len(values)
            protocol = hashlib.sha256(encoded([endpoint, warmup, repeats, synced])).hexdigest()[:12]
            timings.setdefault(method, {})[f"directions{directions}-queries{queries}-{protocol}"] = record
        compact["endpoint_measurements"] = timings
        compact["inverse_audits"] = {method: {key: value for key, value in row.items() if key != "history"}
                                      for method, row in result.get("inverse", {}).items()}
        compact["unresolved_reference_records"] = len(result.get("unresolved_records", []))
        compact["raw_detail_scope"] = "individual parents/queries/timing repeats/components remain in exact original artifact references"
        return compact
    if command.startswith("imported-") and isinstance(result.get("runs"), list):
        records, method_records, attribution_records, endpoint_records = {}, {}, {}, {}
        for run in result["runs"]:
            if not isinstance(run, dict) or "run_id" not in run:
                continue
            record = {key: run[key] for key in ("seed", "profile", "source_revision", "gpu_stratum", "classical") if key in run}
            for method, data in run.get("methods", {}).items():
                details = {key: data[key] for key in ("training", "total", "fixed_audit_coverage",
                           "baseline_costs", "inverse") if key in data}
                details["cohorts"] = {row["split"] + "/" + row["family"]:
                    {**{key: value for key, value in row.items() if key not in {"worst_errors", "split", "family"}},
                     **row.get("worst_errors", {})} for row in data.get("cohorts", [])}
                comparisons = {}
                for comparison in data.get("endpoint_costs", {}).get("comparisons", []):
                    groups = {}
                    for workload in comparison.get("workloads", []):
                        metrics = {key: workload[key] for key in ("directions", "queries", "matched_physical_parents",
                                   "learned_device", "control_device") if key in workload}
                        for mode in ("warm", "cold"):
                            measurement = workload.get(mode, {})
                            metrics[mode + "_status"] = measurement.get("status")
                            for entity in ("learned_over_control", "learned_seconds", "control_seconds"):
                                for statistic, value in (measurement.get(entity) or {}).items():
                                    metrics[mode + "_" + entity + "_" + statistic] = value
                        workload_id = f"directions{workload.get('directions')}-queries{workload.get('queries')}"
                        groups[workload_id] = metrics
                        endpoint_records[run["run_id"] + "." + method + "." + comparison["control"] + "." + workload_id] = {
                            "run_id": run["run_id"], "seed": run.get("seed"), "method": method,
                            "control": comparison["control"], **metrics}
                    comparisons[comparison["control"]] = {"status": comparison["status"], "workload_count": len(groups)}
                details["endpoint_comparisons"] = comparisons
                details.update(run_id=run["run_id"], seed=run.get("seed"), method=method)
                method_records[run["run_id"] + "." + method] = details
            for parent in run.get("parents", []):
                for method, data in parent.get("methods", {}).items():
                    attributed = {key: value for key, value in data.items() if not key.startswith("worst_")}
                    attributed.update(run_id=run["run_id"], seed=run.get("seed"), parent_id=parent["parent_id"],
                                      method=method, split=parent.get("split"), family=parent.get("family"))
                    for label in ("weak", "nonlinear"):
                        worst = data.get("worst_" + label + "_row")
                        if not worst:
                            continue
                        raw, ledger = worst["raw"], worst["attribution"]
                        item = {key: raw[key] for key in ("time", "direction_index", "query_index", "absolute_error",
                                "nonlinear_gradient_error", "predicted_position", "predicted_atom_weight") if key in raw}
                        item["query_kind"] = raw.get("query", {}).get("kind")
                        source = ledger if label == "weak" else ledger["nonlinear"]
                        item.update(signed_error=source["signed_error"], contributions=source["contributions"])
                        attributed["worst_" + label] = item
                    attribution_records[run["run_id"] + "." + parent["parent_id"] + "." + method] = attributed
            records[run["run_id"]] = record
        return {"runs_by_id": records, "method_records": method_records, "attribution_records": attribution_records,
                "endpoint_records": endpoint_records,
                "independent_unit": result.get("independent_unit"), "status": result.get("status"),
                "decomposition": result.get("decomposition"),
                "scope": "separate seeds/methods; overlapping components and repeated physical parents are not pooled"}
    if command == "report":
        return {"status": result.get("status"), "claims": result.get("claims"),
                "collected_artifact_count": len(result.get("artifacts", [])),
                "detail": "phase summaries and exact report.json reference"}
    if command == "inverse":
        return {key: value for key, value in result.items() if key != "history"}
    return result


def import_metrics(command, result):
    metrics = {}
    for key in ("elapsed_seconds", "wall_seconds", "total_seconds", "steps_completed", "best_step",
                "best_validation_score", "parents", "physical_parents", "accepted_steps",
                "initial_trusted_objective", "final_trusted_objective", "trusted_objective_evaluations",
                "trusted_gradient_reference_evaluations", "exact_gradient_fallbacks", "tests_executed",
                "failures", "errors", "max_conservation_residual", "max_range_violation"):
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            metrics[key] = value
    return metrics


def application_state(phases, provenance, *, ambiguous=False):
    if ambiguous:
        return "UNKNOWN", None, "shared legacy artifact directory cannot identify a retry"
    failure = next((obj for command, obj, path in phases if command == "failure"), None)
    if failure:
        return "FAILED", 1, "application failure artifact"
    if not provenance:
        return "UNKNOWN", None, "no final application provenance; scheduler state unobserved"
    terminal = phases[-1][1] if phases else {}
    if terminal.get("status") == "paused":
        return "INTERRUPTED", 75, "application recorded a paused optimizer boundary"
    if terminal.get("status") == "failed":
        return "FAILED", 1, "application recorded a failed terminal result"
    if phases:
        return "COMPLETED", 0, "terminal application artifact and provenance; scheduler state unobserved"
    return "UNKNOWN", None, "no recognized terminal artifact"


def import_runs(root, run_ids, artifact_dirs, output, *, accounting_file=None, accounting=False):
    from . import tower_reporting as reporter
    if not run_ids and not artifact_dirs:
        raise ValueError("select --run IDs and/or --artifact-dir paths explicitly")
    if len(set(run_ids)) != len(run_ids) or len(set(map(str, artifact_dirs))) != len(artifact_dirs):
        raise ValueError("duplicate explicit source selection")
    if accounting and accounting_file:
        raise ValueError("choose --accounting or --accounting-file")
    target = plain_path(root, output)
    if target.exists():
        raise FileExistsError("choose a fresh Tower export directory")
    sources, count, receipts = [], 0, {}
    for run_id in run_ids:
        if not IDENT.fullmatch(run_id):
            raise ValueError("--run requires an exact run ID")
        run = plain_path(root, root / "runs" / run_id, exists=True)
        rows = job_rows(root, run, receipts)
        sources.append((run, rows))
        count += len(rows)
    reviews = [plain_path(root, path, exists=True) for path in artifact_dirs]
    count += len(reviews)
    if count > MAX_ATTEMPTS:
        raise ValueError("at most 256 attempts may be imported")
    for source in [run for run, rows in sources] + reviews:
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError("export must not overlap original evidence")
    records = {}
    if accounting_file:
        records = parse_accounting(read_bytes(root, accounting_file, limit=MAX_PLANNING).decode())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    if accounting:
        ids = [row["job_id"] for run, rows in sources for row in rows]
        if not ids:
            raise ValueError("--accounting requires selected pipeline jobs")
        command = ["sacct", "--parsable2", "--units=K", "--jobs=" + ",".join(ids),
                   "--format=JobIDRaw,State,ExitCode,ElapsedRaw,TotalCPU,AllocCPUS,NNodes,AllocTRES,ReqMem,TimelimitRaw,Partition,Account,QOS,Submit,Start,End,MaxRSS"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                env={**os.environ, "TZ": "UTC"})
        raw = "# ssmo-sacct-timezone=UTC\n" + result.stdout
        (target / "accounting.txt").write_text(raw)
        (target / "accounting.stderr.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"sacct failed ({result.returncode}); captured evidence preserved in {target}")
        if len(raw.encode()) > MAX_PLANNING:
            raise ValueError("accounting capture exceeds 1 MiB")
        records = parse_accounting(raw)
    attempts = []
    for run, rows in sources:
        keys = [(r["stage"], r["method"], r["seed"]) for r in rows]
        for row in rows:
            identity = f"{run.name}-{row['stage']}-{row['method']}-seed{row['seed']}-job{row['job_id']}"
            if len(identity) > 110:
                identity = identity[:80] + "-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            attempt = target / ("ssmo-" + identity)
            artifact = plain_path(root, artifact_directory(run, row))
            phases, provenance = [], None
            for command, relative in phase_files(row["stage"]):
                path = plain_path(root, artifact / relative)
                if path.exists():
                    value = read_json(root, path, receipts=receipts)
                    if command == "generate":
                        value = {"status": "generated", "parents": len(value.get("parents", [])),
                                 "config_hash": value.get("config_hash")}
                    phases.append((command, value, path))
            provenance_path = plain_path(root, artifact / "provenance.json")
            if provenance_path.exists():
                provenance = read_json(root, provenance_path, receipts=receipts)
            ambiguous = keys.count((row["stage"], row["method"], row["seed"])) > 1
            recorded_job = (provenance or {}).get("slurm", {}).get("SLURM_JOB_ID")
            if recorded_job:
                ambiguous = str(recorded_job) != row["job_id"]
            if ambiguous:
                phases = []
                provenance = None
            failure_paths = []
            if artifact.exists():
                failure_paths = sorted(path for path in artifact.iterdir()
                                       if path.name == "failure.json" or re.fullmatch(r"failure-[0-9]+\.json", path.name))
            if len(failure_paths) > 16:
                raise ValueError("failure artifact count exceeds bounded import profile")
            for path in failure_paths:
                failure = read_json(root, path, receipts=receipts)
                failure_job = failure.get("provenance", {}).get("slurm", {}).get("SLURM_JOB_ID")
                if str(failure_job) == row["job_id"] or not failure_job and not ambiguous:
                    phases.append(("failure", failure, path))
                    ambiguous = False
            app_state, app_exit, evidence = application_state(phases, provenance, ambiguous=ambiguous)
            scheduler = records.get(row["job_id"])
            state, exit_code = (scheduler["state"], scheduler["exit_code"]) if scheduler else (app_state, app_exit)
            if scheduler and app_state == "FAILED" and state == "COMPLETED":
                raise ValueError("application failure contradicts completed scheduler evidence")
            fields = dict(scheduler["fields"]) if scheduler else {}
            resources = {key: fields.pop(key) for key in tuple(fields) if key in QUERY_KEYS - {"name", "script_sha256", "input_size", "parameters", "memory_scope"}}
            logs = {kind: run / "logs" / f"{row['stage']}-{row['method']}-{row['job_id']}.{suffix}"
                    for kind, suffix in (("stdout", "out"), ("stderr", "err"))}
            config_path = plain_path(root, run / "config.yaml")
            source_dir = plain_path(root, run / "source")
            reporter.start_attempt(attempt, pipeline_dir=run, stage=row["stage"], method=row["method"],
                                   seed=int(row["seed"]), job_id=row["job_id"], resources=resources,
                                   config_path=config_path if config_path.exists() else None,
                                   source_dir=source_dir if source_dir.exists() else None,
                                   scheduler_logs=logs, imported=True, state="UNKNOWN",
                                   metadata={"historical_import": True, "source_run": str(run.relative_to(root)),
                                             "source_identity": identity,
                                             "application_state": app_state, "state_evidence": "sacct" if scheduler else evidence,
                                             "scheduler_state": scheduler["state"] if scheduler else "unobserved",
                                             "slurm_exit_code": scheduler["slurm_exit_code"] if scheduler else None,
                                             "dependency": row["dependency"], "artifact_retry_ambiguous": ambiguous})
            if not ambiguous:
                for command, value, path in phases:
                    phase_prov_path = plain_path(root, path.parent / "provenance.json")
                    phase_prov = read_json(root, phase_prov_path, receipts=receipts) if phase_prov_path.exists() else None
                    reporter.report_phase(command, compact_result(command, value), path.parent,
                                          provenance_record=phase_prov, attempt_dir=attempt)
                    reporter.emit_event("imported_summary", import_metrics(command, value), attempt_dir=attempt)
            if ambiguous or not phases:
                reporter.emit_event("imported_summary", {}, attempt_dir=attempt)
            if not scheduler and provenance and not ambiguous:
                elapsed = provenance.get("elapsed_seconds")
                if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0 and row["stage"] != "report":
                    fields["runtime_seconds"] = elapsed
            summary = reporter.finish_attempt(attempt, exit_code, state=state, summary_fields=fields)
            attempts.append({"path": str(attempt.relative_to(root)), "id": summary["id"],
                             "job_id": row["job_id"], "source_run": str(run.relative_to(root)), "stage": row["stage"]})
    for review in reviews:
        identity = "ssmo-review-" + review.name[:70] + "-" + hashlib.sha256(str(review.relative_to(root)).encode()).hexdigest()[:12]
        attempt = target / identity
        selected = []
        for group in ("", "replication-review", "diagnostics", "cost-review"):
            summary_path = plain_path(root, review / group / "summary.json")
            if summary_path.exists():
                selected.append(("imported-" + (group or "review"), read_json(root, summary_path, receipts=receipts), summary_path))
        provenance_path = plain_path(root, review / "provenance.json")
        provenance = read_json(root, provenance_path, receipts=receipts) if provenance_path.exists() else None
        is_review = bool(selected)
        if not selected:
            for command, filename in (("audit", "audit.json"), ("generate", "parents.json"),
                    ("train", "training_summary.json"), ("evaluate", "evaluation.json"),
                    ("gpu-audit", "gpu_audit.json"), ("representation", "representation.json"),
                    ("numerical", "numeric_teacher_audit.json"), ("inverse", "inverse.json"),
                    ("compaction", "log-compaction.json"), ("report", "report.json"), ("failure", "failure.json")):
                path = plain_path(root, review / filename)
                if path.exists():
                    value = read_json(root, path, receipts=receipts)
                    if command == "generate":
                        value = {"status": "generated", "parents": len(value.get("parents", [])),
                                 "config_hash": value.get("config_hash")}
                    selected.append((command, value, path))
        state, exit_code, evidence = ("COMPLETED", None, "static review files imported") if is_review else application_state(selected, provenance)
        failure = next((value for command, value, path in selected if command == "failure"), {})
        identity_provenance = provenance or failure.get("provenance", {})
        stage = "review" if is_review else identity_provenance.get("command", failure.get("command", "artifact-import"))
        job_id = None if is_review else identity_provenance.get("slurm", {}).get("SLURM_JOB_ID")
        if job_id is not None and not JOB_ID.fullmatch(str(job_id)):
            raise ValueError("artifact provenance has an invalid actual job ID")
        training = next((value for command, value, path in selected if command == "train"), {})
        reporter.start_attempt(attempt, stage=stage, method=training.get("method", "not_applicable"),
                               seed=training.get("seed"), job_id=job_id,
                               imported=True, state="UNKNOWN",
                               metadata={"historical_import": True, "source_artifact_dir": str(review.relative_to(root)),
                                         "state_evidence": evidence, "source_identity": str(review.relative_to(root)),
                                         "scheduler_state": "unobserved" if job_id else "not a Slurm execution"})
        for command, value, path in selected:
            reporter.report_phase(command, compact_result(command, value), path.parent,
                                  provenance_record=provenance, attempt_dir=attempt)
        for relative in ("assessment.md", "README.md", "SHA256SUMS", "source-checksums.sha256", "source-index.tsv"):
            path = plain_path(root, review / relative)
            if path.exists():
                logs = read_json(root, attempt / "logs.json", limit=256 << 10)
                if not any(os.path.normpath(attempt / entry["path"]) == str(path) for entry in logs["logs"]):
                    reporter.register_log(attempt, "review." + relative.replace(".", "-"), path,
                                          label=relative, group="Review receipts")
        reporter.emit_event("imported_summary", {"review_summary_count": len(selected)}, attempt_dir=attempt)
        fields = {}
        elapsed = (provenance or {}).get("elapsed_seconds")
        if not is_review and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0:
            fields["runtime_seconds"] = elapsed
        summary = reporter.finish_attempt(attempt, exit_code, state=state, summary_fields=fields)
        attempts.append({"path": str(attempt.relative_to(root)), "id": summary["id"],
                         "source_artifact_dir": str(review.relative_to(root)), "stage": stage})
    index = {"schema": "ssmo.tower-export/v1", "attempts": attempts,
             "scope": "explicit historical sources; no scientific execution or modification",
             "imported_at_epoch": time.time()}
    write_json(root, target / "index.json", index)
    write_json(root, target / "sources.json", {"schema": "ssmo.tower-source-receipts/v1", "inputs": receipts,
               "scope": "exact JSON/identity bytes read; raw logs/checkpoints were not read, copied or decompressed"})
    return index


def selected_attempts(root, attempt_dirs, export_dirs):
    selected = [plain_path(root, path, exists=True) for path in attempt_dirs]
    for value in export_dirs:
        directory = plain_path(root, value, exists=True)
        index = read_json(root, directory / "index.json", limit=MAX_PLANNING)
        if index.get("schema") != "ssmo.tower-export/v1":
            raise ValueError("not an explicit SSMO Tower export index")
        for item in index.get("attempts", []):
            path = plain_path(root, item["path"], exists=True)
            if not path.is_relative_to(directory):
                raise ValueError("export index attempt lies outside its export")
            selected.append(path)
    if not selected or len(selected) > MAX_ATTEMPTS:
        raise ValueError("select 1..256 explicit attempts")
    return selected


def planning(root, attempt_dirs, export_dirs, output, *, reference=None, replace=False):
    attempts = selected_attempts(root, attempt_dirs, export_dirs)
    rows, seen, jobs, by_path = [], {}, {}, {}
    for path in attempts:
        row = read_json(root, path / "summary.json", limit=256 << 10)
        run = read_json(root, path / "run.json", limit=64 << 10)
        if row.get("schema") != "tower.summary/v1" or run.get("schema") != "tower.run/v1" or row.get("id") != run.get("run_id") or row.get("state") != run.get("state"):
            raise ValueError("attempt inventory and summary disagree")
        ident, job = row["id"], row.get("job_id")
        if not IDENT.fullmatch(ident) or row["state"] not in STATES:
            raise ValueError("invalid attempt identity/state")
        comparable = {key: value for key, value in row.items() if key != "metadata"}
        if ident in seen or job and job in jobs:
            prior = seen.get(ident) or jobs[job]
            if comparable != prior:
                raise ValueError("conflicting snapshots for the same attempt/job; explicitly select one")
            by_path[path] = row
            continue
        seen[ident] = comparable
        if job:
            jobs[job] = comparable
        rows.append(row)
        by_path[path] = row
    bundle = {"version": 1, "kind": "tower.planning", "history": rows,
              "description": "Explicit SSMO attempt inventory; missing scheduler resources stay unknown; no scaling or forecast evidence invented."}
    if reference:
        path = plain_path(root, reference, exists=True)
        if path not in by_path:
            raise ValueError("reference must be an explicitly selected attempt")
        bundle["query"] = {key: value for key, value in by_path[path].items() if key in QUERY_KEYS and value is not None}
    encoded(bundle, MAX_PLANNING)
    destination = plain_path(root, output)
    if destination.exists() and replace:
        old = read_json(root, destination, limit=MAX_PLANNING)
        if old.get("kind") != "tower.planning" or old.get("version") != 1:
            raise ValueError("refusing to replace an unrelated file")
    write_json(root, destination, bundle, replace=replace)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import", help="read explicitly selected existing sources into fresh Tower attempts")
    imp.add_argument("--run", action="append", default=[])
    imp.add_argument("--artifact-dir", action="append", default=[])
    imp.add_argument("--output-dir", default="runs/ssmo-tower-export-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid()))
    imp.add_argument("--accounting-file")
    imp.add_argument("--accounting", action="store_true", help="capture sacct once (30 seconds maximum), never submit or poll")
    plan = commands.add_parser("planning", help="aggregate explicit attempts or explicit export indexes")
    plan.add_argument("--attempt-dir", action="append", default=[])
    plan.add_argument("--export-dir", action="append", default=[])
    plan.add_argument("--reference")
    plan.add_argument("--output", default="reports/planning.json")
    plan.add_argument("--replace", action="store_true")
    listing = commands.add_parser("list", help="print exact attempt paths from a selected export index")
    listing.add_argument("--export-dir", required=True)
    for command in ("launch", "validate"):
        sub = commands.add_parser(command, help="invoke your installed Tower; never install or modify it")
        sub.add_argument("--attempt-dir", required=True)
        sub.add_argument("--planning-file")
    args = parser.parse_args(argv)
    root = project_root()
    try:
        if args.command == "import":
            result = import_runs(root, args.run, args.artifact_dir, args.output_dir,
                                 accounting_file=args.accounting_file, accounting=args.accounting)
            print(f"SSMO: imported {len(result['attempts'])} attempts into {plain_path(root, args.output_dir)}")
            for row in result["attempts"]:
                print(row["path"])
        elif args.command == "planning":
            result = planning(root, args.attempt_dir, args.export_dir, args.output,
                              reference=args.reference, replace=args.replace)
            print(f"SSMO: planning saved {len(result['history'])} distinct attempts: {plain_path(root, args.output)}")
        elif args.command == "list":
            for path in selected_attempts(root, [], [args.export_dir]):
                print(path.relative_to(root))
        else:
            attempt = plain_path(root, args.attempt_dir, exists=True)
            if not (attempt / "run.json").is_file():
                raise ValueError("choose a concrete Tower attempt directory")
            executable = shutil.which("tower")
            if not executable:
                raise ValueError("Tower executable is unavailable; activate your existing Tower installation")
            if args.command == "validate":
                command = [executable, "run", "validate", str(plain_path(root, ".tower/contracts/outputs.v1.json", exists=True)), str(attempt)]
            else:
                command = [executable, "--profile", "carc", "--config", str(plain_path(root, ".tower/config.json", exists=True)),
                           "--workdir", str(attempt), "--tab", "research", "--research-view", "experiment"]
                if args.planning_file:
                    command += ["--planning-file", str(plain_path(root, args.planning_file, exists=True))]
            return subprocess.call(command, cwd=root)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"SSMO Tower: {error}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
