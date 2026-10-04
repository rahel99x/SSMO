#!/usr/bin/env python3
"""Review completed frozen pilots using standard-library artifact reads only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pwd
import re
import sys
from datetime import datetime, timezone


CARC_ROOT = Path("/home1/aadaniel/projects/SSMO")
METHODS = ("measure", "state_only")
CLASSICAL = "classical_front_regression"
ERRORS = ("absolute_error_max", "nonlinear_gradient_error_max", "support_error_max", "weight_error_max")


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def project_root(value=None):
    configured = os.environ.get("SSMO_PROJECT_ROOT")
    legacy = os.environ.get("SSMO_ROOT")
    if configured and legacy and Path(configured).resolve() != Path(legacy).resolve():
        raise ValueError("SSMO_PROJECT_ROOT and SSMO_ROOT disagree")
    supplied = Path(value or configured or legacy or Path(__file__).resolve().parents[1])
    root = supplied.resolve(strict=True)
    if not supplied.is_absolute() or supplied != root:
        raise ValueError("project root must be an absolute canonical path without symlink aliases")
    if not root.is_dir():
        raise ValueError("project root must be an existing directory")
    if (pwd.getpwuid(os.getuid()).pw_name == "aadaniel" or os.environ.get("SLURM_JOB_ID")) and root != CARC_ROOT:
        raise ValueError(f"CARC storage requires {CARC_ROOT}")
    return root


def confined(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes SSMO project root: {path}")
    return resolved


class Inputs:
    def __init__(self, root):
        self.root = root
        self.receipts = {}

    def read(self, value):
        path = confined(self.root, value)
        if not path.is_file():
            raise ValueError(f"required artifact missing: {path}")
        data = path.read_bytes()
        name = str(path.relative_to(self.root))
        self.receipts[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return data

    def json(self, value):
        result = json.loads(self.read(value))
        if not isinstance(result, dict):
            raise ValueError(f"expected a JSON object: {value}")
        # Reject non-finite JSON extensions before composing a result.
        content_hash(result)
        return result


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return value


def cohorts(evaluation, method):
    rows = []
    seen = set()
    for row in evaluation["parent_aggregates"]:
        if row["method"] != method:
            continue
        key = (row["split"], row["family"])
        if key in seen:
            raise ValueError(f"duplicate cohort for {method}: {key}")
        seen.add(key)
        total = integer(row["physical_parents"], "physical parent count")
        passed = integer(row["accuracy_gate_parents_passed"], "passed parent count")
        failed = integer(row["accuracy_gate_parents_failed"], "failed parent count")
        if total < 1 or passed + failed != total:
            raise ValueError(f"inconsistent parent gate counts for {method}: {key}")
        errors = {}
        for metric in ERRORS:
            detail = row["metrics"].get(metric)
            errors[metric] = number(detail["worst"], metric) if detail is not None else None
        # Invalid charts may have no finite weak-error statistic; preserve this
        # explicitly rather than replacing absent errors with successful zeros.
        invalid = integer(row["invalid_chart_rows"], "invalid chart row count")
        if errors["absolute_error_max"] is None and not invalid:
            raise ValueError(f"missing weak-error evidence for {method}: {key}")
        rows.append({"split": key[0], "family": key[1], "physical_parents": total,
                     "passed": passed, "failed": failed, "invalid_chart_rows": invalid, "worst_errors": errors})
    if not rows:
        raise ValueError(f"missing evaluation method: {method}")
    rows.sort(key=lambda row: (row["split"], row["family"]))
    return rows


def totals(rows):
    return {key: sum(row[key] for row in rows) for key in ("physical_parents", "passed", "failed", "invalid_chart_rows")}


def gpu_stratum(hardware):
    if hardware.get("cuda_executed") is not True or not {"forward", "jvp", "mixed_derivative_backward"}.issubset(hardware.get("kernel_checks", [])):
        raise ValueError("GPU audit did not execute all required CUDA kernels")
    return {key: hardware[key] for key in ("model", "total_memory_bytes", "capability", "torch_version", "cuda_runtime")}


def read_run(inputs, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) or run_id in {".", ".."}:
        raise ValueError(f"invalid run ID: {run_id}")
    run = confined(inputs.root, Path("runs") / run_id)
    if not run.is_dir():
        raise ValueError(f"run directory missing: {run_id}")
    config_bytes = inputs.read(run / "config.yaml")
    if not config_bytes.strip():
        raise ValueError(f"empty frozen config: {run_id}")
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    manifest = inputs.json(run / "artifacts/data/parents.json")
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("parents"), list):
        raise ValueError(f"unsupported parent manifest: {run_id}")
    manifest_hash = content_hash(manifest)
    submission = {}
    for line in inputs.read(run / "submission-manifest.txt").decode().splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in submission:
            raise ValueError(f"malformed submission manifest: {run_id}")
        submission[key] = value
    if submission.get("run_id") != run_id or submission.get("pipeline") != "pilot":
        raise ValueError(f"run is not a recorded pilot: {run_id}")
    if confined(inputs.root, submission["ssmo_project_root"]) != inputs.root:
        raise ValueError(f"submission project root mismatch: {run_id}")
    rows = list(csv.DictReader(io.StringIO(inputs.read(run / "jobs.tsv").decode()), delimiter="\t"))
    train_rows = [row for row in rows if row["stage"] == "train"]
    if len(train_rows) != 2 or {row["method"] for row in train_rows} != set(METHODS):
        raise ValueError(f"pilot requires one training job per method: {run_id}")
    seed_values = {row["seed"] for row in train_rows}
    if len(seed_values) != 1 or not next(iter(seed_values)).isdigit():
        raise ValueError(f"inconsistent training seeds: {run_id}")
    seed = int(next(iter(seed_values)))
    if submission.get("seed") != str(seed):
        raise ValueError(f"submission seed mismatch: {run_id}")
    for method in METHODS:
        evaluation_rows = [row for row in rows if row["stage"] == "evaluate" and row["method"] == method]
        if len(evaluation_rows) != 1 or evaluation_rows[0]["seed"] != str(seed):
            raise ValueError(f"pilot requires the matching evaluation job: {run_id}/{method}")

    source = confined(inputs.root, run / "source")
    science_files = sorted((source / "singular_sensitivity").rglob("*.py"))
    if not science_files:
        raise ValueError(f"frozen scientific source missing: {run_id}")
    source_hashes = {str(path.relative_to(source)): hashlib.sha256(inputs.read(path)).hexdigest() for path in science_files}
    pins = sorted((source / "requirements").rglob("*.txt")) + [source / "pyproject.toml"]
    if len(pins) < 2:
        raise ValueError(f"frozen dependency requirements missing: {run_id}")
    dependency_hashes = {str(path.relative_to(source)): hashlib.sha256(inputs.read(path)).hexdigest() for path in pins}
    freeze_hash = hashlib.sha256(inputs.read(run / "artifacts/dependency-freeze.txt")).hexdigest()
    gpu_audit = inputs.json(run / "artifacts/gpu-smoke/gpu_audit.json")
    if gpu_audit.get("status") != "passed":
        raise ValueError(f"GPU audit failed: {run_id}")
    hardware_stratum = gpu_stratum(gpu_audit)

    methods, evaluations = {}, {}
    for method in METHODS:
        directory = run / f"artifacts/train-{method}-seed{seed}"
        training = inputs.json(directory / "training_summary.json")
        evaluation = inputs.json(run / f"artifacts/evaluate-{method}-seed{seed}/evaluation.json")
        for path in (directory / "hardware.json", run / f"artifacts/evaluate-{method}-seed{seed}/hardware.json"):
            if gpu_stratum(inputs.json(path)) != hardware_stratum:
                raise ValueError(f"GPU/software stratum changed within pilot: {run_id}/{method}")
        if training["status"] != "complete" or training["method"] != method or training["seed"] != seed:
            raise ValueError(f"training incomplete or method/seed mismatch: {run_id}/{method}")
        recorded_hardware = training["hardware"]
        if (recorded_hardware.get("gpu_name") != hardware_stratum["model"]
                or recorded_hardware.get("gpu_total_bytes") != hardware_stratum["total_memory_bytes"]
                or recorded_hardware.get("torch_version") != hardware_stratum["torch_version"]):
            raise ValueError(f"training hardware disagrees with kernel audit: {run_id}/{method}")
        if evaluation["status"] != "completed_scoped_pilot":
            raise ValueError(f"evaluation incomplete: {run_id}/{method}")
        if training["dataset_hash"] != manifest_hash:
            raise ValueError(f"training dataset hash mismatch: {run_id}/{method}")
        checkpoint = evaluation["checkpoints"].get(method)
        if set(evaluation["checkpoints"]) != {method} or checkpoint is None:
            raise ValueError(f"evaluation checkpoint method mismatch: {run_id}/{method}")
        best = confined(inputs.root, training["best_checkpoint"])
        expected = confined(inputs.root, directory / "best.pt")
        if best != expected or confined(inputs.root, checkpoint["path"]) != expected or not best.is_file():
            raise ValueError(f"selected checkpoint path missing or mismatch: {run_id}/{method}")
        if (checkpoint["dataset_hash"] != manifest_hash or checkpoint["step"] != training["best_step"]
                or checkpoint["best_step"] != training["best_step"]):
            raise ValueError(f"selected checkpoint evidence mismatch: {run_id}/{method}")
        provenance_path = directory / "provenance.json"
        if provenance_path.exists() or provenance_path.is_symlink():
            provenance = inputs.json(provenance_path)
            if provenance["manifest_sha256"] != manifest_hash or provenance["config_sha256"] != training["config_hash"]:
                raise ValueError(f"training provenance hash mismatch: {run_id}/{method}")
        evaluation_provenance = run / f"artifacts/evaluate-{method}-seed{seed}/provenance.json"
        if evaluation_provenance.exists() or evaluation_provenance.is_symlink():
            provenance = inputs.json(evaluation_provenance)
            if provenance["manifest_sha256"] != manifest_hash:
                raise ValueError(f"evaluation provenance manifest mismatch: {run_id}/{method}")
        method_cohorts = cohorts(evaluation, method + "_chart")
        best_step = integer(training["best_step"], "best step")
        steps = integer(training["steps_completed"], "completed steps")
        if best_step > steps:
            raise ValueError(f"best step exceeds completed training: {run_id}/{method}")
        peak = training.get("peak_reserved_bytes")
        consistency = [row for row in evaluation["direction_and_chart_consistency"] if row["method"] == method + "_chart"]
        endpoints = [{key: row[key] for key in ("parent_id", "directions", "queries", "endpoint", "measurement")}
                     for row in evaluation["timings"] if row["method"] == method + "_chart" and "measurement" in row]
        methods[method] = {
            "training": {"stopped_reason": training["stopped_reason"], "steps_completed": steps, "best_step": best_step,
                         "best_validation_score": number(training["best_validation_score"], "validation score"),
                         "elapsed_seconds": number(training["elapsed_seconds"], "training elapsed seconds"),
                         "peak_reserved_mib": number(peak, "peak reserved bytes") / 2**20 if peak is not None else None,
                         "config_hash": training["config_hash"], "hardware": training["hardware"],
                         "query_protocol": training["query_protocol"], "direction_convention": training["direction_convention"]},
            "selected_checkpoint": checkpoint, "cohorts": method_cohorts, "total": totals(method_cohorts),
            "fixed_audit_coverage": evaluation["fixed_audit_coverage"],
            "unresolved_records": evaluation["unresolved_records"], "direction_and_chart_consistency": consistency,
            "measured_endpoints": endpoints,
            "failures": [row for row in evaluation["accuracy_gate"]["parent_method_failures"] if row["method"] == method + "_chart"],
        }
        evaluations[method] = evaluation
    protocols = [{key: evaluation[key] for key in ("domain", "selected_parent_ids", "held_out_query_bank", "fixed_audit_coverage")}
                 | {"absolute_tolerance": evaluation["accuracy_gate"]["absolute_tolerance"]} for evaluation in evaluations.values()]
    if protocols[0] != protocols[1]:
        raise ValueError(f"evaluation protocols differ between methods: {run_id}")
    for key in ("query_protocol", "direction_convention"):
        if methods["measure"]["training"][key] != methods["state_only"]["training"][key]:
            raise ValueError(f"training {key} differs between methods: {run_id}")
    if methods["measure"]["cohorts"] and [(row["split"], row["family"], row["physical_parents"]) for row in methods["measure"]["cohorts"]] != [
            (row["split"], row["family"], row["physical_parents"]) for row in methods["state_only"]["cohorts"]]:
        raise ValueError(f"learned methods use different parent cohorts: {run_id}")
    classical_raw = [[row for row in evaluation["parent_aggregates"] if row["method"] == CLASSICAL]
                     for evaluation in evaluations.values()]
    if classical_raw[0] != classical_raw[1]:
        raise ValueError(f"duplicated classical control differs between evaluations: {run_id}")
    classical = cohorts(evaluations["measure"], CLASSICAL)
    learned_counts = [(row["split"], row["family"], row["physical_parents"]) for row in methods["measure"]["cohorts"]]
    if [(row["split"], row["family"], row["physical_parents"]) for row in classical] != learned_counts:
        raise ValueError(f"classical and learned controls use different parent cohorts: {run_id}")
    source_revision = run / "source-revision.txt"
    revision = inputs.read(source_revision).decode().strip() if source_revision.exists() else None
    return {"run_id": run_id, "seed": seed, "profile": submission.get("profile"), "source_revision": revision,
            "frozen_config_sha256": config_sha, "parent_manifest_sha256": manifest_hash, "evaluation_protocol": protocols[0],
            "scientific_source_sha256": content_hash(source_hashes), "dependency_pins_sha256": content_hash(dependency_hashes),
            "dependency_freeze_sha256": freeze_hash, "gpu_stratum": hardware_stratum,
            "methods": methods, "classical": {"cohorts": classical, "total": totals(classical), "counted_once": True}}


def format_report(summary):
    lines = ["SSMO frozen pilot review", "Seeds share physical parents: report each seed separately; no pooled sample size.",
             "Gate = recorded weak-query tolerance and no invalid chart; completion does not establish research advantage."]
    for run in summary["runs"]:
        lines += ["", f"{run['run_id']} seed={run['seed']} profile={run['profile']} tolerance={run['evaluation_protocol']['absolute_tolerance']} "
                  f"gpu={run['gpu_stratum']['model']} CUDA kernels=passed"]
        for method in METHODS:
            result = run["methods"][method]
            training, total = result["training"], result["total"]
            lines.append(f"  {method}: {total['passed']}/{total['physical_parents']} pass; invalid_rows={total['invalid_chart_rows']}; "
                         f"stop={training['stopped_reason']} steps={training['steps_completed']} best_step={training['best_step']} "
                         f"validation={training['best_validation_score']:.6g} train_s={training['elapsed_seconds']:.3f} reserved_MiB={training['peak_reserved_mib']}")
            for row in result["cohorts"]:
                errors = ["unavailable" if row["worst_errors"][key] is None else f"{row['worst_errors'][key]:.8g}" for key in ERRORS]
                lines.append(f"    {row['split']}/{row['family']}: {row['passed']}/{row['physical_parents']} "
                             f"weak={errors[0]} nonlinear={errors[1]} support={errors[2]} weight={errors[3]} invalid_rows={row['invalid_chart_rows']}")
            statuses = dict(Counter((record.get("status", "unspecified") + ":" + record.get("family", "unspecified")) for record in result["unresolved_records"]))
            lines.append(f"    fixed_audit={json.dumps(result['fixed_audit_coverage'], sort_keys=True)} unresolved={json.dumps(statuses, sort_keys=True)}")
            diagnostics = result["direction_and_chart_consistency"]
            maxima = {key: max((row[key] for row in diagnostics if isinstance(row.get(key), (int, float))), default=None)
                      for key in ("zero_direction_max_error", "additivity_max_error", "scaling_max_error", "chart_finite_variation_max_error")}
            lines.append(f"    consistency={json.dumps(maxima, sort_keys=True)}")
        classical = run["classical"]["total"]
        lines.append(f"  classical (counted once): {classical['passed']}/{classical['physical_parents']} pass; invalid_rows={classical['invalid_chart_rows']}")
    lines += ["", "Preserve this review with raw run artifacts. Keep the frozen preset and tolerance; review failures before expanding GPU work."]
    return "\n".join(lines) + "\n"


def summarize(root, run_ids, output_dir):
    root = project_root(root)
    output = confined(root, output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError(f"review output already exists; choose a new directory: {output}")
    if not output.parent.is_dir():
        raise ValueError("review output parent directory must already exist")
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("provide one or more distinct pilot run IDs")
    inputs = Inputs(root)
    runs = [read_run(inputs, run_id) for run_id in run_ids]
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("replication review requires distinct seeds")
    for run in runs:
        if output.is_relative_to(confined(root, Path("runs") / run["run_id"])):
            raise ValueError("review output must be separate from source runs")
    for key, label in (("frozen_config_sha256", "frozen scientific config"), ("parent_manifest_sha256", "parent manifest"),
                       ("evaluation_protocol", "evaluation protocol"), ("scientific_source_sha256", "frozen scientific source"),
                       ("dependency_pins_sha256", "dependency pins"), ("dependency_freeze_sha256", "installed dependencies"),
                       ("gpu_stratum", "actual GPU/software stratum")):
        if any(run[key] != runs[0][key] for run in runs[1:]):
            raise ValueError(f"{label} differs across pilots; do not combine this replication review")
    for key in ("query_protocol", "direction_convention"):
        if any(run["methods"]["measure"]["training"][key] != runs[0]["methods"]["measure"]["training"][key] for run in runs[1:]):
            raise ValueError(f"training {key} differs across pilots")
    summary = {"schema_version": 1, "runs": runs, "independent_unit": "physical parent within each seed; parents are reused across seeds",
               "pooled_parent_count": None, "research_advantage_established": False}
    report = format_report(summary)
    json_bytes = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    text_bytes = report.encode()
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "project_root": str(root), "read_only_source_runs": run_ids,
                  "inputs": inputs.receipts, "outputs": {"summary.json": hashlib.sha256(json_bytes).hexdigest(),
                                                         "summary.txt": hashlib.sha256(text_bytes).hexdigest()},
                  "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # All inputs are checked before creating a fresh output; never overwrite or
    # clean an existing directory, including a preserved unsuccessful review.
    output.mkdir()
    for name, data in (("summary.json", json_bytes), ("summary.txt", text_bytes),
                       ("provenance.json", (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode())):
        with (output / name).open("xb") as stream:
            stream.write(data)
    return summary, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--run", action="append", required=True, dest="run_ids", help="recorded pilot run ID; repeat for each seed")
    parser.add_argument("--output-dir", required=True, help="new project-contained directory with an existing parent")
    args = parser.parse_args(argv)
    try:
        _, report = summarize(args.project_root, args.run_ids, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"SSMO: pilot review refused: {error}", file=sys.stderr)
        return 2
    print(report, end="")
    print(f"Saved review: {confined(project_root(args.project_root), args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
