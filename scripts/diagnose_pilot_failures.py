#!/usr/bin/env python3
"""Read-only, standard-library attribution of already exposed pilot failures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

from pilot_error_calculus import attribute_row
from summarize_pilots import Inputs, METHODS, confined, content_hash, integer, number, project_root


DEFAULT_REVIEW = "runs/ssmo-pilot-review-20261004T000302Z-2641811"
FORMULA_FILES = ("__init__.py", "reference.py", "queries.py", "baselines.py", "representation.py", "validation.py")
HELPER_FILES = ("diagnose_pilot_failures.py", "pilot_error_calculus.py", "summarize_pilots.py")
BLOCK_BYTES = 1024 * 1024


def canonical_directory(root, value):
    requested = Path(value)
    if not requested.is_absolute():
        requested = root / requested
    path = confined(root, requested)
    if path != requested.absolute() or not path.is_dir():
        raise ValueError(f"artifact directory must have an existing canonical path: {requested}")
    return path


def regular(root, value, within=None):
    """Reject aliases even when their targets remain somewhere inside SSMO."""
    requested = Path(value)
    if not requested.is_absolute():
        requested = root / requested
    path = confined(root, requested)
    if path != requested.absolute() or (within is not None and not path.is_relative_to(within)):
        raise ValueError(f"artifact must have a canonical path within its source directory: {requested}")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"artifact must be a regular file with one hard link: {path}")
    return path


def signature(path):
    info = path.stat(follow_symlinks=False)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def digest_file(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def read_recorded(inputs, recorded, value):
    path = regular(inputs.root, value)
    name = str(path.relative_to(inputs.root))
    if name not in recorded:
        raise ValueError(f"review provenance is missing required input: {name}")
    data = inputs.read(path)
    if inputs.receipts[name] != recorded[name]:
        raise ValueError(f"review provenance input checksum mismatch: {name}")
    return data


def recorded_json(inputs, recorded, value):
    data = json.loads(read_recorded(inputs, recorded, value))
    if not isinstance(data, dict):
        raise ValueError(f"expected an artifact JSON object: {value}")
    content_hash(data)
    return data


def protocol(evaluation):
    return {key: evaluation[key] for key in ("domain", "selected_parent_ids", "held_out_query_bank", "fixed_audit_coverage")} | {
        "absolute_tolerance": evaluation["accuracy_gate"]["absolute_tolerance"]}


def load_sources(inputs, review_dir, output):
    root = inputs.root
    review_dir = canonical_directory(root, review_dir)
    if output.is_relative_to(review_dir):
        raise ValueError("diagnostic output must be separate from the completed review")
    receipt = inputs.json(regular(root, review_dir / "provenance.json", review_dir))
    if receipt.get("project_root") != str(root) or not isinstance(receipt.get("inputs"), dict):
        raise ValueError("review provenance project root or input receipts are incompatible")
    for name in ("summary.json", "summary.txt"):
        data = inputs.read(regular(root, review_dir / name, review_dir))
        if hashlib.sha256(data).hexdigest() != receipt.get("outputs", {}).get(name):
            raise ValueError(f"review output checksum mismatch: {name}")
    summary = inputs.json(review_dir / "summary.json")
    if summary.get("schema_version") != 1 or not isinstance(summary.get("runs"), list) or not summary["runs"]:
        raise ValueError("unsupported or empty completed pilot review")
    if summary.get("pooled_parent_count") is not None:
        raise ValueError("pilot review must retain separate initialization seeds")
    run_ids = [run["run_id"] for run in summary["runs"]]
    if len(set(run_ids)) != len(run_ids) or receipt.get("read_only_source_runs") != run_ids:
        raise ValueError("review provenance run list is inconsistent")
    if len({run["seed"] for run in summary["runs"]}) != len(run_ids):
        raise ValueError("review requires distinct initialization seeds")
    contexts = []
    for run in summary["runs"]:
        run_id, seed = run["run_id"], integer(run["seed"], "run seed")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) or run_id in {".", ".."}:
            raise ValueError(f"invalid reviewed run ID: {run_id}")
        directory = canonical_directory(root, root / "runs" / run_id)
        if output.is_relative_to(directory):
            raise ValueError("diagnostic output must be separate from all source runs")
        config = read_recorded(inputs, receipt["inputs"], directory / "config.yaml")
        if hashlib.sha256(config).hexdigest() != run["frozen_config_sha256"]:
            raise ValueError(f"frozen config disagrees with reviewed run: {run_id}")
        manifest = recorded_json(inputs, receipt["inputs"], directory / "artifacts/data/parents.json")
        if manifest.get("schema_version") != 1 or content_hash(manifest) != run["parent_manifest_sha256"]:
            raise ValueError(f"parent manifest disagrees with reviewed run: {run_id}")
        if manifest["domain"] != run["evaluation_protocol"]["domain"]:
            raise ValueError(f"manifest domain differs from frozen evaluation protocol: {run_id}")
        science = directory / "source/singular_sensitivity"
        source_files = sorted(science.rglob("*.py"))
        if not source_files:
            raise ValueError(f"frozen scientific source missing: {run_id}")
        hashes = {str(path.relative_to(directory / "source")): hashlib.sha256(
            read_recorded(inputs, receipt["inputs"], path)).hexdigest() for path in source_files}
        if content_hash(hashes) != run["scientific_source_sha256"]:
            raise ValueError(f"scientific source disagrees with reviewed run: {run_id}")
        for name in FORMULA_FILES:
            frozen = read_recorded(inputs, receipt["inputs"], science / name)
            current = inputs.read(regular(root, root / "singular_sensitivity" / name))
            if frozen != current:
                raise ValueError(f"current formula source differs from frozen evidence: {run_id}/{name}")
        parents = {record["parent_id"]: record for record in manifest["parents"]}
        if len(parents) != len(manifest["parents"]):
            raise ValueError(f"duplicate physical-parent IDs: {run_id}")
        failures = run["methods"]["measure"]["failures"]
        targets = {record["parent_id"]: record for record in failures}
        if len(targets) != len(failures):
            raise ValueError(f"duplicate reviewed measure failures: {run_id}")
        tolerance = number(run["evaluation_protocol"]["absolute_tolerance"], "frozen weak tolerance")
        if tolerance <= 0:
            raise ValueError("frozen weak tolerance must be positive")
        for parent_id, failure in targets.items():
            if failure["method"] != "measure_chart" or failure["invalid_chart_rows"] != 0:
                raise ValueError(f"selected failure has an invalid or incompatible chart: {run_id}/{parent_id}")
            if number(failure["worst_absolute_error"], "reviewed worst error") <= tolerance:
                raise ValueError(f"reviewed failure does not exceed frozen weak gate: {run_id}/{parent_id}")
            if parent_id not in parents or parent_id not in run["evaluation_protocol"]["selected_parent_ids"]:
                raise ValueError(f"failure is absent from frozen evaluated parents: {run_id}/{parent_id}")
            if parents[parent_id]["family"] not in {"shock", "constant"} or parents[parent_id]["split"] != failure["split"]:
                raise ValueError(f"failure requires a regular one-front parent: {run_id}/{parent_id}")
        for method in METHODS:
            evaluation_dir = directory / f"artifacts/evaluate-{method}-seed{seed}"
            evaluation = recorded_json(inputs, receipt["inputs"], evaluation_dir / "evaluation.json")
            if evaluation.get("status") != "completed_scoped_pilot" or protocol(evaluation) != run["evaluation_protocol"]:
                raise ValueError(f"evaluation protocol differs from completed review: {run_id}/{method}")
            recorded_failures = [record for record in evaluation["accuracy_gate"]["parent_method_failures"]
                                 if record["method"] == method + "_chart"]
            if recorded_failures != run["methods"][method]["failures"]:
                raise ValueError(f"evaluation failure list differs from completed review: {run_id}/{method}")
            original = evaluation_dir / "query_errors.jsonl"
            if Path(evaluation["artifacts"]["query_errors"]) != original:
                raise ValueError(f"raw query artifact path differs from exact evaluated stage: {run_id}/{method}")
            source = locate_raw(inputs, directory / "artifacts", original)
            contexts.append({"run": run, "method": method, "manifest": manifest, "parents": parents,
                             "targets": targets, "source": source, "evaluation": evaluation})
    return summary, contexts


def locate_raw(inputs, artifacts, original):
    artifacts = confined(inputs.root, artifacts)
    mapping_path = artifacts / "log-compaction.json"
    entry = None
    if mapping_path.exists() or mapping_path.is_symlink():
        mapping = inputs.json(regular(inputs.root, mapping_path, artifacts))
        if mapping.get("status") != "completed_lossless_log_compaction":
            raise ValueError("raw log compaction is incomplete; preserve and inspect its receipt")
        matches = [row for row in mapping["files"] if row.get("original") == str(original.relative_to(artifacts))]
        if len(matches) > 1:
            raise ValueError(f"ambiguous raw archive receipt: {original}")
        entry = matches[0] if matches else None
    if original.exists() or original.is_symlink():
        path, compressed = regular(inputs.root, original, artifacts), False
        if entry is not None and entry.get("state") != "verified_archive_original_preserved":
            raise ValueError("compacted raw log unexpectedly retains its original; inspect receipt")
    else:
        if entry is None or entry.get("state") != "compacted":
            raise ValueError(f"required raw query log missing with no verified archive: {original}")
        if entry.get("archive") != str(original.with_suffix(".jsonl.gz").relative_to(artifacts)):
            raise ValueError(f"archive receipt path is incompatible with original: {original}")
        path, compressed = regular(inputs.root, artifacts / entry["archive"], artifacts), True
    file_receipt = digest_file(path)
    if compressed and integer(entry["compressed_bytes"], "compressed bytes") != file_receipt["bytes"]:
        raise ValueError(f"compressed raw log byte count disagrees with receipt: {path}")
    inputs.receipts[str(path.relative_to(inputs.root))] = file_receipt
    return {"path": path, "compressed": compressed, "compaction": entry, "signature": signature(path), "file_receipt": file_receipt}


def validate_row(row, context):
    parent = context["parents"][row["parent_id"]]
    domain = context["manifest"]["domain"]
    queries = context["evaluation"]["held_out_query_bank"]
    if row.get("status") != "regular":
        raise ValueError(f"selected parent chart is invalid or unresolved: {row['parent_id']}/{row.get('status')}")
    content_hash(row)
    for key in ("split", "family", "parameters"):
        if row[key] != parent[key]:
            raise ValueError(f"raw parent {key} disagrees with frozen manifest: {row['parent_id']}")
    if row["domain"] != domain or row["time"] not in parent["times"] or row.get("reference_tier") != "exact_analytic_burgers":
        raise ValueError(f"raw time/domain/reference disagrees with frozen protocol: {row['parent_id']}")
    direction_index = integer(row["direction_index"], "raw direction index")
    query_index = integer(row["query_index"], "raw query index")
    if (direction_index >= len(parent["directions"]) or row["direction"] != parent["directions"][direction_index]
            or query_index >= len(queries) or row["query"] != queries[query_index]):
        raise ValueError(f"raw direction/query disagrees with frozen protocol: {row['parent_id']}")
    norm = math.sqrt(sum(value * value for value in row["direction"]))
    if not math.isclose(number(row["direction_norm"], "recorded direction norm"), norm, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"raw direction norm disagrees with physical direction: {row['parent_id']}")
    return parent, (row["time"], direction_index, query_index), norm


def stream_context(context, destination=None):
    source, method, targets = context["source"], context["method"], context["targets"]
    path = source["path"]
    if signature(path) != source["signature"]:
        raise ValueError(f"raw query log changed during diagnosis: {path}")
    digest, size, total_rows, selected_rows = hashlib.sha256(), 0, 0, 0
    seen = {parent_id: set() for parent_id in targets}
    base_charts, direction_charts = {}, {}
    results = {parent_id: {"query_rows": 0, "nonzero_query_rows": 0, "worst_weak_row": None,
                          "worst_nonlinear_row": None, "max_reconstruction_residual": 0.0,
                          "max_nonlinear_reconstruction_residual": 0.0,
                          "max_nonlinear_absolute_error_residual": 0.0} for parent_id in targets}
    opener = gzip.open if source["compressed"] else open
    with opener(path, "rb") as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            if not line.strip():
                raise ValueError(f"raw query log contains an empty JSONL record: {path}")
            total_rows += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"raw query log record is not a JSON object: {path}")
            if row.get("parent_id") not in targets or row.get("method") != method + "_chart":
                continue
            parent, key, norm = validate_row(row, context)
            parent_id = parent["parent_id"]
            if key in seen[parent_id]:
                raise ValueError(f"duplicate selected raw query row: {parent_id}/{key}")
            seen[parent_id].add(key)
            base_key = (parent_id, row["time"])
            base_chart = (row["predicted_position"], tuple(row["predicted_states"]))
            direction_key = (parent_id, row["time"], row["direction_index"])
            direction_chart = (tuple(row["predicted_diffuse"]), row["predicted_motion"], row["predicted_atom_weight"])
            if (base_key in base_charts and base_charts[base_key] != base_chart) or (
                    direction_key in direction_charts and direction_charts[direction_key] != direction_chart):
                raise ValueError(f"raw chart changed across queries or directions: {parent_id}/{key}")
            base_charts[base_key], direction_charts[direction_key] = base_chart, direction_chart
            ledger = attribute_row(row)
            result = results[parent_id]
            diagnostic = {"run_id": context["run"]["run_id"], "seed": context["run"]["seed"], "method": method,
                          "raw": row, "attribution": ledger}
            result["query_rows"] += 1
            selected_rows += 1
            result["max_reconstruction_residual"] = max(result["max_reconstruction_residual"], abs(ledger["reconstruction_residual"]))
            result["max_nonlinear_absolute_error_residual"] = max(result["max_nonlinear_absolute_error_residual"],
                                                                  abs(ledger["nonlinear"]["absolute_error_residual"]))
            result["max_nonlinear_reconstruction_residual"] = max(result["max_nonlinear_reconstruction_residual"],
                                                                  abs(ledger["nonlinear"]["reconstruction_residual"]))
            if norm > 0:
                result["nonzero_query_rows"] += 1
                if result["worst_weak_row"] is None or row["absolute_error"] > result["worst_weak_row"]["raw"]["absolute_error"]:
                    result["worst_weak_row"] = diagnostic
                if (result["worst_nonlinear_row"] is None or row["nonlinear_gradient_error"]
                        > result["worst_nonlinear_row"]["raw"]["nonlinear_gradient_error"]):
                    result["worst_nonlinear_row"] = diagnostic
            if destination is not None:
                destination.write((json.dumps(diagnostic, sort_keys=True, allow_nan=False) + "\n").encode())
    restored = {"sha256": digest.hexdigest(), "bytes": size}
    entry = source["compaction"]
    expected = {"sha256": entry["sha256_uncompressed"], "bytes": entry["original_bytes"]} if entry is not None else source["file_receipt"]
    if restored != expected:
        raise ValueError(f"raw query log uncompressed checksum/bytes mismatch: {path}")
    if signature(path) != source["signature"]:
        raise ValueError(f"raw query log changed during diagnosis: {path}")
    for parent_id, result in results.items():
        parent = context["parents"][parent_id]
        expected_rows = {(time, direction, query) for time in parent["times"] for direction in range(len(parent["directions"]))
                         for query in range(len(context["evaluation"]["held_out_query_bank"]))}
        if seen[parent_id] != expected_rows or result["worst_weak_row"] is None:
            raise ValueError(f"missing selected parent query rows: {context['run']['run_id']}/{method}/{parent_id}")
        worst = result["worst_weak_row"]["raw"]["absolute_error"]
        tolerance = context["run"]["evaluation_protocol"]["absolute_tolerance"]
        result["weak_gate_passed"] = worst <= tolerance
        failures = {row["parent_id"]: row for row in context["run"]["methods"][method]["failures"]}
        failure = failures.get(parent_id)
        if ((failure is None) != result["weak_gate_passed"] or (failure is not None and (
                failure["invalid_chart_rows"] != 0 or not math.isclose(worst, failure["worst_absolute_error"], rel_tol=1e-12, abs_tol=1e-14)))):
            raise ValueError(f"raw worst error disagrees with reviewed weak gate: {context['run']['run_id']}/{method}/{parent_id}")
    return results, {"raw_records_scanned": total_rows, "selected_query_rows": selected_rows, "uncompressed": restored,
                     "compressed": source["compressed"], "path": str(path)}


def report_text(summary):
    lines = ["SSMO existing pilot failure attribution (CPU; no new model evaluation)",
             "Signed telescoping ledger in a fixed order; component attribution is exploratory, not a causal proof.",
             "Seeds reuse physical parents. Weak gate, models, data and protocol remain frozen.",
             "Each method below uses its own worst query; failures.jsonl enables comparison of matched query rows.",
             "Full selected query rows, including zero directions, are saved in failures.jsonl."]
    for run in summary["runs"]:
        lines += ["", f"{run['run_id']} seed={run['seed']} frozen weak tolerance={run['absolute_tolerance']}"]
        for parent in run["parents"]:
            lines.append(f"  {parent['parent_id']} split={parent['split']} family={parent['family']}")
            for method in METHODS:
                result = parent["methods"][method]
                worst = result["worst_weak_row"]
                raw, ledger = worst["raw"], worst["attribution"]
                pieces = " ".join(f"{key}={value:+.8g}" for key, value in ledger["contributions"].items())
                lines.append(f"    {method}: rows={result['query_rows']} weak_gate={'pass' if result['weak_gate_passed'] else 'fail'} "
                             f"worst_abs={raw['absolute_error']:.10g} signed={ledger['signed_error']:+.10g} "
                             f"time={raw['time']} direction={raw['direction_index']} query={raw['query_index']}:{raw['query']['kind']}")
                lines.append(f"      {pieces}; reconstruction_residual={ledger['reconstruction_residual']:+.3g}")
                reference, prediction = ledger["reference_front"], ledger["prediction_front"]
                lines.append(f"      front support exact={reference['position']:.10g} predicted={prediction['position']:.10g}; "
                             f"signed mass exact={reference['weight']:+.10g} predicted={prediction['weight']:+.10g}")
                nonlinear_row = result["worst_nonlinear_row"]
                nonlinear, nonlinear_raw = nonlinear_row["attribution"]["nonlinear"], nonlinear_row["raw"]
                lines.append(f"      worst_nonlinear_abs={nonlinear['recorded_absolute_error']:.10g} "
                             f"signed={nonlinear['signed_error']:+.10g}; "
                             f"time={nonlinear_raw['time']} direction={nonlinear_raw['direction_index']} "
                             f"query={nonlinear_raw['query_index']}:{nonlinear_raw['query']['kind']}")
                nonlinear_pieces = " ".join(f"{key}={value:+.8g}" for key, value in nonlinear["contributions"].items())
                lines.append(f"      nonlinear terms: {nonlinear_pieces}; reconstruction_residual={nonlinear['reconstruction_residual']:+.3g}")
                lines.append(f"      max_abs_reconstruction_residual={result['max_reconstruction_residual']:.3g}; "
                             f"max_abs_nonlinear_reconstruction_residual={result['max_nonlinear_reconstruction_residual']:.3g}")
    lines += ["", "Use this ledger to diagnose exposed failures only. Changed studies require independent validation and a fresh final holdout."]
    return "\n".join(lines) + "\n"


def diagnose(root, review_dir, output_dir):
    root = project_root(root)
    output = confined(root, output_dir)
    requested = Path(output_dir)
    if not requested.is_absolute():
        requested = root / requested
    if output != requested.absolute() or output.exists() or output.is_symlink():
        raise ValueError(f"diagnostic output already exists or is aliased; choose a fresh canonical directory: {output}")
    if not output.parent.is_dir():
        raise ValueError("diagnostic output parent directory must already exist")
    helper_receipts = {name: digest_file(regular(root, root / "scripts" / name)) for name in HELPER_FILES}
    inputs = Inputs(root)
    reviewed, contexts = load_sources(inputs, review_dir, output)
    by_method, counters = {}, []
    for context in contexts:
        results, counter = stream_context(context)
        by_method[(context["run"]["run_id"], context["method"])] = results
        counters.append(counter | {"run_id": context["run"]["run_id"], "method": context["method"]})
    runs = []
    for run in reviewed["runs"]:
        parents = []
        for failure in run["methods"]["measure"]["failures"]:
            parent_id = failure["parent_id"]
            context = next(row for row in contexts if row["run"]["run_id"] == run["run_id"])
            parents.append({"parent_id": parent_id, "split": failure["split"], "family": context["parents"][parent_id]["family"],
                            "methods": {method: by_method[(run["run_id"], method)][parent_id] for method in METHODS}})
        runs.append({"run_id": run["run_id"], "seed": run["seed"],
                     "absolute_tolerance": run["evaluation_protocol"]["absolute_tolerance"], "parents": parents})
    summary = {"schema_version": 1, "status": "completed_exploratory_failure_attribution", "runs": runs,
               "review_directory": str(confined(root, review_dir)), "pooled_parent_count": None,
               "research_advantage_established": False, "raw_counters": counters,
               "decomposition": "fixed telescoping order: diffuse coefficient, diffuse support, atom weight, atom position; not a causal identification",
               "numerical_reconstruction_tolerance": "1e-10 * max(1, absolute reference, absolute prediction); separate from frozen weak gate"}
    report = report_text(summary)
    # Validate complete coverage and hashes before creating output. The second
    # streaming pass writes all selected rows without retaining the full logs.
    output.mkdir()
    failure_path = output / "failures.jsonl"
    try:
        with failure_path.open("xb") as destination:
            for context in contexts:
                results, counter = stream_context(context, destination)
                if results != by_method[(context["run"]["run_id"], context["method"])]:
                    raise ValueError("raw query evidence changed between validation and writing")
        json_bytes = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        text_bytes = report.encode()
        outputs = {"summary.json": hashlib.sha256(json_bytes).hexdigest(), "summary.txt": hashlib.sha256(text_bytes).hexdigest(),
                   "failures.jsonl": digest_file(failure_path)["sha256"]}
        raw_paths = {str(context["source"]["path"].relative_to(root)) for context in contexts}
        for name, expected in inputs.receipts.items():
            if name not in raw_paths and digest_file(regular(root, root / name)) != expected:
                raise ValueError(f"metadata input changed during diagnosis: {name}")
        for name, expected in helper_receipts.items():
            if digest_file(regular(root, root / "scripts" / name)) != expected:
                raise ValueError(f"diagnostic helper changed during calculation: {name}")
        helpers = {name: receipt["sha256"] for name, receipt in helper_receipts.items()}
        provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "project_root": str(root), "inputs": inputs.receipts,
                      "outputs": outputs, "helper_sha256": helpers, "raw_counters": counters,
                      "read_only_source_runs": [run["run_id"] for run in reviewed["runs"]]}
        for name, data in (("summary.json", json_bytes), ("summary.txt", text_bytes),
                           ("provenance.json", (json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())):
            with (output / name).open("xb") as stream:
                stream.write(data)
    except Exception as error:
        # Preserve partial diagnostic evidence; never overwrite or clean it.
        with (output / "failed.json").open("xb") as stream:
            stream.write((json.dumps({"status": "failed_diagnostic_output_preserved", "error": str(error)}) + "\n").encode())
        raise
    return summary, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        _, report = diagnose(args.project_root, args.review_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError) as error:
        print(f"SSMO: pilot failure diagnosis refused: {error}", file=sys.stderr)
        return 2
    print(report, end="")
    print(f"Saved diagnosis: {confined(project_root(args.project_root), args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
