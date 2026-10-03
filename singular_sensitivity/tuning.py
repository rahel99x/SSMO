"""Small CPU validation-only search; never a CARC or GPU optimum claim.

Each candidate runs in a fresh process so RSS and complete invocation costs
include imports, label construction, training, validation and checkpoint I/O.
The search uses a fixed manifest and the existing independent validation bank;
it never calls final evaluation or selects on test/OOD/audit records.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .runtime import (atomic_write_json, confined_path, content_hash,
                      initialize_storage, project_root, provenance)


CANDIDATES = (
    {"id": "baseline", "width": 64, "depth": 3, "lr": 0.001, "batch_size": 32},
    {"id": "small-low-lr", "width": 32, "depth": 2, "lr": 0.001, "batch_size": 16},
    {"id": "small", "width": 32, "depth": 2, "lr": 0.003, "batch_size": 16},
    {"id": "small-batch32", "width": 32, "depth": 2, "lr": 0.003, "batch_size": 32},
    {"id": "small-depth3", "width": 32, "depth": 3, "lr": 0.003, "batch_size": 16},
    {"id": "wide-depth2", "width": 64, "depth": 2, "lr": 0.003, "batch_size": 32},
)
SEEDS = (17, 29)


def pareto_front(records: list[dict]) -> list[dict]:
    """Return successful finite validation-score/complete-time nondominated rows."""
    valid = [row for row in records if row.get("status") == "complete"
             and math.isfinite(row["score"]) and row["score"] >= 0
             and math.isfinite(row["complete_seconds"]) and row["complete_seconds"] > 0]
    return [row for row in valid if not any(
        other["score"] <= row["score"] and other["complete_seconds"] <= row["complete_seconds"]
        and (other["score"] < row["score"] or other["complete_seconds"] < row["complete_seconds"])
        for other in valid)]


def select_confirmed(records: list[dict], seeds=SEEDS) -> dict:
    """Prefer minimum worst-seed validation score; use time within 5% of it.

    An incomplete or failed replicate excludes that configuration. Runtime
    measurements do not replace validation quality, and no test result enters.
    """
    grouped = {}
    for row in records:
        grouped.setdefault(row["candidate_id"], []).append(row)
    confirmed = []
    for candidate_id, rows in grouped.items():
        if {row["seed"] for row in rows} != set(seeds) or len(rows) != len(seeds):
            continue
        if any(row.get("status") != "complete" or not math.isfinite(row["score"])
               or row["score"] < 0 or not math.isfinite(row["complete_seconds"])
               or row["complete_seconds"] <= 0 for row in rows):
            continue
        confirmed.append({"candidate_id": candidate_id,
                          "worst_seed_score": max(row["score"] for row in rows),
                          "mean_complete_seconds": sum(row["complete_seconds"] for row in rows) / len(rows),
                          "peak_host_rss_bytes": max(row["host_peak_rss_bytes"] for row in rows)})
    if not confirmed:
        raise ValueError("no configuration has complete finite results for every confirmation seed")
    lowest = min(row["worst_seed_score"] for row in confirmed)
    eligible = [row for row in confirmed if row["worst_seed_score"] <= lowest * 1.05]
    return min(eligible, key=lambda row: (row["mean_complete_seconds"], row["candidate_id"]))


def _worker(config_path: Path, manifest_path: Path, run_dir: Path) -> int:
    from .config import load_config
    from .data import load_manifest
    import torch
    from .training import train

    config = load_config(config_path)
    torch.set_num_threads(1)
    dataset = load_manifest(manifest_path)
    summary = train(config, dataset, run_dir, device="cpu")
    from .models import ChartModel
    summary["parameter_count"] = sum(parameter.numel() for parameter in ChartModel(
        width=config["training"]["width"], depth=config["training"]["depth"]).parameters())
    summary["scientific_source_sha256"] = {name: hashlib.sha256(
        (Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("training.py", "models.py", "data.py", "reference.py", "queries.py")}
    atomic_write_json(run_dir / "worker_summary.json", summary)
    return 0 if summary["status"] == "complete" else 3


def run_search(config: dict, run_dir: Path, max_seconds: float = 300) -> dict:
    if not 1 <= max_seconds <= 300:
        raise ValueError("local search budget must be between 1 and 300 seconds")
    run_dir = confined_path(run_dir)
    if run_dir.exists():
        raise FileExistsError("search requires a fresh project-local run directory")
    run_dir.mkdir(parents=True)
    started = time.monotonic()
    from .data import save_manifest

    effective = copy.deepcopy(config)
    effective.update(device="cpu", precision="fp32", compile=False, threads=1, method="measure")
    effective["data"] = {"seed": 1729, "train_parents": 512, "validation_parents": 64,
                         "test_parents": 64, "ood_parents": 32, "times": [0.15, 0.4, 0.65]}
    effective["training"].update(steps=500, directions_per_parent=3, eval_every=25,
                                 patience=6, min_delta=0.00000001, checkpoint_every=50,
                                 max_seconds=40)
    manifest = save_manifest(effective, run_dir / "data")
    records = []

    def launch(candidate, seed):
        remaining = max_seconds - (time.monotonic() - started)
        if remaining < 3:
            return False
        candidate_dir = confined_path(run_dir / f"{candidate['id']}-seed{seed}")
        candidate_dir.mkdir()
        candidate_config = copy.deepcopy(effective)
        candidate_config["training"].update({key: value for key, value in candidate.items() if key != "id"})
        candidate_config["training"]["seed"] = seed
        atomic_write_json(candidate_dir / "config.json", candidate_config)
        candidate_started = time.monotonic()
        status = "failed"
        returncode = None
        with (candidate_dir / "worker.stdout").open("w") as stdout, (candidate_dir / "worker.stderr").open("w") as stderr:
            try:
                result = subprocess.run([sys.executable, "-m", "singular_sensitivity.tuning", "--worker", "--config",
                                         str(candidate_dir / "config.json"), "--manifest",
                                         str(run_dir / "data" / "parents.json"), "--run-dir", str(candidate_dir)],
                                        cwd=project_root(), stdout=stdout, stderr=stderr,
                                        timeout=min(50, remaining), check=False)
                returncode = result.returncode
                status = "complete" if returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        row = {"candidate_id": candidate["id"], "settings": dict(candidate), "seed": seed,
               "status": status, "returncode": returncode,
               "complete_seconds": time.monotonic() - candidate_started}
        if (candidate_dir / "worker_summary.json").exists():
            summary = json.loads((candidate_dir / "worker_summary.json").read_text())
            row.update(score=summary["best_validation_score"], best_step=summary["best_step"],
                       steps_completed=summary["steps_completed"], stopped_reason=summary["stopped_reason"],
                       host_peak_rss_bytes=summary["host_peak_rss_bytes"],
                       parameter_count=summary["parameter_count"],
                       training_teacher_seconds=summary["training_teacher_seconds"],
                       validation_teacher_seconds=summary["validation_teacher_seconds"],
                       training_elapsed_seconds=summary["elapsed_seconds"],
                       scientific_source_sha256=summary["scientific_source_sha256"])
        records.append(row)
        atomic_write_json(run_dir / "records.json", records)
        print(json.dumps(row, sort_keys=True), flush=True)
        return True

    for candidate in CANDIDATES:
        if not launch(candidate, SEEDS[0]):
            break
    front = pareto_front(records)
    if not front:
        raise RuntimeError("no completed candidate; failed results remain in the search directory")
    # Confirm best accuracy, the fastest near-best (within 5%) score/time
    # contender and the fastest Pareto alternative. At most three finalists.
    best = min(front, key=lambda row: (row["score"], row["candidate_id"]))
    fastest = min(front, key=lambda row: (row["complete_seconds"], row["candidate_id"]))
    shortlist = [best["candidate_id"]]
    near_best = [row for row in front if row["score"] <= best["score"] * 1.05]
    near_fastest = min(near_best, key=lambda row: (row["complete_seconds"], row["candidate_id"]))
    if near_fastest["candidate_id"] not in shortlist:
        shortlist.append(near_fastest["candidate_id"])
    if fastest["candidate_id"] != best["candidate_id"]:
        if fastest["candidate_id"] not in shortlist:
            shortlist.append(fastest["candidate_id"])
    else:
        alternate = sorted((row for row in records if row.get("status") == "complete"
                            and row["candidate_id"] != best["candidate_id"]),
                           key=lambda row: (row["score"], row["complete_seconds"]))
        if alternate:
            shortlist.append(alternate[0]["candidate_id"])
    for candidate_id in shortlist:
        candidate = next(candidate for candidate in CANDIDATES if candidate["id"] == candidate_id)
        if not launch(candidate, SEEDS[1]):
            break
    selected = select_confirmed(records)
    source_hashes = [row["scientific_source_sha256"] for row in records if row.get("status") == "complete"]
    if any(record != source_hashes[0] for record in source_hashes):
        raise RuntimeError("scientific source changed during the search; trials remain preserved")
    result = {"status": "complete", "selection": selected,
              "selected_settings": next(candidate for candidate in CANDIDATES
                                        if candidate["id"] == selected["candidate_id"]),
              "records": records, "first_seed_pareto_ids": [row["candidate_id"] for row in front],
              "confirmation_ids": shortlist, "complete_search_seconds": time.monotonic() - started,
              "budget_seconds": max_seconds, "manifest_hash": content_hash(manifest),
              "selection_protocol": "saved-best fixed independent validation state_mse+weak_mse; min_delta=1e-8; minimum worst two-seed score; fastest within 5%",
              "claims": {"sealed_test_used_for_selection": False, "carc_measured": False,
                         "gpu_optimum_established": False, "global_optimum_established": False}}
    atomic_write_json(run_dir / "tuning_summary.json", result)
    record = provenance(effective, manifest)
    record["complete_search_seconds"] = result["complete_search_seconds"]
    atomic_write_json(run_dir / "provenance.json", record)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/carc_pilot.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    initialize_storage(threads=1)
    if args.worker:
        return _worker(confined_path(args.config), confined_path(args.manifest), confined_path(args.run_dir))
    from .config import load_config
    result = run_search(load_config(args.config), confined_path(args.run_dir), args.max_seconds)
    print(json.dumps(result["selection"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
