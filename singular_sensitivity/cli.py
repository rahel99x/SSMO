"""Bounded, explicit research stages. No scheduler submission occurs in Python."""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

from .runtime import (atomic_write_json, confined_path, hardware_audit,
                      initialize_storage, memory_record, project_root, provenance)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "generate", "train", "evaluate", "representation", "numerical", "gpu-audit", "inverse", "report", "plan"):
        sub = subcommands.add_parser(command)
        sub.add_argument("--config", default="configs/smoke.yaml")
        if command != "plan":
            sub.add_argument("--run-dir", required=True)
        if command in {"train", "evaluate"}:
            sub.add_argument("--manifest", required=True)
        if command == "train":
            sub.add_argument("--method", choices=["measure", "state_only"], default="measure")
            sub.add_argument("--seed", type=int, default=None)
            sub.add_argument("--resume")
        if command in {"evaluate", "inverse"}:
            sub.add_argument("--checkpoint", required=command == "evaluate")
        if command == "evaluate":
            sub.add_argument("--state-only-checkpoint")
        if command == "gpu-audit":
            sub.add_argument("--expected-model", required=True)
        if command == "report":
            sub.add_argument("--artifact-source", help="Existing project-contained artifacts to include without modifying them")
        if command in {"train", "evaluate", "inverse"}:
            sub.add_argument("--device", choices=["cpu", "cuda:0"])
    return parser


def _audit(run_dir: Path) -> dict:
    tests = Path(__file__).resolve().parents[1] / "tests"
    suite = unittest.TestLoader().discover(str(tests), pattern="test_*.py")
    if suite.countTestCases() == 0:
        raise RuntimeError("A zero-test audit does not validate the workflow")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (run_dir / "tests.log").write_text(stream.getvalue())
    summary = {"tests_executed": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses),
               "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped) - len(result.expectedFailures) - len(result.unexpectedSuccesses),
               "status": "passed" if result.wasSuccessful() else "failed", "execution": "CPU tests; mocked Slurm is not CARC execution"}
    atomic_write_json(run_dir / "audit.json", summary)
    if not result.wasSuccessful():
        print(stream.getvalue(), file=sys.stderr)
    return summary


def _plan(config: dict) -> dict:
    counts = {key: value for key, value in config["data"].items() if key.endswith("parents")}
    return {"workflow": "exact scalar Burgers controls; initial-data-only single-front learning",
            "device": config["device"], "counts": counts,
            "training": config["training"], "evaluation": config["evaluation"],
            "required_checks": ["FP64 mathematical and mixed-gradient audit", "parent-separated smoke", "checkpoint resume", "allocated GPU kernel/memory calibration before expansion"],
            "gated_extensions": ["learned multiple-front/event architecture", "refined finite-volume teachers", "2-D/systems", "BF16/compile", "confirmatory efficacy/cost claim"],
            "submission": "none; scripts/submit.sh has a separate explicit --submit mode"}


def _report(run_dir: Path, artifact_source: str | Path | None = None) -> dict:
    run_dir = confined_path(run_dir)
    sources = []
    excluded = {"source", ".cache", ".venv", ".git", "local", "__pycache__"}
    if artifact_source is not None:
        source = confined_path(artifact_source)
        if not source.is_dir():
            raise ValueError("Report artifact source must be an existing project-contained directory")
        if source.name in excluded:
            raise ValueError("Report artifact source cannot be a source snapshot or cache directory")
        sources.append(source)
    if run_dir not in sources:
        sources.append(run_dir)
    artifacts = []
    seen = set()
    for source in sources:
        # Prune source snapshots and caches before walking them. Never follow
        # directory symlinks; validate paths before reading individual files.
        for directory, children, files in os.walk(source, followlinks=False):
            children[:] = sorted(name for name in children if name not in excluded)
            for name in children:
                confined_path(Path(directory) / name)
            for name in sorted(files):
                if not name.endswith(".json") or name in {"parents.json", "provenance.json", "report.json"}:
                    continue
                path = Path(directory) / name
                canonical = confined_path(path)
                if canonical in seen:
                    continue
                seen.add(canonical)
                try:
                    value = json.loads(canonical.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    artifacts.append({"source_directory": str(source), "path": str(path.relative_to(source)),
                                      "status": value.get("status"), "steps_completed": value.get("steps_completed"),
                                      "summary": value})
    report = {"status": "collected", "artifacts": artifacts,
              "artifact_sources": [str(source) for source in sources],
              "claims": {"current_instance_artifacts_only": sources == [run_dir], "carc_jobs_validated": False,
                         "research_advantage_established": False,
                         "note": "Parent job/accounting results and hardware-stratified held-out tolerance/cost gates need an actual CARC run."}}
    atomic_write_json(run_dir / "report.json", report)
    return report


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    # Establish storage before importing NumPy/PyTorch or executing any workload.
    initialize_storage(threads=1)
    from .config import load_config
    config = load_config(args.config)
    threads = min(int(config.get("threads", 1)), int(os.environ.get("SLURM_CPUS_PER_TASK", config.get("threads", 1))))
    initialize_storage(threads=threads)
    if args.command == "plan":
        print(json.dumps(_plan(config), indent=2))
        return 0
    run_dir = confined_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "provenance.json").exists() and args.command not in {"generate", "report"} and not (args.command == "train" and args.resume):
        print("Stage artifacts already exist; select a fresh run directory to preserve earlier results", file=sys.stderr)
        return 2
    started = time.monotonic()
    dataset = None
    try:
        if args.command in {"train", "evaluate"}:
            from .data import load_manifest
            dataset = load_manifest(args.manifest)
        device = getattr(args, "device", None) or config["device"]
        if args.command == "audit":
            result = _audit(run_dir)
        elif args.command == "generate":
            from .data import save_manifest
            dataset = save_manifest(config, run_dir)
            result = {"status": "generated", "parents": len(dataset["parents"]), "manifest": str(run_dir / "parents.json")}
        elif args.command == "train":
            from .training import train
            import torch
            torch.set_num_threads(threads)
            effective = copy.deepcopy(config)
            effective["device"] = device
            effective["method"] = args.method
            effective["training"]["seed"] = args.seed if args.seed is not None else config["training"]["seeds"][0]
            atomic_write_json(run_dir / "hardware.json", hardware_audit(device))
            result = train(effective, dataset, run_dir, device=device,
                           resume=confined_path(args.resume) if args.resume else None)
            config = effective
        elif args.command == "evaluate":
            from .validation import evaluate
            import torch
            torch.set_num_threads(threads)
            atomic_write_json(run_dir / "hardware.json", hardware_audit(device))
            result = evaluate(config, dataset, run_dir, checkpoint=confined_path(args.checkpoint), device=device,
                              state_only_checkpoint=confined_path(args.state_only_checkpoint) if args.state_only_checkpoint else None)
        elif args.command == "representation":
            from .validation import representation_study
            result = representation_study(config, run_dir)
        elif args.command == "numerical":
            from .numerical import numerical_audit
            result = numerical_audit(config, run_dir)
        elif args.command == "gpu-audit":
            device = "cuda:0"
            result = hardware_audit("cuda:0", args.expected_model)
            result["status"] = "passed"
            atomic_write_json(run_dir / "gpu_audit.json", result)
        elif args.command == "inverse":
            from .inverse import inverse_tracking
            provider = None
            if args.checkpoint:
                import numpy as np
                import torch
                torch.set_num_threads(threads)
                from .training import load_model
                from .models import chart_measure, squared_objective_gradient
                atomic_write_json(run_dir / "hardware.json", hardware_audit(device))
                model, _ = load_model(confined_path(args.checkpoint), device=device)
                def provider(alpha, observation_time, target):
                    with torch.no_grad():
                        a = torch.tensor(np.asarray(alpha)[None], dtype=torch.float32, device=device)
                        t = torch.tensor([observation_time], dtype=torch.float32, device=device)
                        v = torch.eye(3, dtype=torch.float32, device=device)[None]
                        measure = chart_measure(model, a, t, v)
                        if not bool(measure.atom_mask.all()):
                            raise ValueError("Learned support outside declared regularity stratum")
                        return squared_objective_gradient(measure, target)[0].cpu().numpy()
            options = config.get("inverse", {})
            result = inverse_tracking(tuple(config["domain"]), steps=int(options.get("steps", 8)),
                                      learning_rate=float(options.get("lr", 0.1)), gradient_provider=provider)
            atomic_write_json(run_dir / "inverse.json", result)
        elif args.command == "report":
            result = _report(run_dir, args.artifact_source)
        else:
            raise AssertionError("Unhandled stage")
        record = provenance(config, dataset)
        record.update(command=args.command, elapsed_seconds=time.monotonic() - started,
                      task_threads=threads,
                      memory=memory_record(device if args.command in {"train", "evaluate", "gpu-audit", "inverse"} else "cpu"))
        atomic_write_json(run_dir / "provenance.json", record)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 75 if result.get("status") == "paused" else 1 if result.get("status") == "failed" else 0
    except Exception as error:
        failure_file = run_dir / "failure.json"
        if failure_file.exists():
            failure_file = run_dir / ("failure-" + str(time.time_ns()) + ".json")
        atomic_write_json(failure_file, {"status": "failed", "command": args.command,
                          "error_type": type(error).__name__, "reason": str(error),
                          "elapsed_seconds": time.monotonic() - started, "provenance": provenance(config, dataset)})
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
