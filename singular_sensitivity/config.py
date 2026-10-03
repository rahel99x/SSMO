"""Frozen small-run configuration with explicit budgets and no plan placeholders."""
from __future__ import annotations

import json
from pathlib import Path

from .runtime import confined_path


def load_config(path: str | Path) -> dict:
    location = confined_path(path)
    if "PLAN" in location.name.upper():
        raise ValueError("Planning manifests cannot be executed")
    if location.suffix == ".json":
        config = json.loads(location.read_text())
    else:
        import yaml
        config = yaml.safe_load(location.read_text())
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected executable schema_version 1 configuration")
    if config.get("domain") != [-2.0, 2.0]:
        raise ValueError("First study domain must be frozen on [-2,2]")
    if config.get("device") not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("Only CPU or one task-visible cuda:0 is supported")
    training, data = config["training"], config["data"]
    for key in ("train_parents", "validation_parents", "test_parents"):
        if not 1 <= int(data[key]) <= 100000:
            raise ValueError(f"Invalid bounded parent count: {key}")
    for key, maximum in (("steps", 10000), ("batch_size", 256), ("directions_per_parent", 8), ("width", 128), ("depth", 4)):
        if not 1 <= int(training[key]) <= maximum:
            raise ValueError(f"Invalid first-study budget: {key}")
    if len(training.get("seeds", [0])) > 3:
        raise ValueError("First study uses at most three initialization seeds")
    if float(training["max_seconds"]) <= 0 or not 1 <= int(config.get("threads", 1)) <= 8:
        raise ValueError("Invalid runtime/thread budget")
    if config.get("precision", "fp32") != "fp32" or config.get("compile", False):
        raise ValueError("First validated training path is eager FP32; optimizations require a separate audit")
