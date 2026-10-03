"""Compact physical-parent manifests. No trajectories or teacher labels as inputs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import FORMULA_VERSION
from .runtime import atomic_write_json, confined_path, content_hash


@dataclass(frozen=True)
class InferenceInput:
    parent_id: str
    family: str
    parameters: tuple[float, ...]
    time: float
    domain: tuple[float, float]


@dataclass(frozen=True)
class ReferenceLabel:
    """A teacher result is a separate object and is never passed to the encoder."""
    sensitivity: Any
    tier: int = 0
    formula_version: str = FORMULA_VERSION


def inference_input(record: dict, time: float, domain: tuple[float, float]) -> InferenceInput:
    return InferenceInput(record["parent_id"], record["family"], tuple(record["parameters"]), time, domain)


def _directions(rng: np.random.Generator, count: int, dimension: int) -> list[list[float]]:
    random = rng.normal(size=(max(1, count), dimension))
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    # Zero direction is an explicit held-in control, never normalized by its norm.
    return [*random.tolist(), [0.0] * dimension]


def generate_manifest(config: dict) -> dict:
    options = config["data"]
    seed = int(options["seed"])
    rng = np.random.default_rng(seed)
    domain = list(config.get("domain", [-2.0, 2.0]))
    if domain != [-2.0, 2.0]:
        raise ValueError("The first manifest protocol is frozen on [-2,2]; revise the protocol for another domain")
    times = [float(t) for t in options.get("times", [0.15, 0.4, 0.65])]
    if not times or not all(0 < t <= 0.8 for t in times):
        raise ValueError("Manifest times must lie in (0,0.8] without boundary interactions")
    count_directions = int(config["training"].get("directions_per_parent", 3))
    parents = []
    seen = set()
    for split, key in (("train", "train_parents"), ("validation", "validation_parents"), ("test", "test_parents"), ("ood", "ood_parents")):
        for index in range(int(options.get(key, 0))):
            # Distinct parents carry every time/direction/resolution/query variant.
            family = "constant" if index % 8 == 0 and split != "ood" else "shock"
            if family == "constant":
                level = float(rng.uniform(-0.6, 0.6))
                parameters = [level, level, float(rng.uniform(-0.5, 0.5))]
            elif split == "ood":
                parameters = [float(rng.uniform(1.3, 1.8)), float(rng.uniform(-1.8, -1.3)),
                              float(rng.choice([-1, 1]) * rng.uniform(0.65, 0.9))]
            else:
                parameters = [float(rng.uniform(0.1, 1.2)), float(rng.uniform(-1.2, -0.1)), float(rng.uniform(-0.5, 0.5))]
            if tuple(parameters) in seen:
                raise RuntimeError("Repeated physical parent across splits")
            seen.add(tuple(parameters))
            parents.append({"parent_id": f"{split}-{index:06d}", "split": split, "family": family,
                            "parameters": parameters, "times": times,
                            "directions": _directions(rng, count_directions, 3),
                            "boundary_conditions": "constant exterior states; no interaction with observation interval boundary",
                            "reference_tier": 0})
    audit_examples = [
        ("shock", [2.0, -0.5, 0.1], [0.6], [[0.2, -0.3, 0.4]]),
        ("rarefaction", [-0.8, 0.9, 0.0], [0.2, 0.6], [[0.2, -0.3, 0.4], [0, 0, 0]]),
        ("collision", [1.5, 0.5, -0.5, -0.4, 0.2], [0.3, 0.6 - 1e-9, 0.6, 0.6 + 1e-9, 0.8], [[0.2, -0.1, 0.3, 0.4, -0.2]]),
    ]
    for index, (family, parameters, audit_times, directions) in enumerate(audit_examples):
        parents.append({"parent_id": f"audit-{index:06d}", "split": "audit", "family": family,
                        "parameters": parameters, "times": audit_times, "directions": directions,
                        "reference_tier": 0})
    return {"schema_version": 1, "formula_version": FORMULA_VERSION, "seed": seed, "domain": domain,
            "protocol": {"flux": "F(u)=u^2/2", "input_access": "initial-condition-only",
                         "direction_norm": "Euclidean in (uL,uR,a), or (uL,uM,uR,a,b)",
                         "split_unit": "physical parent including all directions, times, grids and queries",
                         "iid_parameter_ranges": [[0.1, 1.2], [-1.2, -0.1], [-0.5, 0.5]],
                         "ood_claim": "held-out parameter/location ranges, same shock topology",
                         "test_selection": "sealed from training and checkpoint selection",
                         "event_policy": "exact collision unresolved unless one-sided limit is validated"},
            "parents": parents}


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("formula_version") != FORMULA_VERSION:
        raise ValueError("Unsupported parent manifest/formula version")
    ids, identities = set(), {}
    for record in manifest["parents"]:
        if record["parent_id"] in ids:
            raise ValueError("Duplicate parent ID")
        ids.add(record["parent_id"])
        parameters = tuple(record["parameters"])
        if record["family"] in {"shock", "rarefaction", "constant"} and len(parameters) == 3:
            # The entropy family is a label, not another physical parent. An
            # equal-state Riemann problem is independent of its fictitious cut.
            identity = ("burgers-riemann", parameters[:2] if parameters[0] == parameters[1] else parameters)
        else:
            identity = (record["family"], parameters)
        if identity in identities and identities[identity] != record["split"]:
            raise ValueError("Physical-parent leakage across splits")
        identities[identity] = record["split"]
        if record["split"] not in {"train", "validation", "test", "ood", "audit"}:
            raise ValueError("Unknown parent split")
        if not np.isfinite(record["parameters"]).all() or not np.isfinite(record["directions"]).all():
            raise ValueError("Nonfinite parent input/directions")
        if any(len(v) != len(record["parameters"]) for v in record["directions"]):
            raise ValueError("Direction dimension differs from parameter definition")


def save_manifest(config: dict, run_dir: Path) -> dict:
    run_dir = confined_path(run_dir)
    result = generate_manifest(config)
    validate_manifest(result)
    destination = run_dir / "parents.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if content_hash(previous) != content_hash(result):
            raise FileExistsError("An immutable parent manifest already exists with another protocol")
    else:
        atomic_write_json(destination, result)
    return result


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(confined_path(path).read_text())
    validate_manifest(manifest)
    return manifest
