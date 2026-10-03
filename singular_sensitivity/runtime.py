"""Project-contained storage, atomic artifacts, provenance, and measured hardware.

No environment values or credentials are recorded. GPU records are observations
of the current task, never evidence of hardware that has not been allocated.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import tempfile
import time
from typing import Any


def project_root() -> Path:
    configured = os.environ.get("SSMO_PROJECT_ROOT") or os.environ.get("PROJECT_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parents[1]
    root = root.absolute()
    if not root.is_dir() or root.resolve() != root:
        raise ValueError("Project root must be an existing real directory, not an escaping symlink")
    return root


def confined_path(path: str | Path, root: str | Path | None = None) -> Path:
    base = Path(root).absolute() if root is not None else project_root()
    if base.resolve() != base:
        raise ValueError("Project root must not be a symlink")
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else base / candidate
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(base) or resolved == base:
        raise ValueError(f"Artifact path must stay strictly inside project root: {candidate}")
    return resolved


def initialize_storage(root: Path | None = None, threads: int = 1) -> None:
    base = root or project_root()
    cache = confined_path(base / ".cache", base)
    locations = {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "PIP_CACHE_DIR": "pip",
        "XDG_CACHE_HOME": "xdg", "XDG_CONFIG_HOME": "xdg-config", "XDG_DATA_HOME": "xdg-data",
        "TORCH_HOME": "torch", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton", "TORCH_EXTENSIONS_DIR": "extensions",
        "MPLCONFIGDIR": "matplotlib", "CUDA_CACHE_PATH": "cuda",
        "PYTHONPYCACHEPREFIX": "bytecode", "CCACHE_DIR": "ccache",
        "NUMBA_CACHE_DIR": "numba", "HF_HOME": "huggingface",
    }
    for name, subdir in locations.items():
        target = confined_path(cache / subdir, base)
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    tempfile.tempdir = os.environ["TMPDIR"]
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", threads))
    if threads < 1 or threads > allocated:
        raise ValueError("Thread count exceeds the task allocation")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    # Required by deterministic CUDA GEMM; set before any CUDA context exists.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = confined_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # The temporary file is on the same persistent filesystem as its destination.
    descriptor, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = confined_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def source_record() -> dict:
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in ("singular_sensitivity", "configs", "scripts", "slurm", "requirements", "tests"):
        for path in sorted((source / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.is_symlink():
                digest.update(str(path.relative_to(source)).encode())
                digest.update(path.read_bytes())
    record = {"source_sha256": digest.hexdigest(), "source_directory": str(source)}
    try:
        head = subprocess.run(["git", "-C", str(source), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True, timeout=5)
        record["git_commit"] = head.stdout.strip() if head.returncode == 0 else None
        status = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        record["git_dirty"] = bool(status.stdout) if status.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        record["git_commit"] = None
    return record


def memory_record(device: str = "cpu") -> dict:
    result = {"host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if str(device).startswith("cuda"):
        import torch
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      device_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        if result["peak_reserved_bytes"] > 0.8 * result["device_total_bytes"]:
            raise RuntimeError("Measured reserved GPU memory exceeds the 80% development budget")
    return result


def provenance(config: dict | None = None, dataset: dict | None = None) -> dict:
    versions = {}
    for name in ("numpy", "scipy", "PyYAML", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(), "platform": platform.platform(),
            "software": versions, "source": source_record(), "config_sha256": content_hash(config) if config is not None else None,
            "manifest_sha256": content_hash(dataset) if dataset is not None else None,
            "slurm": {name: os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK")},
            "direction_convention": "fixed physical x; (u_left-u_right)*Ds[v]; Euclidean parameter directions"}


def hardware_audit(device: str = "cpu", expected_model: str | None = None) -> dict:
    import torch
    record = {"device": device, "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda}
    if not str(device).startswith("cuda"):
        return record | {"cuda_executed": False}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU workflow requires exactly one task-visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    name = properties.name.lower().replace(" ", "")
    if "mig" in name:
        raise RuntimeError("MIG allocation changes the hardware protocol; review it before proceeding")
    expected = (expected_model or os.environ.get("SSMO_EXPECTED_GPU", "")).lower().replace(" ", "")
    expected = {"a10040": "a100", "a100-40gb": "a100"}.get(expected, expected)
    if not expected:
        raise ValueError("Set an explicit expected GPU model for a hardware-stratified run")
    if expected not in name or (expected == "l40" and "l40s" in name):
        raise RuntimeError(f"Expected {expected}, received {properties.name}")
    if expected == "a100" and not 38 * 2**30 <= properties.total_memory <= 42 * 2**30:
        raise RuntimeError("A100 protocol requires the verified 40 GB variant")
    memory_ranges = {"a30": (21, 25), "a40": (40, 49), "l40": (40, 49), "l40s": (40, 49)}
    if expected in memory_ranges:
        minimum, maximum = memory_ranges[expected]
        if not minimum * 2**30 <= properties.total_memory <= maximum * 2**30:
            raise RuntimeError(f"Unexpected {expected} memory capacity; review the hardware protocol")
    record.update(model=properties.name, total_memory_bytes=properties.total_memory, capability=list(torch.cuda.get_device_capability(0)))
    record["visible_device_uuid"] = str(getattr(properties, "uuid", "unavailable"))
    record["nvidia_smi_scope"] = "node hardware report; torch properties identify task-visible device zero"
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        record["nvidia_smi"] = smi.stdout.strip() if smi.returncode == 0 else {"unavailable": True, "returncode": smi.returncode}
    except (OSError, subprocess.TimeoutExpired):
        record["nvidia_smi"] = {"unavailable": True}
    # Execute real forward, JVP and mixed-derivative backward kernels in the same task.
    torch.cuda.reset_peak_memory_stats()
    x = torch.arange(16, device="cuda:0", dtype=torch.float64).reshape(4, 4) / 16
    weight = torch.nn.Parameter(torch.eye(4, device="cuda:0", dtype=torch.float64))
    value, tangent = torch.func.jvp(lambda z: (z @ weight).sin().sum(), (x,), (torch.ones_like(x),))
    tangent.square().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(value) or not torch.isfinite(tangent) or weight.grad is None or not torch.isfinite(weight.grad).all():
        raise RuntimeError("CUDA JVP/mixed-derivative kernel validation failed")
    return record | {"cuda_executed": True, "kernel_checks": ["forward", "jvp", "mixed_derivative_backward"], "memory": memory_record(device)}
