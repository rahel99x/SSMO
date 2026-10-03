"""Lossless compaction of known completed raw logs in one current run only."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from .runtime import atomic_write_json, confined_path


BLOCK_BYTES = 1024 * 1024
EVALUATION_LOGS = ("query_errors.jsonl", "representation.jsonl", "evaluation_parents.jsonl")


def _path(path: Path, artifacts: Path) -> Path:
    resolved = confined_path(path)
    if resolved != path.absolute() or not resolved.is_relative_to(artifacts):
        raise ValueError(f"Compaction rejects symlinks or paths outside current artifacts: {path}")
    return resolved


def _signature(path: Path) -> tuple:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Compaction requires a regular file with one hard link: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _digest(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while block := stream.read(BLOCK_BYTES):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _receipt(path: Path, record: dict) -> None:
    atomic_write_json(path, record)
    _sync_directory(path.parent)


def _compress(original: Path, archive: Path) -> tuple[tuple, str, int]:
    before = _signature(original)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + archive.name + ".", suffix=".partial", dir=original.parent)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "wb") as output:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=1, mtime=0, fileobj=output) as compressed:
                with os.fdopen(os.open(original, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
                    while block := source.read(BLOCK_BYTES):
                        digest.update(block)
                        size += len(block)
                        compressed.write(block)
            output.flush()
            os.fsync(output.fileno())
        expected = (digest.hexdigest(), size)
        with gzip.open(temporary, "rb") as restored:
            if _digest(restored) != expected:
                raise RuntimeError("Compressed log failed decompressed checksum/size verification")
        if _signature(original) != before or size != before[2]:
            raise RuntimeError("Original log changed during compaction; preserved it")
        # link is an atomic no-overwrite publication on the same filesystem.
        os.link(temporary, archive)
        _sync_directory(archive.parent)
        return before, expected[0], size
    finally:
        temporary.unlink(missing_ok=True)


def _completed(directory: Path, summary_name: str, status: str, artifacts: Path) -> bool:
    summary = _path(directory / summary_name, artifacts)
    if not summary.is_file():
        return False
    with summary.open() as stream:
        try:
            value = json.load(stream)
        except json.JSONDecodeError:
            return False
    return isinstance(value, dict) and value.get("status") == status


def compact_logs(artifact_directory: str | Path) -> dict:
    """Compact only completed evaluation/representation logs; retain a mapping.

    A verified, durably recorded archive precedes each original removal. Failed
    verification keeps the original; already completed entries remain recorded.
    Source-run artifacts, active logs and model/training files are never visited.
    """
    requested = Path(artifact_directory)
    artifacts = confined_path(requested)
    if artifacts != requested.absolute() or artifacts.name != "artifacts" or artifacts.parent.parent.name != "runs" or not artifacts.is_dir():
        raise ValueError("Compaction requires the real current runs/<id>/artifacts directory")
    receipt_path = _path(artifacts / "log-compaction.json", artifacts)
    if receipt_path.exists():
        raise FileExistsError("Compaction receipt exists; preserve it and use a fresh run")
    candidates = []
    for directory in sorted(artifacts.glob("evaluate-*")):
        directory = _path(directory, artifacts)
        if directory.is_dir() and _completed(directory, "evaluation.json", "completed_scoped_pilot", artifacts):
            candidates.extend(directory / name for name in EVALUATION_LOGS)
    representation = _path(artifacts / "representation", artifacts)
    if representation.is_dir() and _completed(representation, "representation.json", "completed_scoped_representation_audit", artifacts):
        candidates.append(representation / "representation.jsonl")
    record = {"status": "completed_lossless_log_compaction", "gzip_level": 1, "block_bytes": BLOCK_BYTES,
              "scope": "current run only; summaries retain original paths, resolved by this mapping", "files": []}
    try:
        for candidate in candidates:
            original = _path(candidate, artifacts)
            if not original.exists():
                raise FileNotFoundError(f"Completed stage is missing expected raw evidence: {original}")
            archive = _path(original.with_suffix(original.suffix + ".gz"), artifacts)
            if archive.exists():
                raise FileExistsError(f"Archive exists; original preserved: {archive}")
            before, checksum, size = _compress(original, archive)
            entry = {"original": str(original.relative_to(artifacts)), "archive": str(archive.relative_to(artifacts)),
                     "sha256_uncompressed": checksum, "original_bytes": size, "compressed_bytes": archive.stat().st_size,
                     "state": "verified_archive_original_preserved"}
            record["files"].append(entry)
            _receipt(receipt_path, record)
            with os.fdopen(os.open(original, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
                unchanged = _digest(source) == (checksum, size)
            if not unchanged or _signature(original) != before:
                raise RuntimeError("Original log changed before removal; preserved it")
            original.unlink()
            entry["state"] = "compacted"
            _sync_directory(original.parent)
            _receipt(receipt_path, record)
    except Exception as error:
        record["status"] = "failed_compaction_verified_archives_recorded"
        record["error"] = str(error)
        _receipt(receipt_path, record)
        raise
    _receipt(receipt_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    result = compact_logs(args.artifacts)
    print(f"Losslessly compacted {len(result['files'])} known completed raw logs; see log-compaction.json")


if __name__ == "__main__":
    main()
