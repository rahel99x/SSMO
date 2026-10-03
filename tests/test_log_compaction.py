"""Lossless log reduction must preserve evidence and reject unsafe scope."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from singular_sensitivity import log_compaction
from singular_sensitivity.runtime import confined_path, project_root


REPO = Path(__file__).resolve().parents[1]


class LogCompactionTests(unittest.TestCase):
    def setUp(self):
        base = confined_path("local/test-tmp")
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ssmo-compaction-", dir=base)
        self.artifacts = Path(self.temp.name) / "runs" / "current" / "artifacts"
        self.artifacts.mkdir(parents=True)
        self.evaluation = self.artifacts / "evaluate-measure-seed17"
        self.evaluation.mkdir()
        (self.evaluation / "evaluation.json").write_text(json.dumps({"status": "completed_scoped_pilot"}))
        self.raw = self.evaluation / "query_errors.jsonl"
        self.payload = (b'{"signed_error":-0.0001,"status":"regular","parent":"physical-001"}\n' * 30000)
        self.raw.write_bytes(self.payload)
        for name in ("representation.jsonl", "evaluation_parents.jsonl"):
            (self.evaluation / name).write_bytes(self.payload)

    def tearDown(self):
        self.temp.cleanup()

    def test_streamed_roundtrip_receipt_hash_and_storage_reduction(self):
        record = log_compaction.compact_logs(self.artifacts)
        archive = self.raw.with_suffix(".jsonl.gz")
        self.assertFalse(self.raw.exists())
        with gzip.open(archive, "rb") as stream:
            self.assertEqual(stream.read(), self.payload)
        entry = record["files"][0]
        self.assertEqual(entry["sha256_uncompressed"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(entry["original_bytes"], len(self.payload))
        self.assertLess(entry["compressed_bytes"], len(self.payload) / 10)
        self.assertEqual(entry["state"], "compacted")
        self.assertEqual(json.loads((self.artifacts / "log-compaction.json").read_text()), record)

    def test_training_active_and_other_run_logs_are_excluded(self):
        training = self.artifacts / "train-measure-seed17"
        training.mkdir()
        (training / "training.jsonl").write_bytes(self.payload)
        (training / "best.pt").write_bytes(b"selected checkpoint")
        incomplete = self.artifacts / "evaluate-incomplete"
        incomplete.mkdir()
        (incomplete / "query_errors.jsonl").write_bytes(self.payload)
        previous = self.artifacts.parent.parent / "previous" / "artifacts" / "evaluate-measure-seed17"
        previous.mkdir(parents=True)
        (previous / "query_errors.jsonl").write_bytes(self.payload)
        log_compaction.compact_logs(self.artifacts)
        self.assertEqual((training / "training.jsonl").read_bytes(), self.payload)
        self.assertEqual((training / "best.pt").read_bytes(), b"selected checkpoint")
        self.assertEqual((incomplete / "query_errors.jsonl").read_bytes(), self.payload)
        self.assertEqual((previous / "query_errors.jsonl").read_bytes(), self.payload)

    def test_failed_decompression_verification_keeps_original(self):
        real_digest = log_compaction._digest

        def corrupt_verification(stream):
            real_digest(stream)
            return "incorrect hash", len(self.payload)

        with mock.patch.object(log_compaction, "_digest", side_effect=corrupt_verification):
            with self.assertRaisesRegex(RuntimeError, "checksum/size"):
                log_compaction.compact_logs(self.artifacts)
        self.assertEqual(self.raw.read_bytes(), self.payload)
        self.assertFalse(self.raw.with_suffix(".jsonl.gz").exists())
        self.assertEqual(list(self.evaluation.glob("*.partial")), [])

    def test_changed_original_before_removal_is_preserved(self):
        real_receipt = log_compaction._receipt
        changed = self.payload + b'{"late_writer":true}\n'

        def write_then_change(path, record):
            real_receipt(path, record)
            self.raw.write_bytes(changed)

        with mock.patch.object(log_compaction, "_receipt", side_effect=write_then_change):
            with self.assertRaisesRegex(RuntimeError, "changed before removal"):
                log_compaction.compact_logs(self.artifacts)
        self.assertEqual(self.raw.read_bytes(), changed)
        self.assertTrue(self.raw.with_suffix(".jsonl.gz").exists())

    def test_archive_conflict_never_overwrites_or_removes_original(self):
        archive = self.raw.with_suffix(".jsonl.gz")
        archive.write_bytes(b"previous evidence")
        with self.assertRaises(FileExistsError):
            log_compaction.compact_logs(self.artifacts)
        self.assertEqual(archive.read_bytes(), b"previous evidence")
        self.assertEqual(self.raw.read_bytes(), self.payload)

    def test_receipt_failure_prevents_original_removal(self):
        real_receipt = log_compaction._receipt
        calls = 0

        def fail_first_write(path, record):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("mock receipt write failed")
            real_receipt(path, record)

        with mock.patch.object(log_compaction, "_receipt", side_effect=fail_first_write):
            with self.assertRaisesRegex(OSError, "receipt write failed"):
                log_compaction.compact_logs(self.artifacts)
        self.assertEqual(self.raw.read_bytes(), self.payload)

    def test_internal_symlink_cannot_modify_another_run(self):
        outside = self.artifacts.parent.parent / "previous" / "artifacts" / "original.jsonl"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(self.payload)
        self.raw.unlink()
        self.raw.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            log_compaction.compact_logs(self.artifacts)
        self.assertEqual(outside.read_bytes(), self.payload)

    def test_representation_marker_allows_only_its_named_raw_log(self):
        self.raw.unlink()
        (self.evaluation / "evaluation.json").unlink()
        directory = self.artifacts / "representation"
        directory.mkdir()
        (directory / "representation.json").write_text(json.dumps({"status": "completed_scoped_representation_audit"}))
        raw = directory / "representation.jsonl"
        raw.write_bytes(self.payload)
        untouched = directory / "arbitrary.jsonl"
        untouched.write_bytes(self.payload)
        record = log_compaction.compact_logs(self.artifacts)
        self.assertEqual(len(record["files"]), 1)
        self.assertFalse(raw.exists())
        self.assertTrue(untouched.exists())

    def test_completed_summary_with_missing_raw_evidence_is_an_error(self):
        self.raw.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing expected raw evidence"):
            log_compaction.compact_logs(self.artifacts)
        self.assertTrue((self.evaluation / "representation.jsonl").exists())

    def test_bash_helper_refuses_login_before_python_or_cache_creation(self):
        env = dict(os.environ, SSMO_PROJECT_ROOT=str(project_root()))
        env.pop("SSMO_ROOT", None)
        env.pop("SLURM_JOB_ID", None)
        result = subprocess.run(["bash", str(REPO / "scripts" / "compact_logs.sh"), str(self.artifacts.parent)],
                                text=True, capture_output=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires an allocated CARC", result.stderr)
        self.assertTrue(self.raw.exists())


if __name__ == "__main__":
    unittest.main()
