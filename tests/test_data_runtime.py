import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from singular_sensitivity.data import generate_manifest, inference_input, validate_manifest
from singular_sensitivity.config import load_config
from singular_sensitivity.runtime import confined_path, initialize_storage, project_root


class DataRuntimeTests(unittest.TestCase):
    def test_generic_project_root_does_not_redirect_ssmo(self):
        with tempfile.TemporaryDirectory(dir=project_root() / ".cache/tmp") as other_project:
            with patch.dict(os.environ, {"PROJECT_ROOT": other_project,
                                         "SSMO_PROJECT_ROOT": "", "SSMO_ROOT": ""}):
                expected = Path(__file__).resolve().parents[1]
                self.assertEqual(project_root(), expected)
                self.assertEqual(confined_path("runs/namespace-check.json"),
                                 expected / "runs/namespace-check.json")

    def test_namespaced_roots_and_legacy_alias_select_same_storage(self):
        with tempfile.TemporaryDirectory(dir=project_root() / ".cache/tmp") as ssmo_directory:
            cases = [(ssmo_directory, ""), ("", ssmo_directory),
                     (ssmo_directory, ssmo_directory)]
            for configured, legacy in cases:
                with self.subTest(configured=configured, legacy=legacy):
                    with patch.dict(os.environ, {"SSMO_PROJECT_ROOT": configured,
                                                 "SSMO_ROOT": legacy, "PROJECT_ROOT": "/"}):
                        self.assertEqual(project_root(), Path(ssmo_directory))
                        self.assertEqual(confined_path("runs/namespace-check.json"),
                                         Path(ssmo_directory) / "runs/namespace-check.json")

    def test_conflicting_ssmo_root_settings_stop_artifact_resolution(self):
        with patch.dict(os.environ, {"SSMO_PROJECT_ROOT": str(project_root()),
                                     "SSMO_ROOT": "/another-project"}):
            with self.assertRaisesRegex(ValueError, "disagree"):
                confined_path("runs/namespace-check.json")

    def test_carc_storage_rejects_other_directory_before_creating_files(self):
        with tempfile.TemporaryDirectory(dir=project_root() / ".cache/tmp") as directory:
            approved = Path(directory) / "approved"
            unrelated = Path(directory) / "unrelated"
            with patch("singular_sensitivity.runtime.CARC_PROJECT_ROOT", approved):
                with patch.dict(os.environ, {"SLURM_JOB_ID": "mock-allocation"}):
                    with self.assertRaisesRegex(ValueError, "CARC storage"):
                        initialize_storage(unrelated)
            self.assertFalse(unrelated.exists())

    def test_carc_storage_uses_ssmo_and_rejects_prior_spelling(self):
        from singular_sensitivity.runtime import CARC_PROJECT_ROOT, _check_carc_storage
        self.assertEqual(CARC_PROJECT_ROOT, Path("/home1/aadaniel/projects/SSMO"))
        with patch.dict(os.environ, {"SLURM_JOB_ID": "mock-allocation"}):
            # The approved path passes policy without touching the filesystem.
            _check_carc_storage(CARC_PROJECT_ROOT)
            with patch("pathlib.Path.mkdir") as mkdir:
                with self.assertRaisesRegex(ValueError, "CARC storage"):
                    initialize_storage(Path("/home1/aadaniel/projects/SSNO"))
                mkdir.assert_not_called()

    def test_compact_reproducible_disjoint_parents(self):
        config = load_config(project_root() / "configs/smoke.yaml")
        manifest = generate_manifest(config)
        self.assertEqual(manifest, generate_manifest(config))
        validate_manifest(manifest)
        self.assertLess(len(json.dumps(manifest)), 200000)
        self.assertFalse(any("shock_positions" in record or "event_time" in record for record in manifest["parents"]))

    def test_input_has_no_teacher_information(self):
        record = {"parent_id": "p", "family": "shock", "parameters": [1, -1, 0], "teacher": {"future_shock": 0.4}}
        inp = inference_input(record, 0.5, (-2, 2))
        self.assertFalse(hasattr(inp, "teacher"))
        self.assertEqual(inp.parameters, (1, -1, 0))

    def test_reject_external_outputs_and_escaping_links(self):
        with self.assertRaises(ValueError):
            confined_path("/tmp/external-output.json")
        with tempfile.TemporaryDirectory(dir=project_root() / ".cache/tmp") as temporary:
            link = Path(temporary) / "escape"
            link.symlink_to("/tmp", target_is_directory=True)
            with self.assertRaises(ValueError):
                confined_path(link / "result.json")

    def test_detect_parent_leakage(self):
        config = load_config(project_root() / "configs/smoke.yaml")
        manifest = generate_manifest(config)
        duplicate = dict(manifest["parents"][0])
        duplicate.update(parent_id="another-id", split="test")
        manifest["parents"].append(duplicate)
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_equal_state_family_and_cut_do_not_create_another_parent(self):
        config = load_config(project_root() / "configs/smoke.yaml")
        manifest = generate_manifest(config)
        original = next(p for p in manifest["parents"] if p["split"] == "train" and p["family"] == "constant")
        duplicate = dict(original)
        duplicate.update(parent_id="fake-new-shock", split="test", family="shock", parameters=[*original["parameters"][:2], 0.9])
        manifest["parents"].append(duplicate)
        with self.assertRaises(ValueError):
            validate_manifest(manifest)
