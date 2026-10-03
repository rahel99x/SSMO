import json
from pathlib import Path
import tempfile
import unittest

from singular_sensitivity.data import generate_manifest, inference_input, validate_manifest
from singular_sensitivity.config import load_config
from singular_sensitivity.runtime import confined_path, project_root


class DataRuntimeTests(unittest.TestCase):
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
