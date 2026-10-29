from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image, ImageStat

from llmca.config import file_hash
from llmca.evaluation import evaluate_run
from llmca.execution import run_experiment
from llmca.figures import architecture
from llmca.generation import generate_to_file
from llmca.paper_pack import check_matrix, freeze_source
from scripts.verify_frozen import verify


class ReleaseTests(unittest.TestCase):
    def test_offline_guard_rejects_provider_before_writing(self) -> None:
        with self.assertRaisesRegex(ValueError, "offline"):
            run_experiment({"offline": True, "models": [{"kind": "openai_compatible"}]})

    def test_distance_is_stable_across_process_hash_seeds(self) -> None:
        code = ("from llmca.baselines import _distance; "
                "print(repr(_distance({'alpha': 1e-10, 'beta': 1, 'gamma': 'x'}, "
                "{'alpha': 1.0, 'beta': 0.9999999999, 'gamma': 'z'})))")
        outputs = [subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True) for seed in (1, 97)]
        self.assertEqual(outputs[0], outputs[1])

    def test_matrix_verification_rejects_missing_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "data.jsonl"
            generate_to_file({"dataset": {"base_scenarios_per_domain": 3}, "generation": {"perturbations": False}}, 42, source)
            config = {"run_id": "check", "offline": True, "dataset": {"input": str(source)},
                      "representations": ["json", "semantic_json"], "models": [{"name": "rules", "kind": "rule"}]}
            cwd = Path.cwd()
            try:
                os.chdir(root)
                run = run_experiment(config, seed=42).resolve()
                evaluate_run(run, resamples=20)
            finally:
                os.chdir(cwd)
            checked = check_matrix(run)
            self.assertEqual(checked["predictions"], 24)
            self.assertTrue(checked["offline_representation_predictions_identical"])
            path = run / "predictions" / "predictions.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "coverage"):
                check_matrix(run)

    def test_source_snapshot_uses_explicit_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src" / "llmca").mkdir(parents=True)
            (root / "src" / "llmca" / "example.py").write_text("value = 1\n")
            (root / ".env").write_text("LOCAL_SECRET=example\n")
            (root / "unrelated.docx").write_bytes(b"unrelated")
            snapshot = root / "snapshot.zip"
            freeze_source(root, snapshot)
            with zipfile.ZipFile(snapshot) as archive:
                self.assertEqual(archive.namelist(), ["src/llmca/example.py"])

    def test_frozen_file_hashes_detect_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data.txt"
            data.write_text("recorded result\n")
            (root / "freeze_manifest.json").write_text(json.dumps({"files": {"data.txt": file_hash(data)}}))
            self.assertEqual(verify(root), 1)
            data.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(root)

    def test_figures_are_real_raster_and_editable_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            png, svg = architecture(Path(td))
            with Image.open(png) as image:
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 300)
                self.assertGreater(ImageStat.Stat(image.convert("L")).var[0], 10)
            self.assertIn("<text", svg.read_text())
            self.assertIn("LLM-CA prototype architecture", svg.read_text())


if __name__ == "__main__":
    unittest.main()
