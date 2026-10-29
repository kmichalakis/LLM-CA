from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_open_source_release import build_release, source_files


class OpenSourceReleaseTests(unittest.TestCase):
    def test_allowlist_excludes_inputs_outputs_secrets_and_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            for name in (
                "src/llmca/demo.py", "src/llmca.egg-info/generated.py",
                "tests/test_demo.py", ".github/workflows/tests.yml", ".env.example",
                ".env", "paper/draft.tex", "annotations/private.json",
                "data/input.jsonl", "runs/output.json", "releases/old.zip",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            (root / "src/llmca/linked.py").symlink_to(root / ".env")
            included = {path.relative_to(root).as_posix() for path in source_files(root)}
            self.assertEqual(included, {
                "src/llmca/demo.py", "tests/test_demo.py",
                ".github/workflows/tests.yml", ".env.example",
            })

    def test_archive_has_checksums_and_rejects_common_credential_patterns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            (root / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
            (root / "LICENSE").write_text("license fixture\n")
            archive, manifest = build_release(root, root / "releases")
            info = json.loads(manifest.read_text())
            self.assertEqual(info["file_count"], 2)
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                self.assertIn("LLM-CA-0.1.0/LICENSE", bundle.namelist())
            (root / "README.md").write_text("sk-" + "x" * 25)
            with self.assertRaisesRegex(ValueError, "Possible credential"):
                build_release(root, root / "rejected")
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
