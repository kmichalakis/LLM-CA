"""One command for the frozen offline demonstration, tests and paper pack."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if any((root / path).exists() for path in ("runs/paper_v1", "validation/paper_v1", "paper_artifacts/v1")):
        raise FileExistsError("release outputs exist; reproduce in a fresh extracted source checkout")
    if not (root / "annotations/pilot/automatic/imported/review_summary.json").exists():
        raise FileNotFoundError("copy the frozen annotation_evidence folder to annotations/pilot first")
    env = {**os.environ, "PYTHONHASHSEED": "0", "MPLCONFIGDIR": str(root / ".matplotlib"),
           "XDG_CACHE_HOME": str(root / ".matplotlib" / "cache")}
    calls = [
        ["-m", "llmca.cli", "generate", "--config", "configs/benchmark.yaml", "--seed", "42", "--out", "data/scenarios/paper_v1.jsonl"],
        ["-m", "llmca.cli", "validate", "--input", "data/scenarios/paper_v1.jsonl", "--strict"],
        ["-m", "llmca.cli", "run", "--config", "configs/experiment_paper_v1.yaml", "--seed", "42"],
        ["-m", "llmca.cli", "evaluate", "--run", "runs/paper_v1", "--bootstrap-resamples", "2000", "--seed", "42"],
        ["-m", "llmca.cli", "report", "--run", "runs/paper_v1", "--format", "md,html"],
        ["scripts/validate_release.py", "--out", "validation/paper_v1"],
        ["scripts/check_reproducibility.py", "--out", "validation/paper_v1/reproducibility.json"],
        ["-m", "llmca.paper_pack"],
        ["scripts/verify_frozen.py", "paper_artifacts/v1"],
    ]
    for args in calls:
        subprocess.run([sys.executable, *args], check=True, env=env)


if __name__ == "__main__":
    main()
