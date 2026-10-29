"""Check full input regeneration and predictions across fresh Python hash seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from llmca.config import file_hash, load_config, stable_hash
from llmca.generation import generate_to_file
from llmca.io import read_jsonl
from llmca.models import Scenario


def normalized(path: Path) -> dict[tuple[str, str, str], str]:
    result = {}
    for row in read_jsonl(path):
        key = (row["scenario_id"], row["model_name"], row["representation"])
        if key in result:
            raise ValueError("duplicate prediction during replay")
        result[key] = stable_hash(row["prediction"])
    return result


def check(out: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    original = root / "runs/paper_v1"
    actual_dataset = root / "data/scenarios/paper_v1.jsonl"
    reference = normalized(original / "predictions/predictions.jsonl")
    summaries = []
    with tempfile.TemporaryDirectory(prefix="llmca-replay-") as td:
        temp = Path(td)
        regenerated = temp / "generated.jsonl"
        count = generate_to_file(load_config(root / "configs/benchmark.yaml"), 42, regenerated)
        matches = file_hash(regenerated) == file_hash(actual_dataset)
        if not matches:
            raise ValueError("seeded generator no longer matches the frozen run input")
        for seed in (1, 97):
            folder = temp / f"hashseed-{seed}"
            folder.mkdir()
            config = load_config(root / "configs/experiment_paper_v1.yaml")
            config["dataset"]["input"] = str(regenerated)
            config_path = folder / "config.json"
            config_path.write_text(json.dumps(config))
            print(f"Replaying the full offline matrix with PYTHONHASHSEED={seed}", flush=True)
            with (folder / "run.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "llmca.cli", "run", "--config", str(config_path), "--seed", "42"],
                               cwd=folder, env={**os.environ, "PYTHONHASHSEED": str(seed)},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            path = folder / "runs/paper_v1/predictions/predictions.jsonl"
            reproduced = normalized(path)
            equal = reference == reproduced
            if not equal:
                raise ValueError(f"predictions differ with hash seed {seed}")
            summaries.append({"pythonhashseed": seed, "predictions": len(reproduced),
                              "normalized_predictions_match": equal,
                              "prediction_jsonl_bytes_match": file_hash(path) == file_hash(original / "predictions/predictions.jsonl")})
        baseline = {s["scenario_id"]: Scenario.model_validate(s).model_dump(mode="json") for s in read_jsonl(root / "data/scenarios/base.jsonl")}
        regenerated_rows = {s["scenario_id"]: Scenario.model_validate(s).model_dump(mode="json") for s in read_jsonl(regenerated)}
        base_semantics_match = baseline == regenerated_rows
    summary = {"generated_records": count, "seed": 42, "regenerated_dataset_hash_matches": matches,
               "dataset_hash": file_hash(actual_dataset), "legacy_annotation_source_semantics_match": base_semantics_match,
               "replays": summaries, "scope": "deterministic offline engine predictions; excludes timings, platform metadata and model re-generation"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(check(Path(args.out)), indent=2))
