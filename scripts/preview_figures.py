"""Render the six paper figures for visual inspection before freezing artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from llmca.figures import annotation_audit, architecture, dataset_composition, robustness, workflow
from llmca.io import read_jsonl
from llmca.paper_pack import csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/paper_v1")
    parser.add_argument("--out", default="validation/figure_preview")
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = read_jsonl(run / "scenarios.jsonl")
    predictions = read_jsonl(run / "predictions/predictions.jsonl")
    architecture(out)
    workflow(out, sum(s["perturbation"]["perturbation_type"] == "complete" for s in scenarios), len(scenarios), len(predictions))
    dataset_composition(out, scenarios)
    robustness(run, csv_rows(run / "metrics/metrics.csv"))
    annotation_audit(out, Path("annotations/pilot/automatic/imported/review_summary.json"))
    for name in ("clean_action_acceptability", "robustness"):
        for extension in ("png", "svg"):
            shutil.copyfile(run / "figures" / f"{name}.{extension}", out / f"{name}.{extension}")
    print(out.resolve())


if __name__ == "__main__":
    main()
