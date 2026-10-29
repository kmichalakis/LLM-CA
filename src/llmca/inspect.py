"""Run and dataset inspection helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .io import read_jsonl
from .models import Scenario


def inspect_dataset(path: str | Path, limit: int = 5) -> str:
    scenarios = [Scenario.model_validate(row) for row in read_jsonl(path)]
    by_domain = Counter(s.domain for s in scenarios)
    by_split = Counter(s.split.value for s in scenarios)
    by_perturbation = Counter(s.perturbation.perturbation_type for s in scenarios)
    lines = [
        f"Scenarios: {len(scenarios)}",
        f"Domains: {dict(by_domain)}",
        f"Splits: {dict(by_split)}",
        f"Perturbations: {dict(by_perturbation)}",
        "Examples:",
    ]
    for scenario in scenarios[:limit]:
        lines.append(
            f"- {scenario.scenario_id}: domain={scenario.domain}, split={scenario.split.value}, "
            f"label={scenario.target.situation_label}, action={scenario.target.preferred_actions[0]}"
        )
    return "\n".join(lines)
