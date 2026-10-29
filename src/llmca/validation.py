"""Dataset validation and schema export."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .domains import DOMAINS
from .io import read_jsonl
from .models import Scenario


def validate_scenarios(path: str | Path, strict: bool = False) -> tuple[list[Scenario], list[str]]:
    errors: list[str] = []
    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    family_splits: dict[str, str] = {}
    domain_counts: dict[str, int] = defaultdict(int)

    for line_no, row in enumerate(read_jsonl(path), 1):
        try:
            scenario = Scenario.model_validate(row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"line {line_no}: {exc}")
            continue
        if scenario.scenario_id in seen_ids:
            errors.append(f"line {line_no}: duplicate scenario_id {scenario.scenario_id}")
        seen_ids.add(scenario.scenario_id)
        if scenario.domain not in DOMAINS:
            errors.append(f"line {line_no}: unknown domain {scenario.domain}")
        else:
            spec_by_key = {var.key: var for var in DOMAINS[scenario.domain].variables}
            for obs in scenario.context:
                if obs.key.startswith("irrelevant_") or obs.key.endswith("_conflict"):
                    continue
                var = spec_by_key.get(obs.key)
                if not var:
                    errors.append(f"line {line_no}: unknown observation key {obs.key!r} for {scenario.domain}")
                    continue
                if obs.unit != var.unit:
                    errors.append(
                        f"line {line_no}: unit mismatch for {obs.key}: expected {var.unit!r}, got {obs.unit!r}"
                    )
                if not obs.is_missing and isinstance(obs.value, (int, float)):
                    if var.minimum is not None and float(obs.value) < var.minimum:
                        errors.append(f"line {line_no}: {obs.key} below valid range")
                    if var.maximum is not None and float(obs.value) > var.maximum:
                        errors.append(f"line {line_no}: {obs.key} above valid range")
        split = family_splits.setdefault(scenario.scenario_family_id, scenario.split.value)
        if split != scenario.split.value:
            errors.append(
                f"line {line_no}: family {scenario.scenario_family_id} leaks across splits "
                f"({split}, {scenario.split.value})"
            )
        context_keys = {obs.key for obs in scenario.context}
        if not set(scenario.target.relevant_observation_keys).issubset(context_keys):
            errors.append(f"line {line_no}: relevant keys missing from context")
        domain_counts[scenario.domain] += 1
        scenarios.append(scenario)

    if strict:
        missing_domains = set(DOMAINS) - set(domain_counts)
        if missing_domains:
            errors.append(f"missing domains: {sorted(missing_domains)}")
    return scenarios, errors


def export_schema(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(Scenario.model_json_schema(), indent=2, sort_keys=True), encoding="utf-8")
