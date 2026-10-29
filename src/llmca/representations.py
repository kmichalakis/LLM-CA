"""Target-hidden context representations for reasoners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from .models import ScenarioView

RepresentationName = Literal["natural_language", "key_value", "json", "semantic_json"]


@dataclass(frozen=True)
class Representation:
    name: RepresentationName
    content: str
    observation_keys: frozenset[str]


def build_representation(view: ScenarioView, name: str) -> Representation:
    keys = frozenset(obs.key for obs in view.context)
    if name == "natural_language":
        lines = [
            f"Scenario {view.scenario_id} is from domain {view.domain}.",
            "Observations:",
        ]
        for obs in view.context:
            value = "missing" if obs.is_missing else obs.value
            unit = f" {obs.unit}" if obs.unit else ""
            lines.append(
                f"- {obs.key}: {value}{unit}; source={obs.source}; "
                f"confidence={obs.confidence}; time={obs.timestamp.isoformat()}; "
                f"meaning={obs.semantic_definition}"
            )
        return Representation("natural_language", "\n".join(lines), keys)
    if name == "key_value":
        content = "\n".join(
            f"{obs.key}={None if obs.is_missing else obs.value} | unit={obs.unit} | "
            f"time={obs.timestamp.isoformat()} | source={obs.source} | meaning={obs.semantic_definition}"
            for obs in view.context
        )
        return Representation("key_value", content, keys)
    if name == "json":
        payload = {
            "scenario_id": view.scenario_id,
            "domain": view.domain,
            "context": [
                {
                    "key": obs.key,
                    "value": None if obs.is_missing else obs.value,
                    "unit": obs.unit,
                    "timestamp": obs.timestamp.isoformat(),
                    "source": obs.source,
                    "confidence": obs.confidence,
                    "is_missing": obs.is_missing,
                }
                for obs in view.context
            ],
        }
        return Representation("json", json.dumps(payload, sort_keys=True, ensure_ascii=False), keys)
    if name == "semantic_json":
        payload = {
            "scenario_id": view.scenario_id,
            "domain": view.domain,
            "context": [obs.model_dump(mode="json") for obs in view.context],
            "field_definitions": {
                "confidence": "0 to 1 quality score, null when unavailable",
                "is_missing": "true means value is intentionally unavailable",
                "semantic_definition": "domain-neutral meaning of the observation key",
            },
        }
        return Representation("semantic_json", json.dumps(payload, sort_keys=True, ensure_ascii=False), keys)
    raise ValueError(f"unknown representation {name!r}")


def verify_content_equivalence(view: ScenarioView, names: list[str]) -> None:
    expected = frozenset(obs.key for obs in view.context)
    for name in names:
        rep = build_representation(view, name)
        if rep.observation_keys != expected:
            raise ValueError(f"representation {name} lost observation keys")
