"""Seeded synthetic scenario generation and perturbations."""

from __future__ import annotations

import copy
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .domains import DOMAINS, VariableSpec, domain_spec, infer_rule_label, safe_action_for_label
from .io import write_jsonl
from .models import Annotation, Observation, PerturbationInfo, Scenario, Target, ValueType


def _split_for_index(index: int, total: int, splits: dict[str, float]) -> str:
    train_cut = int(total * splits.get("train", 0.6))
    validation_cut = train_cut + int(total * splits.get("validation", 0.2))
    if index < train_cut:
        return "train"
    if index < validation_cut:
        return "validation"
    return "test"


def _sample_value(rng: random.Random, spec: VariableSpec, label_hint: str | None = None) -> Any:
    if spec.values:
        return rng.choice(spec.values)
    if spec.value_type == "bool":
        return rng.choice([True, False])
    if spec.minimum is not None and spec.maximum is not None:
        if spec.unit in {"persons", "count", "hour", "minutes"}:
            return rng.randint(int(spec.minimum), int(spec.maximum))
        return round(rng.uniform(float(spec.minimum), float(spec.maximum)), 2)
    return label_hint or "unknown"


def _domain_values(domain: str, rng: random.Random, label_index: int) -> dict[str, Any]:
    spec = domain_spec(domain)
    values = {var.key: _sample_value(rng, var) for var in spec.variables}

    labels = spec.labels
    desired = labels[label_index % len(labels)]
    if domain == "smart_home":
        if desired == "absent":
            values.update(occupancy=0, motion_level="none", stove_on=False)
        elif desired == "cooking":
            values.update(occupancy=max(1, values["occupancy"]), motion_level="high", stove_on=True)
        elif desired == "sleeping":
            values.update(occupancy=max(1, values["occupancy"]), hour_of_day=23, ambient_lux=25, stove_on=False)
        elif desired == "resting":
            values.update(occupancy=max(1, values["occupancy"]), motion_level="low", stove_on=False)
        else:
            values.update(occupancy=max(1, values["occupancy"]), motion_level="none", ambient_lux=500, stove_on=False)
    elif domain == "smart_learning":
        if desired == "completed":
            values.update(task_progress=98, errors_recent=0, idle_minutes=1)
        elif desired == "needs_support":
            values.update(task_progress=35, errors_recent=5, idle_minutes=18)
        elif desired == "engaged":
            values.update(task_progress=60, errors_recent=1, idle_minutes=3)
        else:
            values.update(task_progress=10, errors_recent=1, idle_minutes=12)
    elif domain == "smart_tourism":
        if desired == "accessibility_conflict":
            values.update(accessibility_required=True, route_has_stairs=True)
        elif desired == "time_constrained":
            values.update(minutes_until_booking=20, weather="rain", route_has_stairs=False)
        elif desired == "exploring":
            values.update(minutes_until_booking=140, crowd_level="low", weather="clear", route_has_stairs=False)
        else:
            values.update(minutes_until_booking=80, crowd_level="high", weather="cold", route_has_stairs=False)
    elif domain == "assisted_living":
        if desired == "possible_anomaly":
            values.update(routine_match=False, inactivity_minutes=125, sensor_confidence=0.9)
        elif desired == "insufficient_evidence":
            values.update(sensor_confidence=0.25)
        else:
            values.update(routine_match=True, inactivity_minutes=12, sensor_confidence=0.92)
    return values


def _observation(domain: str, spec: VariableSpec, value: Any, timestamp: datetime) -> Observation:
    value_type = spec.value_type
    if value_type == "bool":
        vt = ValueType.bool
    elif value_type == "categorical":
        vt = ValueType.categorical
    elif value_type == "list":
        vt = ValueType.list
    else:
        vt = ValueType.scalar
    return Observation(
        key=spec.key,
        value=value,
        value_type=vt,
        unit=spec.unit,
        timestamp=timestamp,
        source=f"{domain}_simulator",
        confidence=1.0,
        is_missing=False,
        semantic_definition=spec.definition,
    )


def _make_scenario(domain: str, index: int, total: int, seed: int, splits: dict[str, float]) -> Scenario:
    rng = random.Random(seed * 100000 + index)
    spec = domain_spec(domain)
    family_id = f"{domain}_f{index:05d}"
    scenario_id = f"{domain}_{index:05d}"
    timestamp = datetime(2026, 1, 1, 12, tzinfo=UTC) + timedelta(minutes=index)
    values = _domain_values(domain, rng, index)
    label, relevant, rationale = infer_rule_label(domain, values)
    preferred = safe_action_for_label(domain, label)
    allowed = sorted(set([preferred, "no_action"]) & set(spec.actions))
    if preferred not in allowed:
        allowed.append(preferred)
    forbidden = [action for action in spec.actions if action not in allowed]
    return Scenario(
        schema_version="1.0",
        scenario_id=scenario_id,
        scenario_family_id=family_id,
        domain=domain,
        origin_type="synthetic",
        source_reference=f"llmca.synthetic.{domain}.v1",
        license="MIT synthetic non-identifying",
        split=_split_for_index(index, total, splits),
        seed=seed,
        timestamp=timestamp,
        context=[_observation(domain, var, values[var.key], timestamp) for var in spec.variables],
        target=Target(
            situation_label=label,
            acceptable_situation_labels=[label],
            relevant_observation_keys=relevant,
            allowed_actions=allowed,
            preferred_actions=[preferred],
            forbidden_actions=forbidden,
            abstain_acceptable=label in {"uncertain", "insufficient_evidence"},
            rationale_reference=rationale,
        ),
        annotation=Annotation(
            annotator_type="rule_generator",
            annotation_protocol_version="synthetic-v1",
            adjudication_status="synthetic_rule",
            ambiguity_level="medium" if label in {"uncertain", "insufficient_evidence"} else "low",
            provenance="deterministic domain state machine; not human-adjudicated",
            independent_labels=[],
        ),
        tags=["synthetic", "base", f"ontology:{spec.version}"],
        perturbation=PerturbationInfo(
            perturbation_type="complete",
            severity=0,
            changed_keys=[],
            parent_id=None,
            repeat_index=0,
            seed=seed,
            target_invariant=True,
        ),
    )


def generate_base_scenarios(
    per_domain: int = 150,
    seed: int = 42,
    splits: dict[str, float] | None = None,
) -> list[Scenario]:
    splits = splits or {"train": 0.6, "validation": 0.2, "test": 0.2}
    scenarios: list[Scenario] = []
    for domain in DOMAINS:
        for index in range(per_domain):
            scenarios.append(_make_scenario(domain, index, per_domain, seed, splits))
    return scenarios


def _clone_with_id(parent: Scenario, suffix: str, perturbation: PerturbationInfo) -> Scenario:
    data = parent.model_dump(mode="python")
    data["scenario_id"] = f"{parent.scenario_id}_{suffix}"
    data["perturbation"] = perturbation.model_dump(mode="python")
    data["tags"] = sorted(set(parent.tags + [f"perturbation:{perturbation.perturbation_type}"]))
    return Scenario.model_validate(data)


def perturb_scenario(parent: Scenario, kind: str, severity: float | int | str, repeat: int, seed: int) -> Scenario:
    rng = random.Random(seed)
    data = parent.model_dump(mode="python")
    context = [Observation.model_validate(obs) for obs in data["context"]]
    changed: list[str] = []
    invariant = True

    if kind == "missingness":
        candidates = [obs for obs in context if obs.key not in parent.target.relevant_observation_keys]
        if not candidates:
            candidates = context[:]
        count = max(1, int(len(context) * float(severity)))
        for obs in rng.sample(candidates, min(count, len(candidates))):
            obs.value = None
            obs.is_missing = True
            changed.append(obs.key)
    elif kind == "irrelevance":
        for i in range(int(severity)):
            key = f"irrelevant_{repeat}_{i}"
            context.append(
                Observation(
                    key=key,
                    value=rng.choice(["decorative", "background", rng.randint(1, 100)]),
                    value_type=ValueType.text,
                    unit=None,
                    timestamp=parent.timestamp,
                    source="irrelevance_injector",
                    confidence=0.8,
                    is_missing=False,
                    semantic_definition="plausible but target-irrelevant distractor",
                )
            )
            changed.append(key)
    elif kind == "noise":
        specs = {var.key: var for var in domain_spec(parent.domain).variables}
        numeric = [obs for obs in context if obs.value_type == ValueType.scalar and not obs.is_missing]
        for obs in numeric[: max(1, len(numeric) // 2)]:
            if isinstance(obs.value, (int, float)):
                delta = float(obs.value) * float(severity)
                value = float(obs.value) + rng.uniform(-delta, delta)
                spec = specs.get(obs.key)
                if spec and spec.minimum is not None:
                    value = max(float(spec.minimum), value)
                if spec and spec.maximum is not None:
                    value = min(float(spec.maximum), value)
                obs.value = round(value, 2)
                changed.append(obs.key)
    elif kind == "conflict":
        base = rng.choice(context)
        conflict = base.model_copy(deep=True)
        conflict.key = f"{base.key}_conflict"
        conflict.source = "conflicting_source"
        conflict.confidence = 0.55
        if isinstance(base.value, bool):
            conflict.value = not base.value
        elif isinstance(base.value, (int, float)):
            conflict.value = round(float(base.value) * 1.8 + 1, 2)
        else:
            conflict.value = f"not_{base.value}"
        context.append(conflict)
        changed.append(conflict.key)
    elif kind == "stale":
        obs = rng.choice(context)
        obs.timestamp = obs.timestamp - timedelta(days=14)
        obs.confidence = 0.4
        changed.append(obs.key)
    elif kind == "counterfactual":
        invariant = False
        if parent.domain == "smart_home":
            for obs in context:
                if obs.key == "occupancy":
                    obs.value = 0
                    changed.append(obs.key)
                if obs.key == "motion_level":
                    obs.value = "none"
                    changed.append(obs.key)
        elif parent.domain == "smart_learning":
            for obs in context:
                if obs.key == "errors_recent":
                    obs.value = 6
                    changed.append(obs.key)
                if obs.key == "idle_minutes":
                    obs.value = 20
                    changed.append(obs.key)
        elif parent.domain == "smart_tourism":
            for obs in context:
                if obs.key == "accessibility_required":
                    obs.value = True
                    changed.append(obs.key)
                if obs.key == "route_has_stairs":
                    obs.value = True
                    changed.append(obs.key)
        elif parent.domain == "assisted_living":
            for obs in context:
                if obs.key == "routine_match":
                    obs.value = False
                    changed.append(obs.key)
                if obs.key == "inactivity_minutes":
                    obs.value = 140
                    changed.append(obs.key)
        values = {obs.key: obs.value for obs in context}
        label, relevant, rationale = infer_rule_label(parent.domain, values)
        preferred = safe_action_for_label(parent.domain, label)
        spec = domain_spec(parent.domain)
        allowed = sorted(set([preferred, "no_action"]) & set(spec.actions))
        if preferred not in allowed:
            allowed.append(preferred)
        data["target"] = Target(
            situation_label=label,
            acceptable_situation_labels=[label],
            relevant_observation_keys=relevant,
            allowed_actions=allowed,
            preferred_actions=[preferred],
            forbidden_actions=[action for action in spec.actions if action not in allowed],
            abstain_acceptable=label in {"uncertain", "insufficient_evidence"},
            rationale_reference=f"counterfactual relabel: {rationale}",
        ).model_dump(mode="python")
    else:
        raise ValueError(f"unknown perturbation {kind!r}")

    data["context"] = [obs.model_dump(mode="python") for obs in context]
    data["scenario_id"] = f"{parent.scenario_id}_{kind}_{repeat}_{str(severity).replace('.', 'p')}"
    data["tags"] = sorted(set(parent.tags + [f"perturbation:{kind}"]))
    data["perturbation"] = PerturbationInfo(
        perturbation_type=kind,  # type: ignore[arg-type]
        severity=severity,
        changed_keys=changed,
        parent_id=parent.scenario_id,
        repeat_index=repeat,
        seed=seed,
        target_invariant=invariant,
    ).model_dump(mode="python")
    return Scenario.model_validate(data)


def add_default_perturbations(base: list[Scenario], config: dict[str, Any], seed: int) -> list[Scenario]:
    scenarios = copy.copy(base)
    generation = config.get("generation", {})
    repeats = int(generation.get("perturbation_repeats", 1))
    for parent_index, parent in enumerate(base):
        if parent.split.value != "test":
            continue
        for repeat in range(repeats):
            pseed = seed * 1000000 + parent_index * 100 + repeat
            for severity in generation.get("missingness_severities", [0.1]):
                scenarios.append(perturb_scenario(parent, "missingness", severity, repeat, pseed + 1))
            for count in generation.get("irrelevance_counts", [5]):
                scenarios.append(perturb_scenario(parent, "irrelevance", count, repeat, pseed + 2))
            for severity in generation.get("noise_severities", [0.1]):
                scenarios.append(perturb_scenario(parent, "noise", severity, repeat, pseed + 3))
            if generation.get("include_conflict", True):
                scenarios.append(perturb_scenario(parent, "conflict", "single", repeat, pseed + 4))
            if generation.get("include_stale", True):
                scenarios.append(perturb_scenario(parent, "stale", "single", repeat, pseed + 5))
            if generation.get("include_counterfactual", True):
                scenarios.append(perturb_scenario(parent, "counterfactual", "single", repeat, pseed + 6))
    return scenarios


def generate_to_file(config: dict[str, Any], seed: int, out: str | Path) -> int:
    dataset = config.get("dataset", {})
    base = generate_base_scenarios(
        per_domain=int(dataset.get("base_scenarios_per_domain", 150)),
        seed=seed,
        splits=dataset.get("splits"),
    )
    scenarios = add_default_perturbations(base, config, seed) if config.get("generation", {}).get("perturbations", True) else base
    return write_jsonl(out, scenarios)
