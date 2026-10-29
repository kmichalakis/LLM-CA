"""Versioned domain ontologies, variable dictionaries, and rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VariableSpec:
    key: str
    value_type: str
    unit: str | None
    definition: str
    values: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class DomainSpec:
    name: str
    version: str
    labels: tuple[str, ...]
    actions: tuple[str, ...]
    variables: tuple[VariableSpec, ...]
    examples: tuple[str, ...]


DOMAINS: dict[str, DomainSpec] = {
    "smart_home": DomainSpec(
        name="smart_home",
        version="1.0",
        labels=("resting", "cooking", "sleeping", "absent", "uncertain"),
        actions=("adjust_lighting", "adjust_thermostat", "notify", "no_action"),
        variables=(
            VariableSpec("occupancy", "scalar", "persons", "number of detected occupants", minimum=0, maximum=6),
            VariableSpec("motion_level", "categorical", None, "recent indoor motion level", ("none", "low", "medium", "high")),
            VariableSpec("stove_on", "bool", None, "whether cooking appliance is active"),
            VariableSpec("ambient_lux", "scalar", "lux", "ambient illuminance", minimum=0, maximum=1200),
            VariableSpec("hour_of_day", "scalar", "hour", "local hour on a 24 hour clock", minimum=0, maximum=23),
            VariableSpec("indoor_temp_c", "scalar", "celsius", "indoor air temperature", minimum=10, maximum=35),
        ),
        examples=tuple(f"smart_home example {i}" for i in range(1, 11)),
    ),
    "smart_learning": DomainSpec(
        name="smart_learning",
        version="1.0",
        labels=("engaged", "needs_support", "completed", "uncertain"),
        actions=("offer_hint", "change_difficulty", "prompt_break", "no_action"),
        variables=(
            VariableSpec("task_progress", "scalar", "percent", "synthetic learner task completion", minimum=0, maximum=100),
            VariableSpec("errors_recent", "scalar", "count", "recent incorrect attempts", minimum=0, maximum=8),
            VariableSpec("idle_minutes", "scalar", "minutes", "time since last interaction", minimum=0, maximum=45),
            VariableSpec("classroom_noise_db", "scalar", "dB", "ambient classroom noise proxy", minimum=30, maximum=90),
            VariableSpec("hint_used", "bool", None, "whether a hint has already been used"),
        ),
        examples=tuple(f"smart_learning example {i}" for i in range(1, 11)),
    ),
    "smart_tourism": DomainSpec(
        name="smart_tourism",
        version="1.0",
        labels=("exploring", "time_constrained", "accessibility_conflict", "uncertain"),
        actions=("suggest_alternative", "change_route", "provide_information", "no_action"),
        variables=(
            VariableSpec("minutes_until_booking", "scalar", "minutes", "time before scheduled booking", minimum=0, maximum=240),
            VariableSpec("crowd_level", "categorical", None, "crowd proxy at candidate location", ("low", "medium", "high")),
            VariableSpec("accessibility_required", "bool", None, "traveler requires accessible route"),
            VariableSpec("route_has_stairs", "bool", None, "candidate route contains stairs"),
            VariableSpec("weather", "categorical", None, "local weather condition", ("clear", "rain", "heat", "cold")),
        ),
        examples=tuple(f"smart_tourism example {i}" for i in range(1, 11)),
    ),
    "assisted_living": DomainSpec(
        name="assisted_living",
        version="1.0",
        labels=("routine_activity", "possible_anomaly", "insufficient_evidence"),
        actions=("ask_for_confirmation", "notify_designated_contact_simulated", "no_action"),
        variables=(
            VariableSpec("room", "categorical", None, "room with recent non-clinical activity", ("kitchen", "living_room", "bedroom", "bathroom")),
            VariableSpec("routine_match", "bool", None, "activity matches expected non-clinical routine"),
            VariableSpec("inactivity_minutes", "scalar", "minutes", "time since last benign activity signal", minimum=0, maximum=240),
            VariableSpec("sensor_confidence", "scalar", "probability", "aggregate confidence from sensors", minimum=0, maximum=1),
            VariableSpec("door_open_minutes", "scalar", "minutes", "minutes an exterior door has been open", minimum=0, maximum=90),
        ),
        examples=tuple(f"assisted_living example {i}" for i in range(1, 11)),
    ),
}


def domain_spec(name: str) -> DomainSpec:
    try:
        return DOMAINS[name]
    except KeyError as exc:
        raise ValueError(f"unknown domain {name!r}") from exc


def safe_action_for_label(domain: str, label: str) -> str:
    if domain == "smart_home":
        return {
            "resting": "adjust_lighting",
            "cooking": "no_action",
            "sleeping": "adjust_thermostat",
            "absent": "no_action",
            "uncertain": "no_action",
        }[label]
    if domain == "smart_learning":
        return {
            "engaged": "no_action",
            "needs_support": "offer_hint",
            "completed": "change_difficulty",
            "uncertain": "prompt_break",
        }[label]
    if domain == "smart_tourism":
        return {
            "exploring": "provide_information",
            "time_constrained": "change_route",
            "accessibility_conflict": "suggest_alternative",
            "uncertain": "no_action",
        }[label]
    if domain == "assisted_living":
        return {
            "routine_activity": "no_action",
            "possible_anomaly": "ask_for_confirmation",
            "insufficient_evidence": "no_action",
        }[label]
    raise ValueError(f"unknown domain {domain!r}")


def infer_rule_label(domain: str, values: dict[str, Any]) -> tuple[str, list[str], str]:
    """Constraint-based target function used for synthetic labels and rule baseline."""

    missing = [var.key for var in domain_spec(domain).variables if values.get(var.key) is None]
    if missing:
        label = "insufficient_evidence" if domain == "assisted_living" else "uncertain"
        return label, missing, "conservative rule abstention for missing observations"

    if domain == "smart_home":
        if values.get("occupancy", 0) == 0:
            return "absent", ["occupancy", "motion_level"], "no detected occupants"
        if values.get("stove_on") is True and values.get("motion_level") in {"medium", "high"}:
            return "cooking", ["stove_on", "motion_level", "occupancy"], "active stove with nearby motion"
        if values.get("hour_of_day", 12) >= 22 and values.get("ambient_lux", 500) < 80:
            return "sleeping", ["hour_of_day", "ambient_lux", "occupancy"], "late hour and low light"
        if values.get("motion_level") in {"low", "medium"}:
            return "resting", ["motion_level", "occupancy", "ambient_lux"], "occupied with moderate motion"
        return "uncertain", ["occupancy", "motion_level"], "insufficient pattern match"

    if domain == "smart_learning":
        if values.get("task_progress", 0) >= 95:
            return "completed", ["task_progress"], "task nearly complete"
        if values.get("errors_recent", 0) >= 4 or values.get("idle_minutes", 0) >= 15:
            return "needs_support", ["errors_recent", "idle_minutes", "hint_used"], "repeated errors or inactivity"
        if values.get("task_progress", 0) >= 35 and values.get("idle_minutes", 99) < 10:
            return "engaged", ["task_progress", "idle_minutes"], "progress with recent activity"
        return "uncertain", ["task_progress", "idle_minutes"], "limited evidence"

    if domain == "smart_tourism":
        if values.get("accessibility_required") is True and values.get("route_has_stairs") is True:
            return "accessibility_conflict", ["accessibility_required", "route_has_stairs"], "route violates accessibility constraint"
        if values.get("minutes_until_booking", 240) <= 35 or values.get("weather") in {"rain", "heat"}:
            return "time_constrained", ["minutes_until_booking", "weather", "crowd_level"], "schedule or weather constraint"
        if values.get("crowd_level") in {"low", "medium"}:
            return "exploring", ["crowd_level", "weather"], "conditions support exploration"
        return "uncertain", ["crowd_level", "minutes_until_booking"], "limited evidence"

    if domain == "assisted_living":
        if values.get("sensor_confidence", 1.0) < 0.45:
            return "insufficient_evidence", ["sensor_confidence"], "low sensor confidence"
        if values.get("routine_match") is False and (
            values.get("inactivity_minutes", 0) >= 90 or values.get("door_open_minutes", 0) >= 25
        ):
            return "possible_anomaly", ["routine_match", "inactivity_minutes", "door_open_minutes"], "non-routine prolonged signal"
        return "routine_activity", ["room", "routine_match", "sensor_confidence"], "matches routine activity"

    raise ValueError(f"unknown domain {domain!r}")
