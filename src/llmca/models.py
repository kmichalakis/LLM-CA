"""Canonical Pydantic models for LLM-CA scenarios and predictions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OriginType(str, Enum):
    synthetic = "synthetic"
    dataset_derived = "dataset_derived"
    real_observation = "real_observation"


class Split(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"


class ValueType(str, Enum):
    scalar = "scalar"
    categorical = "categorical"
    bool = "bool"
    list = "list"
    text = "text"


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    value: Any
    value_type: ValueType
    unit: str | None = None
    timestamp: datetime
    source: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_missing: bool
    semantic_definition: str = Field(min_length=1)

    @model_validator(mode="after")
    def missing_values_are_explicit(self) -> "Observation":
        if self.is_missing and self.value is not None:
            raise ValueError("missing observations must use null value")
        return self


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    situation_label: str
    acceptable_situation_labels: list[str]
    relevant_observation_keys: list[str]
    allowed_actions: list[str]
    preferred_actions: list[str]
    forbidden_actions: list[str]
    abstain_acceptable: bool
    rationale_reference: str

    @model_validator(mode="after")
    def target_sets_are_consistent(self) -> "Target":
        if self.situation_label not in self.acceptable_situation_labels:
            raise ValueError("situation_label must be acceptable")
        if not set(self.preferred_actions).issubset(set(self.allowed_actions)):
            raise ValueError("preferred_actions must be a subset of allowed_actions")
        if set(self.allowed_actions) & set(self.forbidden_actions):
            raise ValueError("allowed_actions and forbidden_actions overlap")
        return self


class AnnotationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer_id: str
    reviewer_type: Literal["human", "software_fixture", "automated_model"]
    task_id: str
    record_hash: str
    situation_label: str


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotator_type: Literal["rule_generator", "human", "mixed"]
    annotation_protocol_version: str
    adjudication_status: Literal["synthetic_rule", "pending", "adjudicated"]
    ambiguity_level: Literal["low", "medium", "high"]
    provenance: str
    independent_labels: list[str] = Field(default_factory=list)
    review_evidence: list[AnnotationEvidence] = Field(default_factory=list)
    adjudicator_id: str | None = None
    adjudicator_type: Literal["human", "software_fixture", "automated_model"] | None = None
    review_package_id: str | None = None


class PerturbationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perturbation_type: Literal[
        "complete",
        "missingness",
        "irrelevance",
        "noise",
        "conflict",
        "stale",
        "counterfactual",
    ]
    severity: float | int | str
    changed_keys: list[str]
    parent_id: str | None
    repeat_index: int
    seed: int
    target_invariant: bool


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    scenario_id: str
    scenario_family_id: str
    domain: str
    origin_type: OriginType
    source_reference: str
    license: str
    split: Split
    seed: int
    timestamp: datetime
    context: list[Observation]
    target: Target
    annotation: Annotation
    tags: list[str] = Field(default_factory=list)
    perturbation: PerturbationInfo = Field(
        default_factory=lambda: PerturbationInfo(
            perturbation_type="complete",
            severity=0,
            changed_keys=[],
            parent_id=None,
            repeat_index=0,
            seed=0,
            target_invariant=True,
        )
    )

    @field_validator("context")
    @classmethod
    def context_keys_are_unique(cls, value: list[Observation]) -> list[Observation]:
        keys = [obs.key for obs in value]
        if len(keys) != len(set(keys)):
            raise ValueError("observation keys must be unique per scenario")
        return value

    @model_validator(mode="after")
    def references_are_valid(self) -> "Scenario":
        keys = {obs.key for obs in self.context}
        missing = set(self.target.relevant_observation_keys) - keys
        if missing:
            raise ValueError(f"target references unknown observation keys: {sorted(missing)}")
        return self


class ScenarioView(BaseModel):
    """Target-hidden scenario view used by prompts and reasoners."""

    scenario_id: str
    scenario_family_id: str
    domain: str
    origin_type: OriginType
    split: Split
    timestamp: datetime
    context: list[Observation]
    tags: list[str]
    perturbation: PerturbationInfo


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    situation_label: str
    relevant_observation_keys: list[str]
    proposed_action: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    abstain: bool
    explanation: str
    cited_context_keys: list[str]


class RawReasonerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    model_name: str
    representation: str
    raw_response: str
    prediction: Prediction
    metadata: dict[str, Any]


def scenario_view(scenario: Scenario) -> ScenarioView:
    return ScenarioView(
        scenario_id=scenario.scenario_id,
        scenario_family_id=scenario.scenario_family_id,
        domain=scenario.domain,
        origin_type=scenario.origin_type,
        split=scenario.split,
        timestamp=scenario.timestamp,
        context=scenario.context,
        tags=scenario.tags,
        perturbation=scenario.perturbation,
    )
