"""Typed records for blinded review and explicit adjudication."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import Observation, Target

ReviewerType = Literal["human", "software_fixture", "automated_model"]
Ambiguity = Literal["low", "medium", "high"]


class AutomationProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str
    model_version: str | None = None
    method: str
    same_session_passes: bool
    generator_rules_visible_to_session: bool
    input_hash: str


class ReviewTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["annotation-1.0"] = "annotation-1.0"
    task_id: str
    domain: str
    timestamp: str
    context: list[Observation]
    situation_labels: list[str]
    action_vocabulary: list[str]


class ReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["annotation-1.0"] = "annotation-1.0"
    task_id: str
    reviewer_id: str
    reviewer_type: ReviewerType = "human"
    status: Literal["pending", "completed"] = "pending"
    target: Target | None = None
    ambiguity_level: Ambiguity | None = None
    notes: str = ""
    automation: AutomationProvenance | None = None

    @model_validator(mode="after")
    def completed_has_judgment(self) -> ReviewRecord:
        if self.status == "completed" and (self.target is None or self.ambiguity_level is None):
            raise ValueError("completed reviews require a target and ambiguity_level")
        if self.reviewer_type == "automated_model" and self.automation is None:
            raise ValueError("automated model reviews require automation provenance")
        if self.reviewer_type == "human" and self.automation is not None:
            raise ValueError("automated provenance cannot be labeled as human review")
        return self


class AdjudicationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["annotation-1.0"] = "annotation-1.0"
    task_id: str
    adjudicator_id: str = ""
    adjudicator_type: ReviewerType = "human"
    status: Literal["pending", "completed"] = "pending"
    target: Target | None = None
    ambiguity_level: Ambiguity | None = None
    rationale: str = ""
    automation: AutomationProvenance | None = None

    @model_validator(mode="after")
    def completed_has_decision(self) -> AdjudicationDecision:
        if self.status == "completed" and (
            not self.adjudicator_id.strip() or self.target is None
            or self.ambiguity_level is None or not self.rationale.strip()
        ):
            raise ValueError("completed decisions require adjudicator_id, target, ambiguity and rationale")
        if self.adjudicator_type == "automated_model" and self.automation is None:
            raise ValueError("automated model decisions require automation provenance")
        if self.adjudicator_type == "human" and self.automation is not None:
            raise ValueError("automated provenance cannot be labeled as human adjudication")
        return self


class ReviewPackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["annotation-1.0"] = "annotation-1.0"
    package_id: str
    source_hash: str
    source_path: str
    seed: int
    split: Literal["train", "validation", "test"]
    per_domain: int = Field(ge=1)
    reviewers: list[str]
    task_to_scenario: dict[str, str]
    tasks_hash: str
