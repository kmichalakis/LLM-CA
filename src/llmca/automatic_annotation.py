"""Validate and materialize recorded model judgments into review submissions.

This module does not infer labels or call a model. Its input is an actual model's
recorded judgment batch; it never reads generated targets from the source dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .annotation import _load_package, _new_directory, validate_judgment
from .annotation_models import Ambiguity, AutomationProvenance, ReviewRecord
from .config import file_hash, stable_hash
from .io import write_jsonl
from .models import Target

JudgmentRow = tuple[str, str, list[str], list[str], list[str], list[str], bool, Ambiguity, str]


class ModelPass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    judgments: list[JudgmentRow]


class ModelJudgmentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    package_id: str
    input_tasks_hash: str
    engine: str
    model_version: str | None
    same_session_passes: bool
    generator_rules_visible_to_session: bool
    passes: dict[str, ModelPass]


def materialize_model_reviews(package: str | Path, judgments: str | Path, out: str | Path) -> Path:
    manifest, tasks = _load_package(package)
    batch = ModelJudgmentBatch.model_validate_json(Path(judgments).read_text(encoding="utf-8"))
    if batch.schema_version != "model-judgments-1.0":
        raise ValueError("unsupported model judgment format")
    if batch.package_id != manifest.package_id or batch.input_tasks_hash != manifest.tasks_hash:
        raise ValueError("model judgments refer to a different package or task snapshot")
    if set(batch.passes) != set(manifest.reviewers):
        raise ValueError("model judgment passes must match all assigned reviewer IDs")
    all_reviews = {}
    for reviewer, model_pass in batch.passes.items():
        records = []
        seen = set()
        for task_id, label, acceptable, keys, allowed, preferred, abstain, ambiguity, rationale in model_pass.judgments:
            if task_id not in tasks or task_id in seen:
                raise ValueError("unknown or duplicate model judgment task")
            seen.add(task_id)
            task = tasks[task_id]
            target = Target(
                situation_label=label, acceptable_situation_labels=acceptable,
                relevant_observation_keys=keys, allowed_actions=allowed, preferred_actions=preferred,
                forbidden_actions=[action for action in task.action_vocabulary if action not in allowed],
                abstain_acceptable=abstain, rationale_reference=rationale,
            )
            validate_judgment(task, target)
            records.append(ReviewRecord(
                task_id=task_id, reviewer_id=reviewer, reviewer_type="automated_model",
                status="completed", target=target, ambiguity_level=ambiguity,
                notes="Model-generated judgment. " + (
                    "Two passes share one Codex session and are correlated. "
                    if batch.same_session_passes and batch.engine == "Codex conversation model"
                    else ("The passes share a model session and are correlated. " if batch.same_session_passes
                          else "Session grouping is declared in the supplied provenance. ")
                ) + ("The session had access to generator rules. " if batch.generator_rules_visible_to_session
                     else "No prior generator-rule exposure is declared in the supplied provenance. ")
                + "No human independence is claimed.",
                automation=AutomationProvenance(
                    engine=batch.engine, model_version=batch.model_version, method=model_pass.method,
                    same_session_passes=batch.same_session_passes,
                    generator_rules_visible_to_session=batch.generator_rules_visible_to_session,
                    input_hash=stable_hash(task.model_dump(mode="json")),
                ),
            ))
        if seen != set(tasks):
            raise ValueError(f"{reviewer}: missing model judgments")
        all_reviews[reviewer] = sorted(records, key=lambda r: r.task_id)
    root = _new_directory(out)
    for reviewer, records in all_reviews.items():
        folder = root / reviewer
        folder.mkdir()
        write_jsonl(folder / "reviews.jsonl", records)
        write_jsonl(folder / "tasks.jsonl", [tasks[k] for k in sorted(tasks)])
        (folder / "review.schema.json").write_text(
            json.dumps(ReviewRecord.model_json_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    (root / "generation_manifest.json").write_text(json.dumps({
        "package_id": manifest.package_id, "judgments_hash": file_hash(judgments),
        "engine": batch.engine, "model_version": batch.model_version,
        "same_session_passes": batch.same_session_passes,
        "generator_rules_visible_to_session": batch.generator_rules_visible_to_session,
        "records": sum(len(rows) for rows in all_reviews.values()),
        "reviewer_type": "automated_model", "source_targets_read_by_materializer": False,
        "output_hashes": {reviewer: file_hash(root / reviewer / "reviews.jsonl") for reviewer in all_reviews},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize recorded model annotation judgments")
    parser.add_argument("--package", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(materialize_model_reviews(args.package, args.judgments, args.out))


if __name__ == "__main__":
    main()
