"""Blinded annotation packages, independent review import and audited application."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from .annotation_models import (
    AdjudicationDecision, ReviewPackageManifest, ReviewRecord, ReviewTask,
)
from .config import file_hash, stable_hash
from .domains import DOMAINS
from .io import read_jsonl, write_jsonl
from .models import Annotation, AnnotationEvidence, Scenario, Target
from .validation import validate_scenarios

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _identity(value: str) -> str:
    if not _ID.fullmatch(value):
        raise ValueError("reviewer IDs must be 1-64 letters, digits, underscores or hyphens")
    return value


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _new_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=False)
    return target


def _load_scenarios(path: str | Path) -> list[Scenario]:
    scenarios, errors = validate_scenarios(path)
    if errors:
        raise ValueError("annotation dataset is invalid: " + "; ".join(errors[:5]))
    if not scenarios:
        raise ValueError("annotation dataset is empty")
    return scenarios


def validate_judgment(task: ReviewTask, target: Target) -> None:
    """Enforce ontology choices, evidence references and a full action partition."""
    labels = set(task.situation_labels)
    actions = set(task.action_vocabulary)
    if target.situation_label not in labels or not set(target.acceptable_situation_labels) <= labels:
        raise ValueError(f"{task.task_id}: unknown situation label")
    if not set(target.relevant_observation_keys) <= {obs.key for obs in task.context}:
        raise ValueError(f"{task.task_id}: unknown relevant observation key")
    if not target.allowed_actions or not target.preferred_actions:
        raise ValueError(f"{task.task_id}: allowed and preferred actions must be nonempty")
    if set(target.allowed_actions) | set(target.forbidden_actions) != actions:
        raise ValueError(f"{task.task_id}: classify every ontology action as allowed or forbidden")
    for field in ("acceptable_situation_labels", "relevant_observation_keys", "allowed_actions",
                  "preferred_actions", "forbidden_actions"):
        values = getattr(target, field)
        if len(values) != len(set(values)):
            raise ValueError(f"{task.task_id}: duplicate values in {field}")
    if not target.rationale_reference.strip():
        raise ValueError(f"{task.task_id}: rationale_reference must explain the judgment")


def export_review_package(
    source: str | Path, out: str | Path, reviewers: list[str],
    per_domain: int = 10, seed: int = 42, split: str = "validation",
) -> Path:
    if per_domain < 1 or split not in {"train", "validation", "test"}:
        raise ValueError("per_domain must be positive and split must be train, validation or test")
    if len(reviewers) < 2 or len(set(reviewers)) != len(reviewers):
        raise ValueError("provide at least two distinct independent reviewer IDs")
    for reviewer in reviewers:
        _identity(reviewer)
    scenarios = _load_scenarios(source)
    by_domain: dict[str, list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        if scenario.split.value == split and scenario.perturbation.perturbation_type == "complete":
            by_domain[scenario.domain].append(scenario)
    if not by_domain:
        raise ValueError(f"no clean cases in the {split} split")
    rng = random.Random(seed)
    selected = []
    for domain in sorted({s.domain for s in scenarios}):
        candidates = sorted(by_domain[domain], key=lambda s: s.scenario_id)
        if len(candidates) < per_domain:
            raise ValueError(f"{domain}: requested {per_domain} cases but only {len(candidates)} available")
        selected.extend(rng.sample(candidates, per_domain))
    source_hash = file_hash(source)
    tasks = []
    mapping = {}
    for scenario in selected:
        task_id = "task_" + stable_hash([source_hash, scenario.scenario_id])[:20]
        spec = DOMAINS[scenario.domain]
        tasks.append(ReviewTask(
            task_id=task_id, domain=scenario.domain, timestamp=scenario.timestamp.isoformat(),
            context=scenario.context, situation_labels=list(spec.labels), action_vocabulary=list(spec.actions),
        ))
        mapping[task_id] = scenario.scenario_id
    tasks.sort(key=lambda t: t.task_id)
    task_data = [t.model_dump(mode="json") for t in tasks]
    package_id = stable_hash([source_hash, task_data, mapping, sorted(reviewers), seed, split, per_domain])
    manifest = ReviewPackageManifest(
        package_id=package_id, source_hash=source_hash, source_path=str(Path(source).resolve()),
        seed=seed, split=split, per_domain=per_domain, reviewers=sorted(reviewers),
        task_to_scenario=mapping, tasks_hash=stable_hash(task_data),
    )
    root = _new_directory(out)
    (root / "private").mkdir()
    _json(root / "private" / "manifest.json", manifest.model_dump(mode="json"))
    write_jsonl(root / "private" / "tasks.jsonl", tasks)
    for reviewer in sorted(reviewers):
        folder = root / "reviewers" / reviewer
        folder.mkdir(parents=True)
        write_jsonl(folder / "tasks.jsonl", tasks)
        write_jsonl(folder / "reviews.jsonl", [ReviewRecord(task_id=t.task_id, reviewer_id=reviewer) for t in tasks])
        _json(folder / "review.schema.json", ReviewRecord.model_json_schema())
        (folder / "README.md").write_text(
            "# Independent review\n\n"
            "Review observations independently. Choose labels, acceptable actions and relevant keys "
            "from the supplied ontology. Classify every action as allowed or forbidden. "
            "Record ambiguity and an evidence-based rationale. Mark a record completed only after review. "
            "Do not consult generated labels or other reviewers. Use a pseudonymous reviewer ID. "
            "Ask the coordinator about observation provenance when needed.\n\n"
            "Edit reviews.jsonl directly or open it from the application's annotation editor.\n",
            encoding="utf-8",
        )
    return root.resolve()


def _load_package(package: str | Path) -> tuple[ReviewPackageManifest, dict[str, ReviewTask]]:
    root = Path(package)
    manifest = ReviewPackageManifest.model_validate_json((root / "private" / "manifest.json").read_text())
    rows = read_jsonl(root / "private" / "tasks.jsonl")
    if stable_hash(rows) != manifest.tasks_hash:
        raise ValueError("review package task hash mismatch")
    tasks = {r["task_id"]: ReviewTask.model_validate(r) for r in rows}
    if len(tasks) != len(rows) or set(tasks) != set(manifest.task_to_scenario):
        raise ValueError("review package task mapping mismatch")
    expected_id = stable_hash([manifest.source_hash, rows, manifest.task_to_scenario,
                               sorted(manifest.reviewers), manifest.seed, manifest.split, manifest.per_domain])
    if manifest.package_id != expected_id:
        raise ValueError("review package manifest hash mismatch")
    if len(manifest.reviewers) < 2 or len(set(manifest.reviewers)) != len(manifest.reviewers):
        raise ValueError("review package requires distinct reviewers")
    return manifest, tasks


def _read_reviews(
    paths: list[str | Path], manifest: ReviewPackageManifest, tasks: dict[str, ReviewTask],
) -> dict[tuple[str, str], ReviewRecord]:
    reviews = {}
    for path in paths:
        for row in read_jsonl(path):
            review = ReviewRecord.model_validate(row)
            if review.task_id not in tasks or review.reviewer_id not in manifest.reviewers:
                raise ValueError("review refers to an unknown task or unassigned reviewer")
            key = (review.task_id, review.reviewer_id)
            if key in reviews:
                raise ValueError(f"duplicate review for {key}")
            if review.target is not None:
                validate_judgment(tasks[review.task_id], review.target)
            reviews[key] = review
    kinds: dict[str, set[str]] = defaultdict(set)
    for review in reviews.values():
        kinds[review.reviewer_id].add(review.reviewer_type)
    if any(len(types) != 1 for types in kinds.values()):
        raise ValueError("reviewer_type must be consistent for each reviewer")
    return reviews


def _target_fields(target: Target) -> dict[str, Any]:
    data = target.model_dump(mode="json")
    data.pop("rationale_reference")
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = sorted(value)
    return data


def _semantic_judgment(review: ReviewRecord) -> dict[str, Any]:
    assert review.target is not None
    return {**_target_fields(review.target), "ambiguity_level": review.ambiguity_level}


def import_reviews(package: str | Path, paths: list[str | Path], out: str | Path) -> Path:
    manifest, tasks = _load_package(package)
    reviews = _read_reviews(paths, manifest, tasks)
    decisions = []
    adjudication_tasks = []
    summaries = []
    agreements = comparable = 0
    for task_id, task in sorted(tasks.items()):
        complete = [reviews[(task_id, r)] for r in manifest.reviewers
                    if (task_id, r) in reviews and reviews[(task_id, r)].status == "completed"]
        missing = [r for r in manifest.reviewers if (task_id, r) not in reviews
                   or reviews[(task_id, r)].status != "completed"]
        judgments = [_semantic_judgment(r) for r in complete]
        for a, b in combinations(judgments, 2):
            comparable += 1
            agreements += a == b
        differences = [field for field in judgments[0]
                       if len({stable_hash(j[field]) for j in judgments}) > 1] if judgments else []
        status = "needs_reviews" if missing else ("disagreement" if differences else "agreement")
        summaries.append({"task_id": task_id, "domain": task.domain, "status": status,
                          "completed_reviews": len(complete), "missing_reviewers": missing,
                          "disagreement_fields": differences})
        consensus = complete[0] if status == "agreement" else None
        decisions.append(AdjudicationDecision(
            task_id=task_id, target=consensus.target if consensus else None,
            ambiguity_level=consensus.ambiguity_level if consensus else None,
        ))
        adjudication_tasks.append({**task.model_dump(mode="json"), "reviews": [r.model_dump(mode="json") for r in complete],
                                   "review_status": status, "missing_reviewers": missing})
    root = _new_directory(out)
    normalized = [reviews[k] for k in sorted(reviews)]
    write_jsonl(root / "submitted_reviews.jsonl", normalized)
    write_jsonl(root / "adjudication_template.jsonl", decisions)
    _json(root / "adjudication.schema.json", AdjudicationDecision.model_json_schema())
    write_jsonl(root / "adjudication_tasks.jsonl", adjudication_tasks)
    summary = {
        "package_id": manifest.package_id, "tasks": len(tasks),
        "completed_reviews": sum(r.status == "completed" for r in normalized),
        "expected_reviews": len(tasks) * len(manifest.reviewers),
        "statuses": dict(Counter(s["status"] for s in summaries)),
        "comparable_reviewer_pairs": comparable, "agreeing_reviewer_pairs": agreements,
        "exact_pair_agreement": agreements / comparable if comparable else None,
        "reviewer_types": {r.reviewer_id: r.reviewer_type for r in normalized},
        "includes_nonhuman_reviews": any(r.reviewer_type != "human" for r in normalized),
        "cases": summaries,
    }
    _json(root / "review_summary.json", summary)
    _json(root / "import_manifest.json", {
        "package_id": manifest.package_id,
        "reviews_hash": file_hash(root / "submitted_reviews.jsonl"),
        "tasks_hash": file_hash(root / "adjudication_tasks.jsonl"),
        "reviewer_types": {r.reviewer_id: r.reviewer_type for r in normalized},
    })
    lines = ["# Annotation review summary", "", f"Tasks: {len(tasks)}",
             f"Completed reviewer records: {summary['completed_reviews']} / {summary['expected_reviews']}",
             f"Reviewer types: {summary['reviewer_types']}",
             "", "| Task | Domain | Status | Missing reviewers | Disagreement fields |",
             "| --- | --- | --- | --- | --- |"]
    lines.extend(f"| {s['task_id']} | {s['domain']} | {s['status']} | {', '.join(s['missing_reviewers'])} | {', '.join(s['disagreement_fields'])} |" for s in summaries)
    lines.extend(["", "Agreement compares labels, action sets, relevance, abstention and ambiguity; free-text rationales are excluded.",
                  "Every task requires an explicit completed adjudicator decision. Automated models and software fixtures are identified by reviewer_type."])
    if any(r.reviewer_type == "automated_model" for r in normalized):
        lines.extend(["", "This import includes model-generated annotations. Shared-session passes are correlated and are not independent human judgments."])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root.resolve()


def apply_adjudication(
    source: str | Path, package: str | Path, imported: str | Path,
    decisions_path: str | Path, out: str | Path,
) -> Path:
    manifest, tasks = _load_package(package)
    if file_hash(source) != manifest.source_hash:
        raise ValueError("source dataset changed since review export")
    folder = Path(imported)
    imported_manifest = json.loads((folder / "import_manifest.json").read_text())
    if imported_manifest["package_id"] != manifest.package_id:
        raise ValueError("import belongs to a different review package")
    if file_hash(folder / "submitted_reviews.jsonl") != imported_manifest["reviews_hash"]:
        raise ValueError("imported review hash mismatch; import reviews again")
    if file_hash(folder / "adjudication_tasks.jsonl") != imported_manifest["tasks_hash"]:
        raise ValueError("adjudication context hash mismatch; import reviews again")
    reviews = _read_reviews([folder / "submitted_reviews.jsonl"], manifest, tasks)
    decisions: dict[str, AdjudicationDecision] = {}
    for row in read_jsonl(decisions_path):
        decision = AdjudicationDecision.model_validate(row)
        if decision.task_id not in tasks or decision.task_id in decisions:
            raise ValueError("unknown or duplicate adjudication task")
        if decision.status != "completed":
            raise ValueError(f"{decision.task_id}: adjudication is pending")
        _identity(decision.adjudicator_id)
        if decision.adjudicator_id in manifest.reviewers:
            raise ValueError("adjudicator must be distinct from independent reviewers")
        assert decision.target is not None
        validate_judgment(tasks[decision.task_id], decision.target)
        for reviewer in manifest.reviewers:
            record = reviews.get((decision.task_id, reviewer))
            if record is None or record.status != "completed":
                raise ValueError(f"{decision.task_id}: incomplete independent reviews")
        decisions[decision.task_id] = decision
    if set(decisions) != set(tasks):
        raise ValueError("a completed adjudication decision is required for every selected task")
    scenarios = _load_scenarios(source)
    by_id = {s.scenario_id: s for s in scenarios}
    audit = []
    for task_id, decision in sorted(decisions.items()):
        scenario = by_id[manifest.task_to_scenario[task_id]]
        task = tasks[task_id]
        if scenario.domain != task.domain or scenario.context != task.context:
            raise ValueError("review task context differs from the source dataset")
        records = [reviews[(task_id, reviewer)] for reviewer in manifest.reviewers]
        original = {"target": scenario.target.model_dump(mode="json"),
                    "annotation": scenario.annotation.model_dump(mode="json")}
        assert decision.target is not None and decision.ambiguity_level is not None
        if _target_fields(scenario.target) != _target_fields(decision.target) and any(
            child.scenario_family_id == scenario.scenario_family_id
            and child.scenario_id != scenario.scenario_id and child.perturbation.target_invariant
            for child in scenarios
        ):
            raise ValueError(f"{task_id}: changing this target would invalidate unreviewed family variants; "
                             "export a pilot from an unperturbed source dataset")
        scenario.target = decision.target
        all_human = decision.adjudicator_type == "human" and all(r.reviewer_type == "human" for r in records)
        has_fixtures = decision.adjudicator_type == "software_fixture" or any(r.reviewer_type == "software_fixture" for r in records)
        scenario.annotation = Annotation(
            annotator_type="human" if all_human else "mixed",
            annotation_protocol_version="independent-review-v1",
            adjudication_status="adjudicated", ambiguity_level=decision.ambiguity_level,
            provenance=f"package:{manifest.package_id}; audit:annotation_audit.jsonl; " +
                       ("human declarations" if all_human else (
                           "includes software fixtures; not human validation" if has_fixtures
                           else "includes automated model annotations; not human validation")),
            independent_labels=[r.target.situation_label for r in records if r.target],
            review_evidence=[AnnotationEvidence(
                reviewer_id=r.reviewer_id, reviewer_type=r.reviewer_type, task_id=task_id,
                record_hash=stable_hash(r.model_dump(mode="json")),
                situation_label=r.target.situation_label,
            ) for r in records if r.target],
            adjudicator_id=decision.adjudicator_id, adjudicator_type=decision.adjudicator_type,
            review_package_id=manifest.package_id,
        )
        Scenario.model_validate(scenario.model_dump(mode="python"))
        audit.append({"task_id": task_id, "scenario_id": scenario.scenario_id,
                      "scenario_family_id": scenario.scenario_family_id, "split": scenario.split.value,
                      "original": original, "reviews": [r.model_dump(mode="json") for r in records],
                      "decision": decision.model_dump(mode="json"), "package_id": manifest.package_id})
    root = _new_directory(out)
    write_jsonl(root / "reviewed_scenarios.jsonl", scenarios)
    write_jsonl(root / "annotation_audit.jsonl", audit)
    _json(root / "annotation_manifest.json", {
        "schema_version": "annotation-1.0", "package_id": manifest.package_id,
        "source_hash": manifest.source_hash, "reviews_hash": imported_manifest["reviews_hash"],
        "decisions_hash": file_hash(decisions_path), "output_hash": file_hash(root / "reviewed_scenarios.jsonl"),
        "audit_hash": file_hash(root / "annotation_audit.jsonl"), "selected_cases": len(decisions),
        "total_cases": len(scenarios), "propagated_to_family_variants": False,
        "includes_software_fixtures": any(r.reviewer_type == "software_fixture" for r in reviews.values())
            or any(d.adjudicator_type == "software_fixture" for d in decisions.values()),
        "includes_automated_model_annotations": any(r.reviewer_type == "automated_model" for r in reviews.values())
            or any(d.adjudicator_type == "automated_model" for d in decisions.values()),
    })
    return root.resolve()
