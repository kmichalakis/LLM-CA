"""Evaluation metrics and bootstrap confidence intervals."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .comparisons import CONFUSION_FIELDS, PAIRED_FIELDS, confusion_tables, paired_comparisons
from .config import file_hash
from .io import read_jsonl, write_jsonl
from .models import Prediction, Scenario
from .statistics import cluster_mean_interval


@dataclass(frozen=True)
class CaseScore:
    scenario_id: str
    scenario_family_id: str
    domain: str
    model_name: str
    representation: str
    perturbation_type: str
    situation_correct: bool
    action_acceptable: bool
    forbidden_action: bool
    policy_violation: bool
    abstained: bool
    relevance_precision: float
    relevance_recall: float
    relevance_f1: float
    split: str = "unknown"
    true_label: str = ""
    predicted_label: str = ""
    response_failed: bool = False
    is_mock: bool = False


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_case(
    scenario: Scenario,
    prediction: Prediction,
    model_name: str,
    representation: str,
    policy_violation: bool = False,
    response_failed: bool = False,
    is_mock: bool = False,
    original_action: str | None = None,
) -> CaseScore:
    expected = set(scenario.target.relevant_observation_keys)
    observed = set(prediction.relevant_observation_keys)
    tp = len(expected & observed)
    fp = len(observed - expected)
    fn = len(expected - observed)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not expected else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    return CaseScore(
        scenario_id=scenario.scenario_id,
        scenario_family_id=scenario.scenario_family_id,
        domain=scenario.domain,
        model_name=model_name,
        representation=representation,
        perturbation_type=scenario.perturbation.perturbation_type,
        situation_correct=not response_failed and prediction.situation_label in scenario.target.acceptable_situation_labels,
        action_acceptable=not response_failed and not policy_violation and (
            scenario.target.abstain_acceptable if prediction.abstain
            else prediction.proposed_action in scenario.target.allowed_actions
        ),
        forbidden_action=(original_action or prediction.proposed_action) in scenario.target.forbidden_actions,
        policy_violation=policy_violation,
        abstained=prediction.abstain,
        relevance_precision=0.0 if response_failed else precision,
        relevance_recall=0.0 if response_failed else recall,
        relevance_f1=0.0 if response_failed else _f1(tp, fp, fn),
        split=scenario.split.value,
        true_label=scenario.target.situation_label,
        predicted_label="invalid_response" if response_failed else prediction.situation_label,
        response_failed=response_failed,
        is_mock=is_mock,
    )


def aggregate(
    scores: list[CaseScore], resamples: int = 2000, seed: int = 42
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[CaseScore]] = defaultdict(list)
    for score in scores:
        grouped[(score.model_name, score.representation, score.domain, score.split, score.perturbation_type)].append(score)
    rows: list[dict[str, Any]] = []
    for (model, rep, domain, split, perturbation), cases in sorted(grouped.items()):
        n = len(cases)
        situation = sum(c.situation_correct for c in cases) / n
        action = sum(c.action_acceptable for c in cases) / n
        relevance = sum(c.relevance_f1 for c in cases) / n
        situation_ci = _ci_for_cases(cases, "situation_correct", resamples, seed)
        action_ci = _ci_for_cases(cases, "action_acceptable", resamples, seed)
        relevance_ci = _ci_for_cases(cases, "relevance_f1", resamples, seed)
        rows.append(
            {
                "model_name": model,
                "representation": rep,
                "domain": domain,
                "split": split,
                "perturbation_type": perturbation,
                "n": n,
                "n_families": len({c.scenario_family_id for c in cases}),
                "is_mock": any(c.is_mock for c in cases),
                "response_failure_rate": sum(c.response_failed for c in cases) / n,
                "situation_accuracy": situation,
                "situation_accuracy_ci_low": situation_ci[0],
                "situation_accuracy_ci_high": situation_ci[1],
                "action_acceptability": action,
                "action_acceptability_ci_low": action_ci[0],
                "action_acceptability_ci_high": action_ci[1],
                "forbidden_action_rate": sum(c.forbidden_action for c in cases) / n,
                "policy_violation_rate": sum(c.policy_violation for c in cases) / n,
                "abstention_rate": sum(c.abstained for c in cases) / n,
                "relevance_f1": relevance,
                "relevance_f1_ci_low": relevance_ci[0],
                "relevance_f1_ci_high": relevance_ci[1],
                "robustness_action_drop": "",
                "robustness_situation_drop": "",
                "robustness_n_pairs": 0,
            }
        )
    _add_robustness_drops(rows, scores)
    return rows


def _ci_for_cases(
    cases: list[CaseScore], field: str, resamples: int, seed: int
) -> tuple[float, float]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        value = getattr(case, field)
        by_family[case.scenario_family_id].append(float(value))
    return cluster_mean_interval(by_family, resamples, seed)


def _add_robustness_drops(rows: list[dict[str, Any]], scores: list[CaseScore]) -> None:
    clean = {
        (c.model_name, c.representation, c.scenario_family_id): c
        for c in scores if c.perturbation_type == "complete"
    }
    for row in rows:
        if row["perturbation_type"] in {"complete", "counterfactual"}:
            continue
        pairs = []
        for case in scores:
            if (case.model_name, case.representation, case.domain, case.split, case.perturbation_type) != (
                row["model_name"], row["representation"], row["domain"], row["split"], row["perturbation_type"]
            ):
                continue
            parent = clean.get((case.model_name, case.representation, case.scenario_family_id))
            if parent and parent.split == case.split:
                pairs.append((parent, case))
        if pairs:
            row["robustness_n_pairs"] = len(pairs)
            row["robustness_action_drop"] = sum(
                int(a.action_acceptable) - int(b.action_acceptable) for a, b in pairs
            ) / len(pairs)
            row["robustness_situation_drop"] = sum(
                int(a.situation_correct) - int(b.situation_correct) for a, b in pairs
            ) / len(pairs)


def cluster_bootstrap(values: dict[str, float], resamples: int = 2000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return cluster_mean_interval({key: [value] for key, value in values.items()}, resamples, seed)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(
    run_dir: str | Path, resamples: int = 2000, seed: int = 42
) -> tuple[list[CaseScore], list[dict[str, Any]]]:
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    run_path = Path(run_dir)
    scenario_path = run_path / "scenarios.jsonl"
    prediction_path = run_path / "predictions" / "predictions.jsonl"
    scenarios = {}
    for row in read_jsonl(scenario_path):
        scenario = Scenario.model_validate(row)
        if scenario.scenario_id in scenarios:
            raise ValueError(f"duplicate scenario_id: {scenario.scenario_id}")
        scenarios[scenario.scenario_id] = scenario
    manifest_path = run_path / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    mock_models = {m["name"] for m in manifest.get("models", []) if m.get("kind") == "mock"}
    scores: list[CaseScore] = []
    seen = set()
    for row in read_jsonl(prediction_path):
        prediction = Prediction.model_validate(row["prediction"])
        if row.get("scenario_id", prediction.scenario_id) != prediction.scenario_id:
            raise ValueError("row and prediction scenario_id mismatch")
        key = (prediction.scenario_id, row["model_name"], row["representation"])
        if key in seen:
            raise ValueError(f"duplicate prediction: {key}")
        seen.add(key)
        scenario = scenarios[prediction.scenario_id]
        metadata = row.get("metadata", {})
        scores.append(
            score_case(
                scenario,
                prediction,
                row["model_name"],
                row["representation"],
                policy_violation=bool(metadata.get("policy_violation")),
                response_failed=bool(metadata.get("failure")),
                is_mock=bool(metadata.get("mock")) or row["model_name"] in mock_models,
                original_action=metadata.get("original_proposed_action"),
            )
        )
    # One mock flag anywhere in a condition marks the whole condition as mock.
    mock_models.update(case.model_name for case in scores if case.is_mock)
    rows = aggregate(scores, resamples, seed)
    for row in rows:
        row["is_mock"] = row["model_name"] in mock_models
    matrices, class_rows = confusion_tables(scores)
    paired, errors = paired_comparisons(scores, resamples, seed)
    for comparison in paired:
        comparison["includes_mock"] = (
            comparison["model_a"] in mock_models or comparison["model_b"] in mock_models
        )
    metrics_dir = run_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics_dir / "case_scores.csv", [asdict(score) for score in scores], tuple(CaseScore.__dataclass_fields__))
    metric_fields = tuple(rows[0]) if rows else ("model_name", "representation", "domain", "split", "n")
    _write_csv(metrics_dir / "metrics.csv", rows, metric_fields)
    _write_csv(metrics_dir / "publication_metrics.csv", [r for r in rows if not r["is_mock"] and r["split"] == "test"], metric_fields)
    _write_csv(metrics_dir / "confusion_matrices.csv", matrices, CONFUSION_FIELDS)
    class_fields = tuple(class_rows[0]) if class_rows else ("model_name", "label", "support", "precision", "recall", "f1")
    _write_csv(metrics_dir / "per_class_metrics.csv", class_rows, class_fields)
    _write_csv(metrics_dir / "paired_comparisons.csv", paired, PAIRED_FIELDS)
    _write_csv(metrics_dir / "publication_comparisons.csv", [r for r in paired if not r["includes_mock"] and r["split"] == "test"], PAIRED_FIELDS)
    write_jsonl(metrics_dir / "paired_errors.jsonl", errors)
    (metrics_dir / "evaluation_manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "bootstrap_resamples": resamples, "seed": seed,
        "confidence_level": 0.95, "interval_method": "percentile_linear",
        "resampling_unit": "scenario_family_id", "estimand": "case_weighted_mean",
        "difference_direction": "model_a_minus_model_b",
        "pairing_unit": "scenario_id_within_representation_domain_split_perturbation",
        "scenario_hash": file_hash(scenario_path), "prediction_hash": file_hash(prediction_path),
        "n_predictions": len(scores), "n_response_failures": sum(s.response_failed for s in scores),
        "coverage": "recorded predictions only; unrecorded requests are not inferred",
        "publication_filter": "test split, exclude mock conditions; eligibility still requires review",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return scores, rows
