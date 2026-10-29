"""Confusion counts and comparisons on matched model predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, Any

from .domains import DOMAINS
from .statistics import cluster_mean_interval

if TYPE_CHECKING:
    from .evaluation import CaseScore

GROUP_FIELDS = ("representation", "domain", "split", "perturbation_type")
METRICS = {
    "situation_accuracy": "situation_correct",
    "action_acceptability": "action_acceptable",
    "relevance_f1": "relevance_f1",
}
CONFUSION_FIELDS = ("model_name", *GROUP_FIELDS, "true_label", "predicted_label", "count")
PAIRED_FIELDS = (
    *GROUP_FIELDS, "model_a", "model_b", "metric", "n_pairs", "n_families",
    "n_only_a", "n_only_b", "mean_a", "mean_b", "difference_a_minus_b",
    "ci_low", "ci_high", "a_better", "b_better", "ties", "includes_mock",
)


def confusion_tables(
    scores: list[CaseScore],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Count canonical-reference labels; acceptable-label accuracy stays separate."""
    grouped: dict[tuple[str, ...], list[CaseScore]] = defaultdict(list)
    for case in scores:
        grouped[(case.model_name, *(getattr(case, field) for field in GROUP_FIELDS))].append(case)
    matrices: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for group, cases in sorted(grouped.items()):
        metadata = dict(zip(("model_name", *GROUP_FIELDS), group))
        counts = Counter((case.true_label, case.predicted_label) for case in cases)
        spec = DOMAINS.get(str(metadata["domain"]))
        reference_labels = {c.true_label for c in cases} | (set(spec.labels) if spec else set())
        labels = sorted(reference_labels | {c.predicted_label for c in cases})
        for actual in labels:
            for predicted in labels:
                matrices.append({**metadata, "true_label": actual, "predicted_label": predicted,
                                 "count": counts[(actual, predicted)]})
        for label in sorted(reference_labels):
            tp = counts[(label, label)]
            support = sum(counts[(label, predicted)] for predicted in labels)
            predicted_n = sum(counts[(actual, label)] for actual in labels)
            precision = tp / predicted_n if predicted_n else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            class_rows.append({**metadata, "label": label, "support": support,
                               "predicted_n": predicted_n, "precision": precision,
                               "recall": recall, "f1": f1})
    return matrices, class_rows


def paired_comparisons(
    scores: list[CaseScore], resamples: int = 2000, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], dict[str, dict[str, CaseScore]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for case in scores:
        group = tuple(getattr(case, field) for field in GROUP_FIELDS)
        model_cases = grouped[group][case.model_name]
        if case.scenario_id in model_cases:
            raise ValueError(f"duplicate prediction for {case.model_name}/{case.scenario_id}")
        model_cases[case.scenario_id] = case
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for group, models in sorted(grouped.items()):
        metadata = dict(zip(GROUP_FIELDS, group))
        for model_a, model_b in combinations(sorted(models), 2):
            a, b = models[model_a], models[model_b]
            common = sorted(a.keys() & b.keys())
            pair_metadata = {**metadata, "model_a": model_a, "model_b": model_b,
                             "n_pairs": len(common),
                             "n_families": len({a[sid].scenario_family_id for sid in common}),
                             "n_only_a": len(a.keys() - b.keys()),
                             "n_only_b": len(b.keys() - a.keys()),
                             "includes_mock": any(c.is_mock for c in (*a.values(), *b.values()))}
            for metric, field in METRICS.items():
                clusters: dict[str, list[float]] = defaultdict(list)
                av, bv = [], []
                for sid in common:
                    if a[sid].scenario_family_id != b[sid].scenario_family_id:
                        raise ValueError(f"family mismatch for paired scenario {sid}")
                    va, vb = float(getattr(a[sid], field)), float(getattr(b[sid], field))
                    av.append(va)
                    bv.append(vb)
                    clusters[a[sid].scenario_family_id].append(va - vb)
                lo, hi = cluster_mean_interval(clusters, resamples, seed) if common else ("", "")
                results.append({
                    **pair_metadata, "metric": metric,
                    "mean_a": sum(av) / len(av) if av else "",
                    "mean_b": sum(bv) / len(bv) if bv else "",
                    "difference_a_minus_b": sum(x - y for x, y in zip(av, bv)) / len(av) if av else "",
                    "ci_low": lo, "ci_high": hi,
                    "a_better": sum(x > y for x, y in zip(av, bv)),
                    "b_better": sum(x < y for x, y in zip(av, bv)),
                    "ties": sum(x == y for x, y in zip(av, bv)),
                })
            for sid in common:
                ca, cb = a[sid], b[sid]
                if any(getattr(ca, f) != getattr(cb, f) for f in (
                    "predicted_label", "situation_correct", "action_acceptable",
                    "relevance_f1", "policy_violation", "response_failed",
                )):
                    errors.append({
                        **metadata, "model_a": model_a, "model_b": model_b,
                        "scenario_id": sid, "scenario_family_id": ca.scenario_family_id,
                        "true_label": ca.true_label,
                        "predicted_label_a": ca.predicted_label,
                        "predicted_label_b": cb.predicted_label,
                        **{f"{metric}_{suffix}": getattr(case, field)
                           for metric, field in METRICS.items()
                           for suffix, case in (("a", ca), ("b", cb))},
                        "response_failed_a": ca.response_failed,
                        "response_failed_b": cb.response_failed,
                    })
    return results, errors
