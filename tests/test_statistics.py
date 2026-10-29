from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llmca.comparisons import confusion_tables, paired_comparisons
from llmca.evaluation import CaseScore, aggregate, evaluate_run, score_case
from llmca.generation import generate_base_scenarios
from llmca.io import write_jsonl
from llmca.models import Prediction, Split, scenario_view
from llmca.reasoners import MockReasoner, ModelConfig
from llmca.representations import build_representation
from llmca.reporting import generate_report
from llmca.statistics import cluster_mean_interval


def case(sid: str, model: str = "a", family: str | None = None, **changes) -> CaseScore:
    base = CaseScore(
        scenario_id=sid, scenario_family_id=family or sid, domain="smart_home",
        model_name=model, representation="json", perturbation_type="complete",
        situation_correct=True, action_acceptable=True, forbidden_action=False,
        policy_violation=False, abstained=False, relevance_precision=1.0,
        relevance_recall=1.0, relevance_f1=1.0, split="test", true_label="resting",
        predicted_label="resting",
    )
    return replace(base, **changes)


class StatisticsTests(unittest.TestCase):
    def test_missing_observations_produce_abstentions_across_domains(self) -> None:
        reasoner = MockReasoner(ModelConfig(name="mock", kind="mock"))
        for scenario in generate_base_scenarios(1):
            for obs in scenario.context:
                missing = scenario.model_copy(deep=True)
                missing_obs = next(o for o in missing.context if o.key == obs.key)
                missing_obs.value = None
                missing_obs.is_missing = True
                view = scenario_view(missing)
                result = reasoner.infer(view, build_representation(view, "json"))
                self.assertTrue(result.prediction.abstain)

    def test_bootstrap_keeps_cluster_members_and_case_weights(self) -> None:
        with patch("llmca.statistics.random.Random") as rng:
            rng.return_value.choice.side_effect = [(3.0, 3), (0.0, 1)]
            lo, hi = cluster_mean_interval({"family_a": [1, 1, 1], "family_b": [0]}, resamples=1)
        self.assertEqual((lo, hi), (0.75, 0.75))

    def test_bootstrap_is_order_independent_seeded_and_validated(self) -> None:
        values = {"b": [0.0, 1.0], "a": [0.3], "c": [1.0]}
        result = cluster_mean_interval(values, 100, 18)
        self.assertEqual(result, cluster_mean_interval(dict(reversed(list(values.items()))), 100, 18))
        for data, n in [({}, 10), ({"a": []}, 10), ({"a": [1]}, 0), ({"a": [float('nan')]}, 10)]:
            with self.assertRaises(ValueError):
                cluster_mean_interval(data, n)

    def test_paired_comparison_matches_ids_and_reports_unmatched(self) -> None:
        scores = [case("s1"), case("s2"), case("only_a"),
                  case("s2", "b", situation_correct=False),
                  case("s1", "b", situation_correct=False), case("only_b", "b")]
        comparisons, errors = paired_comparisons(scores, 100)
        result = next(r for r in comparisons if r["metric"] == "situation_accuracy")
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual((result["n_only_a"], result["n_only_b"]), (1, 1))
        self.assertEqual((result["mean_a"], result["mean_b"]), (1, 0))
        self.assertEqual((result["difference_a_minus_b"], result["ci_low"], result["ci_high"]), (1, 1, 1))
        self.assertEqual((result["a_better"], result["b_better"], result["ties"]), (2, 0, 0))
        self.assertEqual({e["scenario_id"] for e in errors}, {"s1", "s2"})

    def test_comparison_direction_and_family_counts(self) -> None:
        scores = [case("s1", family="f"), case("s2", family="f", situation_correct=False),
                  case("s1", "b", family="f"), case("s2", "b", family="f")]
        result = paired_comparisons(scores, 100)[0][0]
        self.assertEqual(result["n_families"], 1)
        self.assertEqual((result["difference_a_minus_b"], result["ci_low"], result["ci_high"]), (-0.5, -0.5, -0.5))

    def test_split_and_representation_are_never_pooled(self) -> None:
        scores = [case("s1"), case("s1", "b", split="train"),
                  case("s1", "b", representation="semantic_json")]
        self.assertEqual(paired_comparisons(scores, 50)[0], [])
        self.assertEqual(len(aggregate(scores, 50)), 3)

    def test_disjoint_pairs_have_no_invented_estimate(self) -> None:
        result = paired_comparisons([case("s1"), case("s2", "b")], 50)[0][0]
        self.assertEqual(result["n_pairs"], 0)
        self.assertEqual(result["difference_a_minus_b"], "")
        self.assertEqual(result["ci_low"], "")

    def test_confusion_counts_and_hand_computed_class_metrics(self) -> None:
        scores = [case("s1"), case("s2", predicted_label="cooking", situation_correct=False),
                  case("s3", true_label="cooking", predicted_label="cooking"),
                  case("s4", true_label="cooking", predicted_label="invalid_response", response_failed=True)]
        matrices, labels = confusion_tables(scores)
        self.assertEqual(sum(r["count"] for r in matrices), 4)
        resting = next(r for r in labels if r["label"] == "resting")
        self.assertEqual(resting["precision"], 1)
        self.assertEqual(resting["recall"], 0.5)
        self.assertAlmostEqual(resting["f1"], 2 / 3)
        cooking = next(r for r in labels if r["label"] == "cooking")
        self.assertEqual((cooking["precision"], cooking["recall"], cooking["f1"]), (0.5, 0.5, 0.5))

    def test_robustness_drop_uses_only_matched_families(self) -> None:
        scores = [case("parent", family="f"), case("unrelated", situation_correct=False),
                  case("child", family="f", perturbation_type="noise", situation_correct=False),
                  case("cf", family="f", perturbation_type="counterfactual", situation_correct=False)]
        rows = aggregate(scores, 50)
        noise = next(r for r in rows if r["perturbation_type"] == "noise")
        self.assertEqual(noise["robustness_situation_drop"], 1.0)
        self.assertEqual(noise["robustness_n_pairs"], 1)
        counterfactual = next(r for r in rows if r["perturbation_type"] == "counterfactual")
        self.assertEqual(counterfactual["robustness_situation_drop"], "")

    def test_failures_unsafe_actions_and_abstention_are_not_success(self) -> None:
        scenario = generate_base_scenarios(1)[0]
        prediction = Prediction(
            scenario_id=scenario.scenario_id, situation_label=scenario.target.situation_label,
            relevant_observation_keys=scenario.target.relevant_observation_keys,
            proposed_action="no_action", confidence=None, abstain=True,
            explanation="fixture", cited_context_keys=[],
        )
        self.assertFalse(score_case(scenario, prediction, "a", "json").action_acceptable)
        scenario.target.abstain_acceptable = True
        self.assertTrue(score_case(scenario, prediction, "a", "json").action_acceptable)
        failed = score_case(scenario, prediction, "a", "json", response_failed=True)
        self.assertFalse(failed.action_acceptable)
        self.assertFalse(failed.situation_correct)
        self.assertEqual(failed.relevance_f1, 0)
        gated = score_case(scenario, prediction, "a", "json", policy_violation=True,
                           original_action=scenario.target.forbidden_actions[0])
        self.assertFalse(gated.action_acceptable)
        self.assertTrue(gated.forbidden_action)

    def test_acceptable_alternative_differs_from_canonical_confusion(self) -> None:
        scenario = generate_base_scenarios(1)[0]
        scenario.target.acceptable_situation_labels.append("cooking")
        pred = Prediction(
            scenario_id=scenario.scenario_id, situation_label="cooking",
            relevant_observation_keys=[], proposed_action="no_action", confidence=None,
            abstain=False, explanation="alternative label", cited_context_keys=[],
        )
        scored = score_case(scenario, pred, "a", "json")
        self.assertTrue(scored.situation_correct)
        matrices = confusion_tables([scored])[0]
        cell = next(r for r in matrices if r["true_label"] == scenario.target.situation_label
                    and r["predicted_label"] == "cooking")
        self.assertEqual(cell["count"], 1)


class EvaluationExportsTests(unittest.TestCase):
    def test_exports_reports_filters_and_duplicate_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "predictions").mkdir()
            scenarios = generate_base_scenarios(1)[:1]
            scenario = scenarios[0]
            scenario.split = Split.test
            write_jsonl(root / "scenarios.jsonl", [scenario])
            rows = []
            for model in ("baseline", "mock"):
                pred = Prediction(
                    scenario_id=scenario.scenario_id, situation_label="resting",
                    relevant_observation_keys=[], proposed_action="no_action", confidence=None,
                    abstain=False, explanation="fixture", cited_context_keys=[],
                )
                rows.append({"scenario_id": scenario.scenario_id, "model_name": model,
                             "representation": "json", "prediction": pred.model_dump(mode="json"),
                             "metadata": {"mock": model == "mock", "failure": model == "mock"}})
            write_jsonl(root / "predictions" / "predictions.jsonl", rows)
            (root / "experiment_manifest.json").write_text(json.dumps({
                "run_id": "test <script>", "created_utc": "fixture", "dataset_hash": "fixture",
                "config_hash": "fixture", "models": [{"name": "mock", "kind": "mock"}],
            }))
            evaluate_run(root, resamples=50)
            metrics = root / "metrics"
            first = {p.name: p.read_bytes() for p in metrics.iterdir()}
            evaluate_run(root, resamples=50)
            self.assertEqual(first, {p.name: p.read_bytes() for p in metrics.iterdir()})
            with (metrics / "publication_metrics.csv").open() as fh:
                published = list(csv.DictReader(fh))
            self.assertEqual([r["model_name"] for r in published], ["baseline"])
            with (metrics / "publication_comparisons.csv").open() as fh:
                self.assertEqual(list(csv.DictReader(fh)), [])
            self.assertEqual(json.loads((metrics / "evaluation_manifest.json").read_text())["n_response_failures"], 1)
            generate_report(root, ["md", "html"])
            content = (root / "report" / "summary.html").read_text()
            self.assertIn("<table>", content)
            self.assertIn("Reference / prediction", content)
            self.assertIn("<details>", content)
            self.assertNotIn("test <script>", content)
            write_jsonl(root / "predictions" / "predictions.jsonl", rows + rows[:1])
            with self.assertRaisesRegex(ValueError, "duplicate prediction"):
                evaluate_run(root, resamples=50)


if __name__ == "__main__":
    unittest.main()
