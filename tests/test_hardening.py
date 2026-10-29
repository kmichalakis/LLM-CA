from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from llmca.evaluation import evaluate_run
from llmca.execution import _enforce_policy, run_experiment
from llmca.generation import generate_base_scenarios, generate_to_file
from llmca.io import read_jsonl, write_jsonl
from llmca.models import Prediction, RawReasonerResult, scenario_view
from llmca.reasoners import parse_prediction
from llmca.representations import build_representation
from llmca.security import scan_for_secrets
from llmca.validation import validate_scenarios


class HardeningTests(unittest.TestCase):
    def test_validation_rejects_bad_units_and_ranges(self) -> None:
        scenario = generate_base_scenarios(per_domain=1, seed=9)[0]
        bad = scenario.model_copy(deep=True)
        bad.context[0].unit = "bad_unit"
        bad.context[0].value = 999999
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.jsonl"
            write_jsonl(path, [bad])
            _scenarios, errors = validate_scenarios(path)
        self.assertTrue(any("unit mismatch" in error for error in errors))
        self.assertTrue(any("above valid range" in error for error in errors))

    def test_prompt_representations_exclude_hidden_target(self) -> None:
        scenario = generate_base_scenarios(per_domain=1, seed=11)[0]
        view = scenario_view(scenario)
        for rep_name in ["natural_language", "key_value", "json", "semantic_json"]:
            content = build_representation(view, rep_name).content
            self.assertNotIn("target", content)
            self.assertNotIn(scenario.target.rationale_reference, content)

    def test_parser_handles_valid_wrapped_and_rejects_bad_outputs(self) -> None:
        scenario = generate_base_scenarios(per_domain=1, seed=4)[0]
        view = scenario_view(scenario)
        payload = {
            "scenario_id": scenario.scenario_id,
            "situation_label": scenario.target.situation_label,
            "relevant_observation_keys": scenario.target.relevant_observation_keys + ["not_real"],
            "proposed_action": scenario.target.preferred_actions[0],
            "confidence": 0.8,
            "abstain": False,
            "explanation": "fixture",
            "cited_context_keys": ["not_real"],
        }
        import json

        parsed = parse_prediction("prefix " + json.dumps(payload) + " suffix", view)
        self.assertNotIn("not_real", parsed.relevant_observation_keys)
        self.assertNotIn("not_real", parsed.cited_context_keys)
        with self.assertRaises(ValueError):
            parse_prediction("I refuse without JSON", view)
        payload.pop("proposed_action")
        with self.assertRaises(Exception):
            parse_prediction(json.dumps(payload), view)

    def test_policy_gate_rejects_forbidden_action_but_tracks_violation(self) -> None:
        scenario = generate_base_scenarios(per_domain=1, seed=6)[0]
        view = scenario_view(scenario)
        forbidden = scenario.target.forbidden_actions[0]
        prediction = Prediction(
            scenario_id=scenario.scenario_id,
            situation_label=scenario.target.situation_label,
            relevant_observation_keys=[],
            proposed_action=forbidden,
            confidence=None,
            abstain=False,
            explanation="unsafe fixture",
            cited_context_keys=[],
        )
        result = RawReasonerResult(
            scenario_id=scenario.scenario_id,
            model_name="fixture",
            representation="json",
            raw_response=prediction.model_dump_json(),
            prediction=prediction,
            metadata={},
        )
        _enforce_policy(scenario, result)
        self.assertTrue(result.metadata["policy_violation"])
        self.assertEqual(result.metadata["original_proposed_action"], forbidden)
        self.assertNotEqual(result.prediction.proposed_action, forbidden)

    def test_budget_cap_and_resume_do_not_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "base.jsonl"
            generate_to_file({"dataset": {"base_scenarios_per_domain": 2}, "generation": {"perturbations": False}}, 5, dataset)
            config = {
                "run_id": "resume_case",
                "dataset": {"input": str(dataset)},
                "representations": ["json"],
                "models": [{"name": "mock_llmca", "kind": "mock"}],
                "inference": {"budget": {"max_requests": 1}},
            }
            cwd = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaises(RuntimeError):
                    run_experiment(config, seed=5, limit=3)
                first_rows = read_jsonl(root / "runs" / "resume_case" / "predictions" / "predictions.jsonl")
                self.assertEqual(len(first_rows), 1)
                config["inference"]["budget"]["max_requests"] = 10
                run_path = run_experiment(config, seed=5, resume=True, limit=3)
                rows = read_jsonl(run_path / "predictions" / "predictions.jsonl")
            finally:
                os.chdir(cwd)
            pairs = {(row["scenario_id"], row["model_name"], row["representation"]) for row in rows}
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(pairs), 3)

    def test_policy_violations_are_counted_in_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scenario = generate_base_scenarios(per_domain=1, seed=8)[0]
            run = root / "runs" / "policy_eval"
            (run / "predictions").mkdir(parents=True)
            write_jsonl(run / "scenarios.jsonl", [scenario])
            pred = Prediction(
                scenario_id=scenario.scenario_id,
                situation_label=scenario.target.situation_label,
                relevant_observation_keys=[],
                proposed_action="no_action",
                confidence=None,
                abstain=False,
                explanation="normalized",
                cited_context_keys=[],
            )
            write_jsonl(
                run / "predictions" / "predictions.jsonl",
                [
                    {
                        "scenario_id": scenario.scenario_id,
                        "model_name": "fixture",
                        "representation": "json",
                        "raw_response": "{}",
                        "prediction": pred.model_dump(mode="json"),
                        "metadata": {"policy_violation": True},
                    }
                ],
            )
            _scores, rows = evaluate_run(run)
            self.assertEqual(rows[0]["policy_violation_rate"], 1.0)

    def test_secret_scanner_detects_api_keys(self) -> None:
        self.assertTrue(scan_for_secrets("Authorization: Bearer sk-" + "testSECRET1234567890"))


if __name__ == "__main__":
    unittest.main()
