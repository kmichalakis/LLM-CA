from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmca.evaluation import evaluate_run, score_case
from llmca.execution import run_experiment
from llmca.generation import generate_to_file
from llmca.models import Prediction
from llmca.reporting import generate_report
from llmca.validation import validate_scenarios


class PipelineTests(unittest.TestCase):
    def test_offline_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "base.jsonl"
            run_id = "test_run"
            generate_to_file(
                {
                    "dataset": {"base_scenarios_per_domain": 3},
                    "generation": {"perturbations": False},
                },
                123,
                dataset,
            )
            scenarios, errors = validate_scenarios(dataset)
            self.assertFalse(errors)
            config = {
                "run_id": run_id,
                "dataset": {"input": str(dataset)},
                "representations": ["semantic_json"],
                "models": [{"name": "mock_llmca", "kind": "mock"}, {"name": "rule_baseline", "kind": "rule"}],
                "inference": {"budget": {"max_requests": 1000}},
            }
            cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                run_path = run_experiment(config, seed=123, limit=6)
                scores, rows = evaluate_run(run_path)
                reports = generate_report(run_path, ["md", "html"])
            finally:
                os.chdir(cwd)
            self.assertGreater(len(scores), 0)
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(path.exists() for path in reports))

    def test_metric_fixture(self) -> None:
        from llmca.generation import generate_base_scenarios

        scenario = generate_base_scenarios(per_domain=1, seed=5)[0]
        prediction = Prediction(
            scenario_id=scenario.scenario_id,
            situation_label=scenario.target.situation_label,
            relevant_observation_keys=scenario.target.relevant_observation_keys[:1],
            proposed_action=scenario.target.preferred_actions[0],
            confidence=1.0,
            abstain=False,
            explanation="fixture",
            cited_context_keys=scenario.target.relevant_observation_keys[:1],
        )
        score = score_case(scenario, prediction, "fixture", "json")
        self.assertTrue(score.situation_correct)
        self.assertTrue(score.action_acceptable)
        self.assertFalse(score.forbidden_action)
        self.assertGreaterEqual(score.relevance_f1, 0.0)
        self.assertLessEqual(score.relevance_f1, 1.0)


if __name__ == "__main__":
    unittest.main()
