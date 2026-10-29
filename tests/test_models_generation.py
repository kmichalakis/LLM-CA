from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmca.config import load_config
from llmca.generation import generate_base_scenarios, generate_to_file, perturb_scenario
from llmca.io import read_jsonl
from llmca.models import Scenario
from llmca.representations import build_representation, verify_content_equivalence
from llmca.validation import validate_scenarios


class GenerationTests(unittest.TestCase):
    def test_seeded_generation_is_reproducible(self) -> None:
        a = [s.model_dump(mode="json") for s in generate_base_scenarios(per_domain=3, seed=7)]
        b = [s.model_dump(mode="json") for s in generate_base_scenarios(per_domain=3, seed=7)]
        self.assertEqual(a, b)

    def test_generated_file_validates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "base.jsonl"
            config = {
                "dataset": {"base_scenarios_per_domain": 4},
                "generation": {"perturbations": True, "missingness_severities": [0.1], "irrelevance_counts": [1], "noise_severities": [0.1]},
            }
            count = generate_to_file(config, 42, out)
            rows = read_jsonl(out)
            scenarios, errors = validate_scenarios(out)
            self.assertEqual(count, len(rows))
            self.assertFalse(errors)
            self.assertGreater(len(scenarios), 0)

    def test_family_split_integrity(self) -> None:
        scenarios = generate_base_scenarios(per_domain=10, seed=2)
        splits = {}
        for scenario in scenarios:
            existing = splits.setdefault(scenario.scenario_family_id, scenario.split)
            self.assertEqual(existing, scenario.split)

    def test_counterfactual_records_target_change(self) -> None:
        parent = generate_base_scenarios(per_domain=1, seed=2)[0]
        child = perturb_scenario(parent, "counterfactual", "single", 0, 99)
        self.assertFalse(child.perturbation.target_invariant)
        self.assertEqual(child.perturbation.parent_id, parent.scenario_id)

    def test_representations_preserve_observation_keys(self) -> None:
        scenario = generate_base_scenarios(per_domain=1, seed=2)[0]
        view = Scenario.model_validate(scenario.model_dump(mode="python"))
        from llmca.models import scenario_view

        hidden = scenario_view(view)
        names = ["natural_language", "key_value", "json", "semantic_json"]
        verify_content_equivalence(hidden, names)
        for name in names:
            rep = build_representation(hidden, name)
            self.assertEqual(rep.observation_keys, {obs.key for obs in scenario.context})


if __name__ == "__main__":
    unittest.main()
