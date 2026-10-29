"""Baseline reasoners."""

from __future__ import annotations

from collections import Counter, defaultdict

from .domains import infer_rule_label, safe_action_for_label
from .models import Prediction, RawReasonerResult, Scenario, ScenarioView
from .reasoners import ModelConfig, Reasoner
from .representations import Representation


class MajorityBaseline(Reasoner):
    def __init__(self, config: ModelConfig, training: list[Scenario]) -> None:
        super().__init__(config)
        self.labels: dict[str, str] = {}
        self.actions: dict[str, str] = {}
        by_domain: dict[str, list[Scenario]] = defaultdict(list)
        for scenario in training:
            by_domain[scenario.domain].append(scenario)
        for domain, rows in by_domain.items():
            self.labels[domain] = Counter(row.target.situation_label for row in rows).most_common(1)[0][0]
            self.actions[domain] = Counter(row.target.preferred_actions[0] for row in rows).most_common(1)[0][0]

    def infer(self, view: ScenarioView, representation: Representation) -> RawReasonerResult:
        label = self.labels.get(view.domain, "uncertain")
        action = self.actions.get(view.domain, "no_action")
        prediction = Prediction(
            scenario_id=view.scenario_id,
            situation_label=label,
            relevant_observation_keys=[],
            proposed_action=action,
            confidence=None,
            abstain=False,
            explanation="Majority-class baseline from training split.",
            cited_context_keys=[],
        )
        return self._result(view, representation, prediction, prediction.model_dump_json(), baseline=True)


class RuleBaseline(Reasoner):
    def infer(self, view: ScenarioView, representation: Representation) -> RawReasonerResult:
        values = {obs.key: None if obs.is_missing else obs.value for obs in view.context}
        label, relevant, rationale = infer_rule_label(view.domain, values)
        prediction = Prediction(
            scenario_id=view.scenario_id,
            situation_label=label,
            relevant_observation_keys=[key for key in relevant if key in representation.observation_keys],
            proposed_action=safe_action_for_label(view.domain, label),
            confidence=None,
            abstain=label in {"uncertain", "insufficient_evidence"},
            explanation=f"Rule baseline: {rationale}.",
            cited_context_keys=[key for key in relevant if key in representation.observation_keys],
        )
        return self._result(view, representation, prediction, prediction.model_dump_json(), baseline=True)


class SupervisedFallbackBaseline(Reasoner):
    """Small supervised nearest-neighbor baseline using only stdlib data structures."""

    def __init__(self, config: ModelConfig, training: list[Scenario]) -> None:
        super().__init__(config)
        self.training = training
        self.majority = MajorityBaseline(config, training)

    def infer(self, view: ScenarioView, representation: Representation) -> RawReasonerResult:
        candidates = [row for row in self.training if row.domain == view.domain]
        if not candidates:
            return self.majority.infer(view, representation)
        values = {obs.key: None if obs.is_missing else obs.value for obs in view.context}
        nearest = min(candidates, key=lambda scenario: _distance(values, {obs.key: obs.value for obs in scenario.context}))
        prediction = Prediction(
            scenario_id=view.scenario_id,
            situation_label=nearest.target.situation_label,
            relevant_observation_keys=[
                key for key in nearest.target.relevant_observation_keys if key in representation.observation_keys
            ],
            proposed_action=nearest.target.preferred_actions[0],
            confidence=None,
            abstain=nearest.target.abstain_acceptable,
            explanation="Nearest-neighbor supervised baseline trained on complete training split.",
            cited_context_keys=[
                key for key in nearest.target.relevant_observation_keys if key in representation.observation_keys
            ],
        )
        return self._result(
            view,
            representation,
            prediction,
            prediction.model_dump_json(),
            baseline=True,
            supervised="nearest_neighbor",
            training_size=len(candidates),
        )


def _distance(a: dict[str, object], b: dict[str, object]) -> float:
    score = 0.0
    keys = sorted(set(a) | set(b))
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            score += 1.0
        elif isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            denom = max(abs(float(av)), abs(float(bv)), 1.0)
            score += min(1.0, abs(float(av) - float(bv)) / denom)
        else:
            score += 0.0 if av == bv else 1.0
    return score
