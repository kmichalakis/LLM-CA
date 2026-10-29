"""Experiment runner and run manifests."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .baselines import MajorityBaseline, RuleBaseline, SupervisedFallbackBaseline
from .config import file_hash, stable_hash
from .io import append_jsonl, read_jsonl, write_jsonl
from .models import RawReasonerResult, Scenario, scenario_view
from .reasoners import ModelConfig, make_reasoner
from .representations import build_representation, verify_content_equivalence
from .validation import validate_scenarios


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _run_dir(run_id: str) -> Path:
    return Path("runs") / run_id


def prepare_run(config: dict[str, Any], seed: int | None, dry_run: bool = False) -> tuple[Path, list[Scenario]]:
    dataset_path = Path(config.get("dataset", {}).get("input", "data/scenarios/base.jsonl"))
    scenarios, errors = validate_scenarios(dataset_path)
    if errors:
        raise ValueError("dataset validation failed:\n" + "\n".join(errors[:20]))
    run_id = str(config.get("run_id") or f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}")
    run_path = _run_dir(run_id)
    if run_path.exists() and not dry_run:
        marker = run_path / "RUN_COMPLETE"
        if marker.exists():
            raise FileExistsError(f"completed run already exists: {run_path}")
    if dry_run:
        return run_path, scenarios
    (run_path / "raw").mkdir(parents=True, exist_ok=True)
    (run_path / "predictions").mkdir(parents=True, exist_ok=True)
    write_jsonl(run_path / "scenarios.jsonl", scenarios)
    prompt_path = Path("prompts/zero_shot_context_reasoning.txt")
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_hash": stable_hash(config),
        "dataset_hash": file_hash(dataset_path),
        "prompt_template_hash": file_hash(prompt_path) if prompt_path.exists() else None,
        "seed": seed,
        "models": config.get("models", []),
        "provider_metadata": {"offline_default": True},
    }
    (run_path / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copyfile(dataset_path, run_path / "input_dataset.snapshot.jsonl")
    return run_path, scenarios


def _completed_pairs(run_path: Path) -> set[tuple[str, str, str]]:
    path = run_path / "predictions" / "predictions.jsonl"
    if not path.exists():
        return set()
    pairs = set()
    for row in read_jsonl(path):
        pairs.add((row["scenario_id"], row["model_name"], row["representation"]))
    return pairs


def run_experiment(
    config: dict[str, Any],
    seed: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    limit: int | None = None,
) -> Path:
    if config.get("offline") and any(
        entry["kind"] not in {"mock", "majority", "rule", "supervised"}
        for entry in config.get("models", [])
    ):
        raise ValueError("offline experiments accept only local deterministic reasoners")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    run_path, scenarios = prepare_run(config, seed, dry_run=dry_run)
    representations = config.get("representations", ["semantic_json"])
    models = config.get("models", [{"name": "mock_llmca", "kind": "mock"}])
    selected = scenarios[:limit] if limit else scenarios
    if dry_run:
        total = len(selected) * len(representations) * len(models)
        print(f"Dry run: {len(selected)} scenarios × {len(representations)} representations × {len(models)} models = {total} calls")
        return run_path

    manifest_path = run_path / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        configuration=config, representations=representations, requested_limit=limit,
        expected_prediction_count=len(selected) * len(representations) * len(models),
        offline=bool(config.get("offline", False)),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_path / "configuration.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    training = [scenario for scenario in scenarios if scenario.split.value == "train" and scenario.perturbation.perturbation_type == "complete"]
    completed = _completed_pairs(run_path) if resume else set()
    budget = config.get("inference", {}).get("budget", {})
    max_requests = int(budget.get("max_requests", 10**9))
    request_count = 0

    for scenario in selected:
        view = scenario_view(scenario)
        verify_content_equivalence(view, representations)
        for rep_name in representations:
            representation = build_representation(view, rep_name)
            for model_entry in models:
                model_config = ModelConfig(
                    name=model_entry["name"],
                    kind=model_entry["kind"],
                    model_id=model_entry.get("model_id"),
                    base_url=model_entry.get("base_url"),
                    temperature=float(model_entry.get("temperature", 0)),
                    max_output_tokens=int(model_entry.get("max_output_tokens", 400)),
                    timeout=float(config.get("inference", {}).get("timeout", 30)),
                    max_retries=int(config.get("inference", {}).get("max_retries", 2)),
                    seed=seed,
                )
                pair = (scenario.scenario_id, model_config.name, rep_name)
                if pair in completed:
                    continue
                if request_count >= max_requests:
                    raise RuntimeError("request budget exhausted")
                reasoner = _reasoner_for(model_config, training)
                result = reasoner.infer(view, representation)
                append_jsonl(run_path / "raw" / "raw_responses.jsonl", [result])
                _enforce_policy(scenario, result)
                append_jsonl(run_path / "predictions" / "predictions.jsonl", [result])
                request_count += 1
    (run_path / "RUN_COMPLETE").write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    return run_path


def _reasoner_for(config: ModelConfig, training: list[Scenario]):
    if config.kind in {"mock", "openai_compatible"}:
        return make_reasoner(config)
    if config.kind == "majority":
        return MajorityBaseline(config, training)
    if config.kind == "rule":
        return RuleBaseline(config)
    if config.kind == "supervised":
        return SupervisedFallbackBaseline(config, training)
    raise ValueError(f"unknown model kind {config.kind!r}")


def _enforce_policy(scenario: Scenario, result: RawReasonerResult) -> None:
    valid_keys = {obs.key for obs in scenario.context}
    result.prediction.cited_context_keys = [key for key in result.prediction.cited_context_keys if key in valid_keys]
    result.prediction.relevant_observation_keys = [key for key in result.prediction.relevant_observation_keys if key in valid_keys]
    if result.prediction.proposed_action not in scenario.target.allowed_actions:
        original_action = result.prediction.proposed_action
        result.metadata["policy_violation"] = True
        result.metadata["original_proposed_action"] = original_action
        if result.prediction.proposed_action in scenario.target.forbidden_actions:
            result.metadata["policy_gate"] = "forbidden_action_detected"
        elif result.prediction.proposed_action != "no_action":
            result.metadata["policy_gate"] = "outside_action_allowlist"
        if "no_action" in scenario.target.allowed_actions:
            result.prediction.proposed_action = "no_action"
            result.prediction.explanation += f" Policy gate replaced unsafe action {original_action!r} with no_action."
        else:
            result.prediction.abstain = True
            result.prediction.explanation += f" Policy gate rejected unsafe action {original_action!r}; abstaining."
