"""Freeze measured software-demo artifacts and draft process-paper material."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.metadata
import json
import platform
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import __version__
from .config import file_hash, stable_hash
from .figures import annotation_audit, architecture, dataset_composition, robustness, workflow
from .io import read_jsonl
from .models import Scenario
from .reporting import _html_table, _markdown_table
from .statistics import cluster_mean_interval


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_table(directory: Path, name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    cells = [[f"{r[f]:.4f}" if isinstance(r[f], float) else str(r[f]) for f in fields] for r in rows]
    (directory / f"{name}.md").write_text("\n".join(_markdown_table(fields, cells)) + "\n", encoding="utf-8")


def check_matrix(run: Path) -> dict[str, Any]:
    scenarios = read_jsonl(run / "scenarios.jsonl")
    manifest = json.loads((run / "experiment_manifest.json").read_text())
    configuration = manifest["configuration"]
    if not configuration.get("offline") or any(m["kind"] not in {"mock", "majority", "rule", "supervised"} for m in configuration["models"]):
        raise ValueError("this process-paper pack requires an explicitly offline matrix")
    if not (run / "RUN_COMPLETE").exists():
        raise ValueError("cannot freeze an incomplete run")
    ids = [s["scenario_id"] for s in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate scenarios")
    splits = {}
    for scenario in scenarios:
        Scenario.model_validate(scenario)
        existing = splits.setdefault(scenario["scenario_family_id"], scenario["split"])
        if existing != scenario["split"]:
            raise ValueError("family split leakage")
    selected = ids[:manifest.get("requested_limit")] if manifest.get("requested_limit") else ids
    expected = {(sid, model["name"], rep) for sid in selected for model in configuration["models"]
                for rep in configuration["representations"]}
    rows = read_jsonl(run / "predictions" / "predictions.jsonl")
    actual = [(r["scenario_id"], r["model_name"], r["representation"]) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("run coverage does not match the complete planned matrix")
    raw = read_jsonl(run / "raw" / "raw_responses.jsonl")
    raw_keys = [(r["scenario_id"], r["model_name"], r["representation"]) for r in raw]
    if len(raw_keys) != len(set(raw_keys)) or set(raw_keys) != expected:
        raise ValueError("raw response coverage does not match the planned matrix")
    if manifest["config_hash"] != stable_hash(configuration):
        raise ValueError("frozen configuration hash mismatch")
    if file_hash(run / "input_dataset.snapshot.jsonl") != manifest["dataset_hash"]:
        raise ValueError("input snapshot hash mismatch")
    evaluation = json.loads((run / "metrics" / "evaluation_manifest.json").read_text())
    if evaluation["prediction_hash"] != file_hash(run / "predictions" / "predictions.jsonl") or evaluation["scenario_hash"] != file_hash(run / "scenarios.jsonl"):
        raise ValueError("evaluation inputs changed after scoring")
    fingerprints: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        fingerprints[(row["scenario_id"], row["model_name"])].add(stable_hash(row["prediction"]))
    invariant = all(len(values) == 1 for values in fingerprints.values())
    return {"scenarios": len(scenarios), "families": len(splits), "predictions": len(rows),
            "expected_predictions": len(expected), "missing_predictions": 0, "duplicate_predictions": 0,
            "raw_responses": len(raw), "raw_coverage_matches": True, "configuration_hash_matches": True,
            "response_failures": sum(bool(r["metadata"].get("failure")) for r in rows),
            "family_split_leakage": False, "input_hash_matches": True,
            "evaluation_hashes_match": True, "offline_representation_predictions_identical": invariant,
            "predictions_by_model": dict(Counter(r["model_name"] for r in rows)),
            "no_action_allowed_in_all_targets": all("no_action" in s["target"]["allowed_actions"] for s in scenarios)}


def _archive_files(path: Path, files: list[tuple[Path, str]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, target in sorted(files, key=lambda pair: pair[1]):
            info = zipfile.ZipInfo(target, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())


def freeze_source(root: Path, out: Path) -> str:
    files = []
    for folder, suffix in (("src", ".py"), ("tests", ".py"), ("scripts", ".py"),
                           ("configs", ".yaml"), ("schemas", ".json"), ("prompts", ".txt"), ("docs", ".md")):
        for path in (root / folder).rglob("*" + suffix):
            if "__pycache__" not in path.parts:
                files.append((path, path.relative_to(root).as_posix()))
    for name in ("pyproject.toml", "README.md", "LICENSE", ".gitignore", ".env.example", "requirements.lock"):
        if (root / name).exists():
            files.append((root / name, name))
    _archive_files(out, files)
    return file_hash(out)


def build_tables(run: Path, annotations: Path, directory: Path) -> dict[str, Any]:
    directory.mkdir()
    scenarios = read_jsonl(run / "scenarios.jsonl")
    counts = Counter((s["domain"], s["origin_type"], s["split"], s["perturbation"]["perturbation_type"]) for s in scenarios)
    rows = [{"domain": d, "origin": o, "split": split, "condition": c, "n": n} for (d, o, split, c), n in sorted(counts.items())]
    write_table(directory, "dataset_counts", rows, ["domain", "origin", "split", "condition", "n"])
    labels = Counter((s["domain"], s["split"], s["perturbation"]["perturbation_type"], s["target"]["situation_label"]) for s in scenarios)
    write_table(directory, "label_counts", [{"domain": d, "split": sp, "condition": c, "label": label, "n": n}
                for (d, sp, c, label), n in sorted(labels.items())], ["domain", "split", "condition", "label", "n"])
    metrics = csv_rows(run / "metrics" / "metrics.csv")
    metric_fields = list(metrics[0])
    clean = [r for r in metrics if r["split"] == "test" and r["perturbation_type"] == "complete" and r["is_mock"] == "False"]
    write_table(directory, "clean_test_baselines", clean, metric_fields)
    case_rows = csv_rows(run / "metrics" / "case_scores.csv")
    overall = []
    for model in sorted({r["model_name"] for r in clean}):
        cases = [r for r in case_rows if r["model_name"] == model and r["split"] == "test"
                 and r["perturbation_type"] == "complete" and r["representation"] == "semantic_json"]
        entry: dict[str, Any] = {"model_name": model, "representation": "semantic_json", "n": len(cases),
                                 "families": len({c["scenario_family_id"] for c in cases})}
        for metric, field in (("situation_accuracy", "situation_correct"), ("action_acceptability", "action_acceptable"), ("relevance_f1", "relevance_f1")):
            clusters: dict[str, list[float]] = defaultdict(list)
            for case in cases:
                value = float(case[field] == "True") if field != "relevance_f1" else float(case[field])
                clusters[case["scenario_family_id"]].append(value)
            values = [v for group in clusters.values() for v in group]
            entry[metric] = sum(values) / len(values)
            entry[metric + "_ci_low"], entry[metric + "_ci_high"] = cluster_mean_interval(clusters)
        overall.append(entry)
    write_table(directory, "overall_clean_test", overall, list(overall[0]))
    write_table(directory, "mock_pipeline_checks", [r for r in metrics if r["split"] == "test" and r["perturbation_type"] == "complete" and r["is_mock"] == "True"], metric_fields)
    write_table(directory, "robustness_test_baselines", [r for r in metrics if r["split"] == "test" and r["perturbation_type"] not in {"complete", "counterfactual"} and r["is_mock"] == "False"], metric_fields)
    write_table(directory, "safety_and_abstention", [r for r in metrics if r["split"] == "test" and r["is_mock"] == "False"], ["model_name", "representation", "domain", "perturbation_type", "n", "forbidden_action_rate", "policy_violation_rate", "abstention_rate", "response_failure_rate"])
    by_id = {s["scenario_id"]: s for s in scenarios}
    counterfactuals = defaultdict(list)
    for s in scenarios:
        if s["perturbation"]["perturbation_type"] == "counterfactual":
            parent = by_id[s["perturbation"]["parent_id"]]
            counterfactuals[s["domain"]].append(parent["target"]["situation_label"] != s["target"]["situation_label"])
    write_table(directory, "counterfactual_label_changes", [{"domain": d, "n_variants": len(changes), "canonical_label_changed": sum(changes), "canonical_label_unchanged": len(changes) - sum(changes)} for d, changes in sorted(counterfactuals.items())], ["domain", "n_variants", "canonical_label_changed", "canonical_label_unchanged"])
    predictions = read_jsonl(run / "predictions" / "predictions.jsonl")
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for p in predictions:
        grouped[(p["model_name"], p["scenario_id"])].add(stable_hash(p["prediction"]))
    paths = [{"model_name": model, "n_scenarios": sum(m == model for m, _ in grouped),
              "n_representations": len({p["representation"] for p in predictions if p["model_name"] == model}),
              "identical_prediction_payloads": all(len(values) == 1 for (m, _), values in grouped.items() if m == model),
              "interpretation": "canonical-view software check; not an LLM representation ablation"}
             for model in sorted({p["model_name"] for p in predictions})]
    write_table(directory, "representation_paths", paths, list(paths[0]))
    summary = json.loads((annotations / "automatic" / "imported" / "review_summary.json").read_text())
    domains = sorted({case["domain"] for case in summary["cases"]})
    audit_rows = [{"domain": domain, "cases": sum(c["domain"] == domain for c in summary["cases"]),
                   "full_agreement": sum(c["domain"] == domain and c["status"] == "agreement" for c in summary["cases"]),
                   "disagreement": sum(c["domain"] == domain and c["status"] == "disagreement" for c in summary["cases"]),
                   "canonical_label_agreement": sum(c["domain"] == domain and "situation_label" not in c["disagreement_fields"] for c in summary["cases"])} for domain in domains]
    write_table(directory, "annotation_audit", audit_rows, list(audit_rows[0]))
    field_counts = Counter(field for c in summary["cases"] for field in c["disagreement_fields"])
    write_table(directory, "annotation_disagreement_fields", [{"field": k, "cases": v} for k, v in sorted(field_counts.items())], ["field", "cases"])
    return {"scenarios": scenarios, "metrics": metrics, "clean": clean, "overall": overall,
            "annotation": summary, "counterfactuals": counterfactuals}


def _write_documents(out: Path, data: dict[str, Any], matrix: dict[str, Any], tests: dict[str, Any]) -> None:
    scenarios, audit = data["scenarios"], data["annotation"]
    clean = sum(s["perturbation"]["perturbation_type"] == "complete" for s in scenarios)
    clean_splits = Counter(s["split"] for s in scenarios if s["perturbation"]["perturbation_type"] == "complete")
    agreements = audit["statuses"].get("agreement", 0)
    disagreements = audit["statuses"].get("disagreement", 0)
    label_agreements = sum("situation_label" not in c["disagreement_fields"] for c in audit["cases"])
    docs = {
        "START_HERE.md": f"""# LLM-CA frozen process demonstration

This package is the stopping point for the prototype implementation and a starting
point for a framework/process paper. It records {matrix['predictions']:,} offline
predictions on {matrix['scenarios']:,} synthetic cases. It contains no controlled
provider-LLM performance experiment.

## Writing order

1. Read `paper_outline.md` for the proposed scope and section structure.
2. Adapt `methods.md`, `results.md` and `limitations.md` into the manuscript.
3. Use `tables/` and the PNG/SVG files in `figures/`; captions are in `figure_captions.md`.
4. Use `validation/` for test evidence and `reproduce.md` for exact commands.
5. Open `index.html` for a readable overview; `experiment/report/summary.html`
   contains detailed comparisons and expandable confusion matrices.

The source archive, pinned dependency versions, experiment, annotation evidence,
and file checksums are included. `freeze_manifest.json` binds every included file.
The adjacent ZIP contains this package and can be verified after extraction.

## Evidence at freeze

- {tests['tests']} automated tests passed with no errors or failures.
- {clean:,} clean cases and {len(scenarios) - clean:,} perturbations across four domains.
- Four representations and four offline conditions; all planned prediction keys present.
- {audit['completed_reviews']} model-review records on {audit['tasks']} validation cases:
  {agreements} full agreements and {disagreements} disagreements.
- Annotation decisions remain pending; original generator targets are used for scoring.

Mock outputs are labeled separately. Rules and mock share the generator's decision
function; high agreement measures implementation consistency. The nearest-neighbor
baseline operates within known domains. The same-session annotation passes are
correlated and were exposed to generator rules. These limits are part of the paper.
""",
        "paper_outline.md": """# Process paper outline

Proposed title: LLM-CA A Reproducible Simulation Framework for Context Reasoning in Pervasive Environments

## Contribution and scope

Present the implemented framework, canonical observations, generation and perturbation
workflow, adapter interface, auditable evaluation, annotation tooling and reproducibility
artifacts. The empirical material demonstrates the offline software workflow and exposes
annotation ambiguities. It does not test the primary LLM generalization hypothesis.

## Suggested sections

1. Introduction: context heterogeneity, reproducible experimentation, proposed framework
   objectives and the precise scope of this prototype demonstration.
2. Related work: context-aware computing, context modeling, LLM reasoning, synthetic
   benchmarks and reproducibility. Add verified scholarly references before submission;
   this writing pack does not invent a literature review.
3. Framework design: canonical schema, target-hidden views, four renderers, adapters,
   offline baselines, action auditing, evaluator and graphical/CLI interfaces.
4. Simulation process: variables, provisional rule labels, seeded sampling, family splits,
   six perturbation types, counterfactual recomputation and provenance.
5. Demonstration protocol: frozen settings, inference matrix, recorded denominators,
   statistics, automated tests and representation-path checks.
6. Exploratory annotation audit: model identity limits, direct/conservative policies,
   disagreements and pending adjudication. Do not call this independent human validation.
7. Results: dataset/table counts, offline checks, baseline behavior and annotation issues.
8. Discussion and limitations: circularity, action-set permissiveness, information
   equivalence, oracle-assisted action auditing and untested LLM/held-out-domain behavior.
9. Conclusion and reproducibility: implementation contribution, artifact availability and
   a concrete future controlled model study.

## Submission tasks requiring author judgment

Choose a venue and its format, supply authors/affiliations, verify related-work citations,
state funding/conflicts/AI assistance, and review the manuscript's claims. The current
software requirements are only partially implemented; do not claim full SRS acceptance.
""",
        "methods.md": f"""# Methods draft

## Framework implementation

LLM-CA version {__version__} is a Python 3.11+ simulation and benchmarking prototype.
Pydantic records store domain, origin, source/license, split, seed, timestamps,
observations, targets, annotation metadata, tags and perturbation provenance.
Observation records provide a key, value/type, unit, source, confidence, explicit
missingness and a semantic definition. Dataset validation checks schema membership,
known domain keys, configured units/ranges and scenario-family split integrity.
The frozen implementation validates configured units; it does not implement general
unit conversion or a comprehensive timezone validator.

The domains are smart home, smart learning, smart tourism and non-clinical assisted
living. Situation/action vocabularies and variable ranges are in `src/llmca/domains.py`
inside the source archive. Labels are provisional outputs of domain rules, not
independently adjudicated observations. Sampling begins with cyclic label-oriented
templates and random nuisance values; final label frequencies can differ from the
requested template balance because rule priorities can override the template hint.

## Dataset and perturbations

With seed 42, the generator produces {clean:,} clean records, 150 per domain.
Family splits contain {clean_splits['train']} training, {clean_splits['validation']}
validation and {clean_splits['test']} test clean cases. Families are assigned by
deterministic index cutoffs, not randomized stratified sampling. Perturbations are
added only to test families, yielding {len(scenarios) - clean:,} variants and
{len(scenarios):,} total records. All records are synthetic.

Missingness levels are 0.10 and 0.25, with at least one field removed and a preference
for fields outside the stored relevance set. Because there are only five or six
base fields, both levels often remove one field and can coincide. Irrelevance adds
five distractors. Noise applies bounded multiplicative variation at level 0.10 to
selected numeric fields. Conflict appends one alternate-source observation with a
modified value. Staleness shifts one observation fourteen days backward and reduces
its confidence. These five conditions retain the original target; preservation is
a benchmark assumption rather than an independent semantic guarantee.

Counterfactual mutations alter domain-specific values and recompute the target.
Recomputation does not ensure a different canonical situation label: the parent may
already satisfy the new condition, or a higher-priority rule can still apply.
`tables/counterfactual_label_changes.csv` reports changed and unchanged labels.
Counterfactuals are excluded from label-invariant degradation calculations.

## Conditions and execution

The full matrix contains natural-language, key/value, JSON and semantic-JSON
representations crossed with majority, rules, nearest neighbor and a deterministic
rule-backed mock: {matrix['predictions']:,} prediction records. Every condition runs
all scenarios. Training/validation outputs are diagnostics; principal result tables
use test cases only. Baseline fitting uses complete training cases only.

The majority baseline predicts each domain's most frequent training label and action,
with no relevance predictions. The rule baseline calls the same decision function
used by the generator. The supervised baseline is a within-domain one-nearest-neighbor
classifier, using normalized numeric distances and categorical mismatches, copying
the nearest training target's label, preferred action and relevance set. The mock
uses the rule function and is explicitly flagged as a software-check condition.
Tie order and distance-key order are deterministic. This is not a scikit-learn model.

These offline reasoners consume the target-hidden canonical observation view directly;
they do not parse the rendered representation text. Identical predictions across
renderers are therefore an expected software-path check, not representation ablation
evidence. Renderer metadata differs, and existing equivalence tests check observation
keys rather than full information equality. The provider-compatible HTTP adapter
was not exercised in this frozen matrix. No external provider requests were made;
provider USD cost and model latency were not measured.

The offline action audit uses per-scenario target action sets, retains proposed
actions in raw records, and normalizes rejected actions in scored records. This is
an oracle-assisted simulation check, not an independently deployable safety policy.
No model output causes physical actuation. Missing canonical fields cause conservative
rule/mock abstention; conflict-source and timestamp signals are not substantively
interpreted by those baseline rules.

## Metrics and statistics

Situation correctness allows any acceptable situation label. Action credit requires
an allowed proposed action, or an acceptable abstention; failures and policy
violations receive no action credit. Relevance precision/recall/F1 compare predicted
key sets with stored reference keys, using zero relevance credit on failures.
Separate metrics count forbidden proposals, policy violations, abstentions and
recorded failures. Confusion/per-class tables use the canonical label, which may
differ from acceptable-label accuracy. Macro-F1 and calibration are not reported.

The evaluator computes case-weighted means with 2,000 seed-42 family-cluster bootstrap
draws and 95% percentile intervals. Each draw resamples whole scenario families with
replacement and recomputes the case mean. Paired comparisons align scenario IDs
within representation, domain, split and perturbation, then resample family-level
blocks of paired differences. Positive A minus B favors A. The overall clean-test
table pools the four domains with an unstratified family bootstrap. The report is
exploratory, without multiplicity adjustment or significance claims.

The archived release checks verify all planned matrix keys, no duplicates, no family
split leakage and matching input/evaluation hashes. Evaluation normally covers only
recorded predictions; the release-level completeness check supplies planned-count
accounting for this particular fully completed run.

## Annotation audit

A seed-42 pilot selected {audit['tasks']} clean validation cases, ten per domain.
Reviewer folders omit generated targets and family metadata. The Codex conversation
model supplied {audit['completed_reviews']} records in two policies: direct context
interpretation and a conservative second pass emphasizing uncertainty and intent.
Both passes used one conversation and generator rules were visible earlier. Exact
model version is unrecorded. Their outputs are preserved as actual recorded model
judgments with `automated_model` provenance, not invented API calls or human records.

Agreement compares canonical label, acceptable labels, relevance keys, allowed/
preferred/forbidden actions, abstention and ambiguity. Ordering and free-text
rationales are ignored. This is a policy-disagreement audit, not an independent
annotation reliability experiment. Adjudication remains pending and none of these
annotations replaces the frozen generator targets.
""",
        "limitations.md": """# Limitations and claim boundary

## Supported by this package

An executable multi-domain simulation framework; seeded generation and family grouping;
four renderer paths; offline baseline execution; tested evaluation/annotation tooling;
complete frozen-run accounting; reusable scientific figures; preserved source, inputs,
versions and hashes; and an exploratory audit of correlated model judgments.

## Not established

LLM superiority, successful domain-agnostic generalization, held-out-domain transfer,
real-world efficacy, independent human label validity, an empirical representation
ablation, clinically meaningful anomalies, deployed action safety, or provider cost/
latency efficiency. No controlled provider/local-LLM experiment is included.

## Specific threats to validity

- Rules and mock share the generator's decision function. Perfect clean labels mainly
  demonstrate consistency and should not be interpreted as independent reasoning ability.
- The nearest-neighbor baseline sees the same domains and closely related templates.
  There is no held-out-domain experiment or external dataset validation.
- Every generated target permits `no_action`. High action acceptability can coexist with
  poor situation recognition; it does not show that an engine selects a useful adaptation.
- The action gate consults target action sets. It is an oracle-assisted offline audit,
  not a deployable domain-neutral policy engine or proof of safety.
- Offline reasoners bypass the rendered text. Renderer-path invariance is structural.
  JSON and other renderers expose different metadata; full information equivalence is
  not established by the key-preservation tests.
- Some template hints are overridden by priority rules. Synthetic labels and invariant
  perturbation targets are provisional; counterfactual recomputation can retain labels.
- The two missingness severities can remove the same field count. Conflict/stale signals
  are not interpreted by the rule engine, so invariance can reflect ignored metadata.
- The annotation passes share a model session and prior rule exposure. Model identity
  is incompletely recorded, disagreements are unresolved, and judgments cannot serve
  as independent validation or a separate test reference set.
- Seeded synthetic timestamps are simulation time. General timezone/unit conversion,
  injection resistance, token/USD budget stops, retry accounting and robust resume
  provenance remain incomplete. No live data/actuation is included.
- Confidence intervals describe variability within this synthetic corpus. They do not
  quantify uncertainty about real-world deployment. The run is deterministic; resamples
  are not repeated model trials. No confirmatory significance claims are warranted.

The SRS remains partially implemented. Preserve this disclosure when describing the
prototype as complete for a process demonstration, rather than fully SRS-compliant.
""",
        "figure_captions.md": """# Figure captions

1. **Architecture.** Canonical observations, target-hidden renderer paths, offline
   reasoners, oracle-assisted action auditing, evaluation and report artifacts.
   Reference targets are hidden from inference but used in offline checks and scoring.
2. **Workflow.** Seeded clean cases, family splits, test-family variants, complete
   offline prediction matrix and frozen artifacts. The model annotation audit is a
   separate validation-set activity and does not replace benchmark targets.
3. **Dataset composition.** Counts by domain/split and perturbation, including both
   missingness settings. Records are synthetic; the family count counts clean originals.
4. **Clean offline scores.** Situation accuracy and action acceptability for clean test
   cases in semantic JSON, with 95% family-cluster percentile intervals. Mock shares the
   generator rules. `no_action` is universally acceptable in generated references.
5. **Stress checks.** Baseline situation accuracy for clean, missing, irrelevant, noisy,
   conflicting and stale context. Categories are not severity curves; counterfactuals
   are excluded. Sample sizes are stated per domain/condition. Invariance can reflect
   ignored metadata rather than successful conflict or staleness reasoning.
6. **Annotation disagreements.** Full agreement and per-field disagreement in two
   correlated Codex-session passes. Field counts overlap. No independent-rater
   reliability claim or confidence interval is attached to this descriptive audit.

Every figure has PNG and editable SVG exports. Avoid converting the mock check into
a paper headline or using these figures to imply a controlled LLM evaluation.
""",
    }
    result_lines = ["# Results draft", "", "## Software verification", "",
                    f"The release test run passed {tests['tests']} automated tests with {tests['failures']} failures and {tests['errors']} errors. "
                    "Measured line/branch coverage and individual test records are in validation/. "
                    "The GUI smoke test is documented separately from automated-suite coverage.", "",
                    "## Dataset and completed matrix", "",
                    f"The frozen experiment contains {len(scenarios):,} synthetic records ({clean:,} clean and {len(scenarios) - clean:,} variants). "
                    f"All {matrix['predictions']:,} planned prediction keys are present, with no duplicates, no family split leakage and "
                    f"{matrix['response_failures']} recorded adapter failures. Each representation has the same normalized predictions for every offline reasoner, "
                    "as expected because these engines consume the canonical view directly.", "",
                    "## Clean test baselines", "",
                    "The following scores pool clean test cases across four domains in semantic JSON. Mock checks are in a separate table. "
                    "These are comparisons against generator references, not controlled LLM results.", ""]
    headers = ["Baseline", "n", "Situation accuracy [95% CI]", "Action acceptability [95% CI]", "Relevance F1 [95% CI]"]
    cells = []
    for row in data["overall"]:
        cells.append([row["model_name"], str(row["n"]), *[
            f"{row[m]:.3f} [{row[m + '_ci_low']:.3f}, {row[m + '_ci_high']:.3f}]"
            for m in ("situation_accuracy", "action_acceptability", "relevance_f1")]])
    result_lines.extend(_markdown_table(headers, cells))
    result_lines.extend(["", "The rule baseline's clean-label agreement is expected from its shared decision function with the generator. "
                         "Action acceptability is an especially weak discriminator because every target permits no_action. "
                         "Consult per-domain and stress-condition tables before discussing aggregate scores.", "",
                         "## Annotation audit", "",
                         f"Of {audit['tasks']} validation cases, {agreements} ({100 * agreements / audit['tasks']:.1f}%) have full-field agreement "
                         f"and {disagreements} contain disagreements. Canonical situation labels agree on {label_agreements} "
                         f"({100 * label_agreements / audit['tasks']:.1f}%) cases. The passes differ in interpretation policy and share a session; "
                         "these percentages describe this audit rather than independent annotator reliability. Preferred actions, acceptable sets "
                         "and uncertainty require adjudication. No pilot labels were applied to the benchmark.", "",
                         "## Reporting boundary", "", "Reported intervals are exploratory. No controlled model inference, "
                         "held-out-domain protocol, independent reference-label study or deployed-system measurement was performed."])
    docs["results.md"] = "\n".join(result_lines) + "\n"
    for name, content in docs.items():
        (out / name).write_text(content, encoding="utf-8")


def _overview(out: Path, matrix: dict[str, Any], data: dict[str, Any], tests: dict[str, Any]) -> None:
    names = [("architecture", "Architecture"), ("workflow", "Workflow"),
             ("dataset_composition", "Synthetic dataset composition"),
             ("clean_action_acceptability", "Clean offline scores"),
             ("robustness", "Baseline stress checks"),
             ("annotation_disagreements", "Exploratory annotation audit")]
    figures = "".join(f"<section><h2>{html.escape(title)}</h2><img src='figures/{name}.svg' alt='{html.escape(title)}'>"
                      f"<p><a href='figures/{name}.png'>PNG</a> | <a href='figures/{name}.svg'>SVG</a></p></section>" for name, title in names)
    rows = [[str(r["model_name"]), str(r["n"]), f"{r['situation_accuracy']:.3f}",
             f"{r['action_acceptability']:.3f}", f"{r['relevance_f1']:.3f}"] for r in data["overall"]]
    documents = ["START_HERE.md", "paper_outline.md", "methods.md", "results.md", "limitations.md", "figure_captions.md", "reproduce.md"]
    links = "".join(f"<li><a href='{name}'>{name}</a></li>" for name in documents)
    table_links = "".join(f"<li><a href='tables/{p.name}'>{p.stem.replace('_', ' ')}</a></li>" for p in sorted((out / "tables").glob("*.csv")))
    body = (f"<h1>LLM-CA Frozen Process Demonstration</h1><p>{matrix['scenarios']:,} synthetic cases; "
            f"{matrix['predictions']:,} offline predictions; {tests['tests']} automated tests passed.</p>"
            "<p>The empirical scope is software verification and synthetic baseline behavior. "
            "A controlled LLM performance experiment is not included. The annotation audit uses two correlated model passes.</p>"
            "<nav><a href='experiment/report/summary.html'>Detailed benchmark report</a> | "
            "<a href='annotation_evidence/automatic/imported/summary.md'>Annotation report</a> | "
            "<a href='validation/tests.log'>Test log</a> | <a href='validation/coverage.txt'>Coverage</a> | "
            "<a href='freeze_manifest.json'>Checksums</a></nav>"
            "<h2>Clean Test Baselines</h2><p>Semantic JSON; generator references; mock excluded. "
            "Full confidence intervals are in results.md and the tables. All targets permit no_action.</p>"
            + _html_table(["Baseline", "n", "Situation accuracy", "Action acceptability", "Relevance F1"], rows)
            + "<h2>Writing Pack</h2><ul>" + links + "</ul>" + figures
            + "<h2>Tables</h2><ul>" + table_links + "</ul>")
    style = ("body{font:15px system-ui,sans-serif;color:#20252a;background:#fff;max-width:1180px;margin:auto;padding:24px}"
             "h1{font-size:26px}h2{font-size:20px}section{margin:32px 0;border-top:1px solid #ddd;padding-top:14px}"
             "img{display:block;max-width:100%;height:auto}p{max-width:100ch;line-height:1.5}"
             "a{color:#185c9c}nav{line-height:1.9}table{border-collapse:collapse;font-variant-numeric:tabular-nums}"
             "td,th{border:1px solid #ccc;padding:7px 10px;text-align:left;white-space:nowrap}"
             "th{background:#eef3f4}.table-scroll{overflow-x:auto}")
    (out / "index.html").write_text("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                                   "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                                   f"<title>LLM-CA Frozen Process Demonstration</title><style>{style}</style>"
                                   f"</head><body>{body}</body></html>", encoding="utf-8")


def build_paper_pack(run_dir: str | Path, annotations_dir: str | Path, validation_dir: str | Path,
                     out_dir: str | Path, root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[2]
    run, annotations, validation, out = map(Path, (run_dir, annotations_dir, validation_dir, out_dir))
    matrix = check_matrix(run)
    tests = json.loads((validation / "test_summary.json").read_text())
    if not tests["successful"] or tests["failures"] or tests["errors"]:
        raise ValueError("release test evidence must pass")
    if not (root / "requirements.lock").exists():
        raise ValueError("freeze the environment into requirements.lock before building the pack")
    if out.exists():
        raise FileExistsError("paper pack already exists; use a new version path")
    out.mkdir(parents=True)
    shutil.copytree(run, out / "experiment")
    shutil.copytree(validation, out / "validation")
    shutil.copytree(annotations, out / "annotation_evidence")
    (out / "inputs").mkdir()
    shutil.copyfile(root / "data" / "scenarios" / "base.jsonl", out / "inputs" / "annotation_source.jsonl")
    shutil.copyfile(root / "requirements.lock", out / "requirements.lock")
    (out / "figures").mkdir()
    data = build_tables(run, annotations, out / "tables")
    architecture(out / "figures")
    workflow(out / "figures", sum(s["perturbation"]["perturbation_type"] == "complete" for s in data["scenarios"]), len(data["scenarios"]), matrix["predictions"])
    dataset_composition(out / "figures", data["scenarios"])
    robustness(out / "experiment", data["metrics"])
    for name in ("clean_action_acceptability", "robustness"):
        for extension in ("png", "svg"):
            shutil.copyfile(out / "experiment" / "figures" / f"{name}.{extension}", out / "figures" / f"{name}.{extension}")
    annotation_audit(out / "figures", annotations / "automatic" / "imported" / "review_summary.json")
    _write_documents(out, data, matrix, tests)
    coverage = json.loads((out / "validation" / "coverage.json").read_text())
    totals = coverage["totals"]
    statement_pct = 100 * totals["covered_lines"] / totals["num_statements"]
    branch_pct = 100 * totals["covered_branches"] / totals["num_branches"]
    core_coverage = []
    for name, details in coverage["files"].items():
        if name.endswith(("generation.py", "evaluation.py")):
            summary = details["summary"]
            core_coverage.append({"module": Path(name).name,
                                  "line_coverage_percent": 100 * summary["covered_lines"] / summary["num_statements"],
                                  "branch_coverage_percent": 100 * summary["covered_branches"] / summary["num_branches"]})
    write_table(out / "tables", "core_test_coverage", core_coverage,
                ["module", "line_coverage_percent", "branch_coverage_percent"])
    with (out / "results.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Measured test coverage\n\nAutomated-suite line coverage across the package is {statement_pct:.1f}% "
                 f"and branch coverage is {branch_pct:.1f}%. Reporting/GUI code is less covered. "
                 "The native GUI check is separate from these percentages.\n\n")
        for row in core_coverage:
            fh.write(f"- {row['module']}: {row['line_coverage_percent']:.1f}% line coverage, {row['branch_coverage_percent']:.1f}% branch coverage.\n")
    (out / "matrix_verification.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    packages = {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()}
    (out / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(),
        "packages": dict(sorted(packages.items())), "release": __version__,
        "git_commit": None, "source_identity": "source_snapshot.zip SHA256"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reproduction = """# Reproduction and verification

## Verify the frozen files

Extract the source snapshot into a separate working directory. From that directory:

```bash
python scripts/verify_frozen.py /absolute/path/to/extracted/paper_pack
```

The checker verifies each file listed in freeze_manifest.json. ZIP integrity can
also be checked with `python -m zipfile -t ../v1.zip`. Keep the adjacent SHA256 file
with the archive when distributing it.

## Reproduce the recorded offline experiment

Use Python 3.12.14 to match this environment, then install the pinned packages.
Installation may download dependencies; experiment/test execution is offline.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r /path/to/paper_pack/requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

Copy paper_pack/inputs/annotation_source.jsonl to data/scenarios/base.jsonl and
copy paper_pack/annotation_evidence to annotations/pilot, preserving the records.
In the extracted source directory, run:

```bash
PYTHONHASHSEED=0 .venv/bin/python scripts/reproduce_release.py
```

The script generates data/scenarios/paper_v1.jsonl from configs/benchmark.yaml,
validates it, runs configs/experiment_paper_v1.yaml, evaluates with 2,000 family
bootstrap draws, creates the benchmark report, runs tests with coverage, and
builds a new paper pack. It stops when run/output directories already exist.
Expected matrix: 1,440 scenarios x 4 representations x 4 offline reasoners = 23,040
predictions. Compare generated scenario/prediction/metrics hashes with the recorded
run; compare normalized prediction JSON by (scenario_id, model_name, representation).

Creation timestamps, platform metadata, measured test durations and coverage file
paths can differ. Font/platform differences may alter figure bytes. Identical full
ZIP bytes across machines are not promised. Cross-hash-seed prediction equality and
seeded input regeneration are separately checked in validation/reproducibility.json.

The recorded model annotations can be replayed from automatic_judgments.json with
llmca.automatic_annotation. That reserializes actual saved judgments; it does not
rerun the original model, whose exact version was not recorded.
"""
    (out / "reproduce.md").write_text(reproduction, encoding="utf-8")
    _overview(out, matrix, data, tests)
    source_hash = freeze_source(root, out / "source_snapshot.zip")
    files = {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    freeze = {"artifact_version": "v1", "package_version": __version__, "experiment": "paper_v1",
              "source_snapshot_sha256": source_hash, "file_count": len(files),
              "files_sha256": stable_hash(files), "files": files,
              "scope": "offline synthetic framework/process demonstration; no controlled LLM efficacy claim"}
    (out / "freeze_manifest.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    archive = out.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"archive already exists: {archive}")
    _archive_files(archive, [(p, out.name + "/" + p.relative_to(out).as_posix()) for p in out.rglob("*") if p.is_file()])
    archive.with_suffix(".zip.sha256").write_text(f"{file_hash(archive)}  {archive.name}\n", encoding="utf-8")
    return out.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/paper_v1")
    parser.add_argument("--annotations", default="annotations/pilot")
    parser.add_argument("--validation", default="validation/paper_v1")
    parser.add_argument("--out", default="paper_artifacts/v1")
    args = parser.parse_args()
    print(build_paper_pack(args.run, args.annotations, args.validation, args.out))
