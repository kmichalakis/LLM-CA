"""Report generation for completed runs."""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .figures import render_run_figures


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |"
    return [line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)]


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th scope='col'>{html.escape(c)}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<div class='table-scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _interval(row: dict[str, str], field: str) -> str:
    return (f"{float(row[field]):.3f} "
            f"[{float(row[field + '_ci_low']):.3f}, {float(row[field + '_ci_high']):.3f}]")


def _confusion_html(rows: list[dict[str, str]]) -> str:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in ("model_name", "representation", "domain", "split", "perturbation_type"))].append(row)
    sections = []
    for group, cells in sorted(grouped.items()):
        labels = sorted({c["true_label"] for c in cells} | {c["predicted_label"] for c in cells})
        counts = {(c["true_label"], c["predicted_label"]): c["count"] for c in cells}
        n = sum(int(c["count"]) for c in cells)
        title = html.escape(" / ".join(group) + f" (n={n})")
        table = _html_table(["Reference / prediction", *labels], [
            [label, *(counts.get((label, predicted), "0") for predicted in labels)] for label in labels
        ])
        sections.append(f"<details><summary>{title}</summary>{table}</details>")
    return "".join(sections)


def generate_report(run_dir: str | Path, formats: Iterable[str] = ("md",)) -> list[Path]:
    run_path = Path(run_dir)
    report_dir = run_path / "report"
    figures_dir = run_path / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_metrics(run_path / "metrics" / "metrics.csv")
    render_run_figures(run_path)

    manifest = json.loads((run_path / "experiment_manifest.json").read_text(encoding="utf-8"))
    evaluation_path = run_path / "metrics" / "evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
    comparisons_path = run_path / "metrics" / "paired_comparisons.csv"
    comparisons = _read_metrics(comparisons_path) if comparisons_path.exists() else []
    confusion_path = run_path / "metrics" / "confusion_matrices.csv"
    confusion = _read_metrics(confusion_path) if confusion_path.exists() else []
    headers = ["Model", "Representation", "Domain", "Split", "Perturbation", "n", "Families",
               "Situation acc [95% CI]", "Action accept [95% CI]", "Failure rate",
               "Forbidden rate", "Policy violations", "Abstention", "Relevance F1 [95% CI]",
               "Robustness action drop (pairs)"]
    metric_cells = []
    for row in metrics:
        drop = row.get("robustness_action_drop", "")
        metric_cells.append([
            row["model_name"], row["representation"], row["domain"], row.get("split", "unknown"),
            row["perturbation_type"], row["n"], row.get("n_families", ""),
            _interval(row, "situation_accuracy"), _interval(row, "action_acceptability"),
            f"{float(row.get('response_failure_rate', 0)):.3f}",
            f"{float(row['forbidden_action_rate']):.3f}", f"{float(row['policy_violation_rate']):.3f}",
            f"{float(row['abstention_rate']):.3f}", _interval(row, "relevance_f1"),
            f"{float(drop):.3f} ({row.get('robustness_n_pairs', '?')})" if drop else "unavailable",
        ])
    pair_headers = ["Model A", "Model B", "Representation", "Domain", "Split", "Perturbation",
                    "Metric", "Pairs", "Families", "Only A", "Only B", "A - B [95% CI]", "Includes mock"]
    pair_cells = []
    for row in comparisons:
        difference = (f"{float(row['difference_a_minus_b']):.3f} "
                      f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]") if row["n_pairs"] != "0" else "unavailable"
        pair_cells.append([row[k] for k in ("model_a", "model_b", "representation", "domain", "split", "perturbation_type", "metric", "n_pairs", "n_families", "n_only_a", "n_only_b")] + [difference, row["includes_mock"]])
    methodology = (
        f"95% percentile intervals use {evaluation.get('bootstrap_resamples', 'unknown')} resamples "
        f"with seed {evaluation.get('seed', 'unknown')}. Whole scenario families are sampled with replacement; "
        "each draw recomputes the case-weighted mean. Paired differences use the same scenario IDs, "
        "representation, domain, split and perturbation. Positive A - B favors A. "
        "Unmatched predictions are counted separately. Failed recorded responses remain in denominators. "
        "Intervals are exploratory and are not adjusted for multiple comparisons. "
        "Single-family or constant-score intervals may collapse and do not establish certainty. "
        "Robustness drops compare matched clean families; counterfactual relabeling is excluded."
    )
    lines = [
        "# LLM-CA Simulation Report",
        "",
        "This report describes a simulation benchmark run. Mock outputs are not scientific results and must not be reported as deployed-system effectiveness.",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Created UTC: `{manifest['created_utc']}`",
        f"- Dataset hash: `{manifest['dataset_hash']}`",
        f"- Config hash: `{manifest['config_hash']}`",
        "",
        "## Aggregate Metrics",
        "",
        *_markdown_table(headers, metric_cells),
    ]
    lines.extend(["", "## Paired Comparisons", "", methodology, "",
                  *_markdown_table(pair_headers, pair_cells), "", "## Confusion Matrices", "",
                  "Matrices use the canonical reference label. Accuracy separately honors all acceptable labels. "
                  "Full matrices appear in the HTML report.", "",
                  "- [Confusion counts](../metrics/confusion_matrices.csv)",
                  "- [Per-class metrics](../metrics/per_class_metrics.csv)",
                  "- [Paired error cases](../metrics/paired_errors.jsonl)",
                  "- [Test metrics excluding mocks](../metrics/publication_metrics.csv)",
                  "- [Test comparisons excluding mocks](../metrics/publication_comparisons.csv)",
                  "", "Filtered exports still require experimental review before publication."])
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Clean action acceptability](../figures/clean_action_acceptability.svg)",
            "",
            "## Threats To Validity",
            "",
            "Synthetic-to-real transfer, rule-generated provisional labels, domain coverage, prompt sensitivity, model drift, baseline fairness, and simulated action validity all remain limitations until a separately validated study is run.",
        ]
    )
    output_paths: list[Path] = []
    if "md" in formats:
        path = report_dir / "summary.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        output_paths.append(path.resolve())
    if "html" in formats:
        body = (
            "<h1>LLM-CA Simulation Report</h1>"
            "<p>This report describes a simulation benchmark. Mock outputs are software checks.</p>"
            f"<p>Run: {html.escape(str(manifest['run_id']))}</p>"
            "<h2>Clean Context Results</h2><img style='max-width:100%;height:auto' "
            "src='../figures/clean_action_acceptability.svg' alt='Clean context offline scores with family-cluster confidence intervals'>"
            "<h2>Aggregate Metrics</h2>" + _html_table(headers, metric_cells)
            + "<h2>Paired Comparisons</h2><p>" + html.escape(methodology) + "</p>"
            + _html_table(pair_headers, pair_cells)
            + "<h2>Confusion Matrices</h2><p>Rows are canonical reference labels; columns are predictions. "
            "Accuracy separately honors all acceptable labels.</p>" + _confusion_html(confusion)
            + "<h2>Exports</h2><ul>" + "".join(
                f"<li><a href='../metrics/{name}'>{name}</a></li>" for name in (
                    "confusion_matrices.csv", "per_class_metrics.csv", "paired_comparisons.csv",
                    "paired_errors.jsonl", "publication_metrics.csv", "publication_comparisons.csv",
                    "evaluation_manifest.json",
                )
            ) + "</ul><h2>Threats To Validity</h2><p>" + html.escape(lines[-1]) + "</p>"
        )
        path = report_dir / "summary.html"
        style = ("body{font:14px system-ui,sans-serif;margin:24px;color:#20252a;background:#fff}"
                 ".table-scroll{overflow-x:auto;max-width:100%;margin:16px 0}"
                 "table{border-collapse:collapse;font-variant-numeric:tabular-nums}"
                 "th,td{border:1px solid #ccc;padding:6px 9px;text-align:left;white-space:nowrap}"
                 "th{background:#eef2f3}details{margin:12px 0}summary{cursor:pointer}p{max-width:100ch}")
        path.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
                        f"<title>LLM-CA Simulation Report</title><style>{style}</style></head>"
                        f"<body>{body}</body></html>", encoding="utf-8")
        output_paths.append(path.resolve())
    return output_paths
