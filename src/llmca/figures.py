"""Scientific figures from recorded run data, with PNG and editable SVG export."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _plotting() -> Any:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("Figure export requires the reports extra: pip install 'llmca[reports]'") from exc
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "svg.fonttype": "none", "svg.hashsalt": "llmca-v1",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _save(figure: Any, directory: Path, name: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    png, svg = directory / f"{name}.png", directory / f"{name}.svg"
    figure.savefig(png, dpi=180, bbox_inches="tight", metadata={"Software": "LLM-CA"})
    figure.savefig(svg, bbox_inches="tight", metadata={"Date": None, "Creator": "LLM-CA"})
    return [png, svg]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _condition(rows: list[dict[str, str]]) -> tuple[str, str]:
    if not rows:
        return "test", "semantic_json"
    split = "test" if any(r.get("split") == "test" for r in rows) else rows[0].get("split", "unknown")
    rep = "semantic_json" if any(r["representation"] == "semantic_json" for r in rows) else rows[0]["representation"]
    return split, rep


def clean_performance(run: Path, rows: list[dict[str, str]]) -> list[Path]:
    plt = _plotting()
    split, rep = _condition(rows)
    clean = [r for r in rows if r.get("split", "unknown") == split
             and r["representation"] == rep and r["perturbation_type"] == "complete"]
    domains = sorted({r["domain"] for r in clean})
    fig, axes = plt.subplots(max(1, (len(domains) + 1) // 2), 2, figsize=(12, max(4, 3.5 * ((len(domains) + 1) // 2))), squeeze=False, layout="constrained")
    for ax, domain in zip(axes.flat, domains):
        selected = sorted([r for r in clean if r["domain"] == domain], key=lambda r: r["model_name"])
        for metric, label, color, marker, offset in (
            ("situation_accuracy", "Situation accuracy", "#2266a4", "o", -0.12),
            ("action_acceptability", "Action acceptability", "#b45309", "s", 0.12),
        ):
            values = [float(r[metric]) for r in selected]
            # A percentile interval can exclude its point estimate; draw endpoints directly.
            for y, r in enumerate(selected):
                ax.hlines(y + offset, float(r[metric + "_ci_low"]), float(r[metric + "_ci_high"]), color=color)
            ax.scatter(values, [y + offset for y in range(len(selected))], label=label, color=color, marker=marker, s=30)
        names = [r["model_name"].replace("_baseline", "").replace("_", " ") for r in selected]
        names = [name + " (mock)" if r.get("is_mock") == "True" else name for name, r in zip(names, selected)]
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Score (case mean; 95% family-cluster percentile CI)")
        n = selected[0]["n"] if selected else "0"
        ax.set_title(f"{domain.replace('_', ' ')}: n={n} clean cases / condition")
        ax.grid(axis="x", alpha=0.2)
    for ax in list(axes.flat)[len(domains):]:
        ax.set_visible(False)
    if domains:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)
    fig.suptitle(f"Offline synthetic pipeline check: {split} split, {rep}\nRule-backed mock shares the generator rules; this is not a controlled LLM experiment", fontsize=12)
    paths = _save(fig, run / "figures", "clean_action_acceptability")
    plt.close(fig)
    return paths


def architecture(directory: Path) -> list[Path]:
    plt = _plotting()
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 4.2), layout="constrained")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")
    labels = ["Canonical data\nObservations + provenance", "Target-hidden view\nFour representations",
              "Offline reasoners\nMock + three baselines", "Audit / policy gate\nNo external actuation",
              "Evaluation\nFamily-cluster intervals", "Reports\nTables + figures + errors"]
    for i, label in enumerate(labels):
        x = 0.15 + i * 2.15
        rect = FancyBboxPatch((x, 2.05), 1.9, 1.0, boxstyle="round,pad=0.04,rounding_size=0.04", facecolor="#eef4f5", edgecolor="#3c666b")
        ax.add_patch(rect)
        ax.text(x + 0.95, 2.55, label, ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + 2.15, 2.55), xytext=(x + 1.91, 2.55), arrowprops={"arrowstyle": "->", "color": "#444"})
    ax.add_patch(FancyBboxPatch((4.55, 0.55), 4.25, 0.8, boxstyle="round,pad=0.04,rounding_size=0.04", facecolor="#fff4dd", edgecolor="#946c28"))
    ax.text(6.675, 0.95, "Reference targets hidden from inference\nUsed by offline action checks and scoring", ha="center", va="center", fontsize=10)
    for x in (7.55, 9.45):
        ax.annotate("", xy=(x, 2.0), xytext=(6.8, 1.38), arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#946c28"})
    ax.set_title("LLM-CA prototype architecture", fontsize=14)
    ax.text(6.5, 0.1, "A provider-compatible adapter exists; no provider inference was exercised in this frozen demonstration.", ha="center", fontsize=10)
    paths = _save(fig, directory, "architecture")
    plt.close(fig)
    return paths


def workflow(directory: Path, n_clean: int, n_total: int, n_predictions: int) -> list[Path]:
    plt = _plotting()
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    labels = [f"Seeded generation\n{n_clean:,} clean cases", "Split by family\nTrain / validation / test",
              f"Test-family perturbations\n{n_total - n_clean:,} variants", f"Full matrix\n{n_predictions:,} predictions",
              "Check + score\nTargets and case denominators", "Archive\nReports, tests, source hashes"]
    locations = [(0.3, 3.3), (4.3, 3.3), (8.3, 3.3), (0.3, 1.8), (4.3, 1.8), (8.3, 1.8)]
    for label, (x, y) in zip(labels, locations):
        ax.add_patch(FancyBboxPatch((x, y), 3.35, 0.85, boxstyle="round,pad=0.04,rounding_size=0.04", edgecolor="#3c666b", facecolor="#eef4f5"))
        ax.text(x + 1.675, y + 0.425, label, ha="center", va="center", fontsize=10)
    for start, end in ((0, 1), (1, 2), (3, 4), (4, 5)):
        x, y = locations[start]
        ex, ey = locations[end]
        ax.annotate("", xy=(ex - 0.05, ey + 0.42), xytext=(x + 3.4, y + 0.42), arrowprops={"arrowstyle": "->"})
    ax.plot([9.98, 9.98, 1.98], [3.25, 2.95, 2.95], color="#333", linewidth=1.2)
    ax.annotate("", xy=(1.98, 2.69), xytext=(1.98, 2.95), arrowprops={"arrowstyle": "->", "color": "#333"})
    ax.text(6, 0.8, "Separate validation-label audit: 40 cases / 80 correlated Codex judgments\n13 full agreements; 27 disagreements; adjudication pending; targets not replaced", ha="center", fontsize=11, bbox={"facecolor": "#fff4dd", "edgecolor": "#946c28", "pad": 10})
    ax.set_title("Frozen demonstration and annotation audit workflow", fontsize=14)
    paths = _save(fig, directory, "workflow")
    plt.close(fig)
    return paths


def dataset_composition(directory: Path, scenarios: list[dict[str, Any]]) -> list[Path]:
    plt = _plotting()
    domains = sorted({s["domain"] for s in scenarios})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    left = [0] * len(domains)
    categories = [("Clean train", "train", "complete", "#2266a4"),
                  ("Clean validation", "validation", "complete", "#d49a2a"),
                  ("Clean test", "test", "complete", "#26826b"),
                  ("Test perturbations", "test", None, "#9a5674")]
    for label, split, kind, color in categories:
        values = [sum(s["domain"] == d and s["split"] == split and (
            s["perturbation"]["perturbation_type"] == kind if kind else s["perturbation"]["perturbation_type"] != "complete"
        ) for s in scenarios) for d in domains]
        axes[0].barh([d.replace("_", " ") for d in domains], values, left=left, label=label, color=color)
        left = [a + b for a, b in zip(left, values)]
    axes[0].set_xlabel("Number of scenario records")
    axes[0].set_title("Composition by domain")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=9)
    counts = Counter(s["perturbation"]["perturbation_type"] for s in scenarios)
    kinds = ["complete", "missingness", "irrelevance", "noise", "conflict", "stale", "counterfactual"]
    values = [counts[k] for k in kinds]
    axes[1].barh(kinds, values, color=["#2266a4"] + ["#26826b"] * 6)
    for index, value in enumerate(values):
        axes[1].text(value + 8, index, str(value), va="center", fontsize=10)
    axes[1].set_xlim(0, max(values, default=1) * 1.18)
    axes[1].set_xlabel("Number of scenario records")
    axes[1].set_title("Perturbation counts (missingness has two levels)")
    fig.suptitle(f"Synthetic dataset: n={len(scenarios):,} records; families={sum(s['perturbation']['perturbation_type'] == 'complete' for s in scenarios):,}", fontsize=13)
    paths = _save(fig, directory, "dataset_composition")
    plt.close(fig)
    return paths


def robustness(run: Path, rows: list[dict[str, str]]) -> list[Path]:
    plt = _plotting()
    split, rep = _condition(rows)
    selected = [r for r in rows if r.get("split", "unknown") == split and r["representation"] == rep
                and r.get("is_mock") != "True"]
    domains = sorted({r["domain"] for r in selected})
    fig, axes = plt.subplots(max(1, (len(domains) + 1) // 2), 2, figsize=(12, 7), layout="constrained", squeeze=False)
    kinds = ["complete", "missingness", "irrelevance", "noise", "conflict", "stale"]
    colors = ["#777777", "#26826b", "#2266a4"]
    for ax, domain in zip(axes.flat, domains):
        models = sorted({r["model_name"] for r in selected if r["domain"] == domain})
        for model_index, model in enumerate(models):
            lookup = {r["perturbation_type"]: r for r in selected if r["domain"] == domain and r["model_name"] == model}
            points = [(index, lookup[k]) for index, k in enumerate(kinds) if k in lookup]
            xs = [i + (model_index - 1) * 0.14 for i, _ in points]
            ys = [float(r["situation_accuracy"]) for _, r in points]
            color = colors[model_index % len(colors)]
            for x, (_, row) in zip(xs, points):
                ax.vlines(x, float(row["situation_accuracy_ci_low"]), float(row["situation_accuracy_ci_high"]), color=color, alpha=0.8)
            ax.scatter(xs, ys, color=color, label=model.replace("_baseline", "").replace("_", " "), s=25)
        ax.set_xticks(range(len(kinds)), [k.replace("irrelevance", "distractors") for k in kinds], rotation=25, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Situation accuracy (95% family-cluster CI)")
        counts = ", ".join(f"{k}: {next((r['n'] for r in selected if r['domain'] == domain and r['perturbation_type'] == k), '0')}" for k in kinds)
        ax.set_title(domain.replace("_", " ") + "\n" + counts, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    for ax in list(axes.flat)[len(domains):]:
        ax.set_visible(False)
    if domains:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9)
    fig.suptitle(f"Synthetic baseline stress checks: {split}, {rep}\nCategorical perturbations; counterfactuals excluded because references are relabeled", fontsize=12)
    paths = _save(fig, run / "figures", "robustness")
    plt.close(fig)
    return paths


def annotation_audit(directory: Path, summary_path: Path) -> list[Path]:
    plt = _plotting()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fields = Counter(field for case in summary["cases"] for field in case["disagreement_fields"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    agree, disagree = summary["statuses"].get("agreement", 0), summary["statuses"].get("disagreement", 0)
    axes[0].barh(["Full agreement", "Disagreement"], [agree, disagree], color=["#26826b", "#bf7052"], height=0.45)
    for index, value in enumerate((agree, disagree)):
        axes[0].text(value + 0.5, index, f"{value} ({100 * value / summary['tasks']:.1f}%)", va="center")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, summary["tasks"])
    axes[0].set_xlabel("Cases")
    axes[0].set_title(f"n={summary['tasks']} cases; {summary['completed_reviews']} model judgments")
    ordered = sorted(fields, key=lambda f: (-fields[f], f))
    axes[1].barh([f.replace("_", " ") for f in ordered], [fields[f] for f in ordered], color="#2266a4")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Cases with field disagreement (overlapping counts)")
    axes[1].set_title("Sources of disagreement")
    for i, field in enumerate(ordered):
        axes[1].text(fields[field] + 0.2, i, str(fields[field]), va="center")
    axes[1].set_xlim(0, max(fields.values(), default=1) * 1.15)
    fig.suptitle("Exploratory annotation audit: two correlated same-session model passes\nGenerator rules were visible; adjudication pending; not independent annotator reliability", fontsize=12)
    paths = _save(fig, directory, "annotation_disagreements")
    plt.close(fig)
    return paths


def render_run_figures(run_dir: str | Path) -> list[Path]:
    run = Path(run_dir)
    rows = _read_csv(run / "metrics" / "metrics.csv")
    return clean_performance(run, rows)
