"""Command-line interface for LLM-CA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .annotation import apply_adjudication, export_review_package, import_reviews
from .config import load_config
from .evaluation import evaluate_run
from .execution import run_experiment
from .generation import generate_to_file
from .inspect import inspect_dataset
from .reporting import generate_report
from .validation import export_schema, validate_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmca", description="LLM-CA simulation and benchmark CLI")
    parser.add_argument("--version", action="version", version=f"llmca {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate synthetic scenarios")
    generate.add_argument("--config", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--out", required=True)

    validate = sub.add_parser("validate", help="validate scenario JSONL")
    validate.add_argument("--input", required=True)
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--schema-out")

    run = sub.add_parser("run", help="run reasoners over scenarios")
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=42)

    evaluate = sub.add_parser("evaluate", help="evaluate a completed run")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--bootstrap-resamples", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=42)

    report = sub.add_parser("report", help="build reports for a run")
    report.add_argument("--run", required=True)
    report.add_argument("--format", default="md")

    inspect = sub.add_parser("inspect", help="inspect a dataset")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--limit", type=int, default=5)

    annotate = sub.add_parser("annotate", help="independent review and adjudication")
    annotation_commands = annotate.add_subparsers(dest="annotation_command", required=True)
    export = annotation_commands.add_parser("export", help="export a blinded pilot package")
    export.add_argument("--input", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--reviewers", nargs="+", required=True)
    export.add_argument("--per-domain", type=int, default=10)
    export.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    export.add_argument("--seed", type=int, default=42)
    review_import = annotation_commands.add_parser("import", help="validate reviews and flag disagreements")
    review_import.add_argument("--package", required=True)
    review_import.add_argument("--reviews", nargs="+", required=True)
    review_import.add_argument("--out", required=True)
    apply = annotation_commands.add_parser("apply", help="write an audited dataset from completed decisions")
    apply.add_argument("--input", required=True)
    apply.add_argument("--package", required=True)
    apply.add_argument("--imported", required=True)
    apply.add_argument("--decisions", required=True)
    apply.add_argument("--out", required=True)

    doctor = sub.add_parser("doctor", help="check local environment")
    doctor.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            config = load_config(args.config)
            count = generate_to_file(config, args.seed, args.out)
            print(f"Wrote {count} scenarios to {args.out}")
            return 0
        if args.command == "validate":
            scenarios, errors = validate_scenarios(args.input, strict=args.strict)
            if args.schema_out:
                export_schema(args.schema_out)
                print(f"Exported schema to {args.schema_out}")
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
            print(f"Validated {len(scenarios)} scenarios")
            return 0
        if args.command == "run":
            config = load_config(args.config)
            run_path = run_experiment(config, seed=args.seed, dry_run=args.dry_run, resume=args.resume, limit=args.limit)
            print(f"Run path: {run_path}")
            return 0
        if args.command == "evaluate":
            scores, rows = evaluate_run(args.run, resamples=args.bootstrap_resamples, seed=args.seed)
            print(f"Evaluated {len(scores)} predictions into {Path(args.run) / 'metrics' / 'metrics.csv'}")
            print(json.dumps(rows[:5], indent=2))
            return 0
        if args.command == "report":
            paths = generate_report(args.run, formats=[part.strip() for part in args.format.split(",")])
            for path in paths:
                print(path)
            return 0
        if args.command == "inspect":
            print(inspect_dataset(args.input, args.limit))
            return 0
        if args.command == "annotate":
            if args.annotation_command == "export":
                path = export_review_package(args.input, args.out, args.reviewers,
                                             args.per_domain, args.seed, args.split)
            elif args.annotation_command == "import":
                path = import_reviews(args.package, args.reviews, args.out)
            else:
                path = apply_adjudication(args.input, args.package, args.imported, args.decisions, args.out)
            print(f"Annotation output: {path}")
            return 0
        if args.command == "doctor":
            info = {
                "python": sys.version,
                "pydantic": _version("pydantic"),
                "pandas": _version("pandas"),
                "tkinter": _version("tkinter"),
                "package": __version__,
            }
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                for key, value in info.items():
                    print(f"{key}: {value}")
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"llmca: error: {exc}", file=sys.stderr)
        return 2
    parser.error("unhandled command")
    return 2


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "available"))
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


if __name__ == "__main__":
    raise SystemExit(main())
