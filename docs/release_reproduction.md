# Frozen demonstration reproduction

The final process-demonstration matrix uses `configs/experiment_paper_v1.yaml`.
It runs 1,440 synthetic cases through four representations and four offline
reasoners. There are 23,040 planned prediction keys. The final benchmark dataset
is freshly generated into `data/scenarios/paper_v1.jsonl`; the annotation pilot's
original `base.jsonl` remains untouched. The two datasets match semantically after
schema defaults are applied, as checked in the release evidence.

## Environment

Use the frozen `requirements.lock` with Python 3.12.14 for the recorded environment.
It pins runtime, plotting, coverage and build-tool versions. Setup may download
dependencies; the demonstration and tests then execute offline.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

In a fresh source checkout with the annotation evidence copied into
`annotations/pilot` and its source copied to `data/scenarios/base.jsonl`:

```bash
PYTHONHASHSEED=0 .venv/bin/python scripts/reproduce_release.py
```

The script stops if frozen output directories exist. It runs generation,
validation, full inference, evaluation with 2,000 resamples, reporting, captured
tests/coverage, cross-hash-seed replay checks, paper-pack creation and checksum
verification. This is an offline framework/process demonstration, not a controlled
LLM experiment or full SRS acceptance run.

## Frozen files

`paper_artifacts/v1` contains the experiment, input snapshot, annotation evidence,
source ZIP, package versions, methods/results drafts, tables, figures and validation
logs. `freeze_manifest.json` records SHA256 digests. The adjacent `v1.zip` includes
the full pack and has its own `v1.zip.sha256` file.

```bash
.venv/bin/python scripts/verify_frozen.py paper_artifacts/v1
.venv/bin/python -m zipfile -t paper_artifacts/v1.zip
```

Seeded input bytes and deterministic prediction bytes were checked in fresh
processes with Python hash seeds 1 and 97. Figures may differ across font/platform
versions; measured durations, creation timestamps and environment metadata can
differ. No promise of byte-identical whole archives across machines is made.

## GUI smoke check

The native annotation editor has a separate opt-in desktop check:

```bash
.venv/bin/python tests/gui_annotation_smoke.py
```

It uses temporary fixture data, saves reviews and a decision, and checks the form
at its minimum window size. It is separate from the automated-suite coverage.
