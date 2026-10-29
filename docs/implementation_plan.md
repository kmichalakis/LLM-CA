# LLM-CA Implementation Plan

This plan maps the SRS requirements to the v1 repository modules and tests.

## Module map

- Schema and auditability: `llmca.models`, `llmca.validation`, `schemas/scenario.schema.json`.
- Domain ontologies and state machines: `llmca.domains`.
- Seeded generation and perturbations: `llmca.generation`.
- Context representations and equivalence checks: `llmca.representations`.
- Provider interface and adapters: `llmca.reasoners`.
- Baselines: `llmca.baselines`.
- Run orchestration, budgets, resume, manifests: `llmca.execution`.
- Metrics and bootstrap utilities: `llmca.evaluation`.
- Family-cluster intervals and paired model diagnostics: `llmca.statistics`, `llmca.comparisons`.
- Reports, tables, figures, audit trail: `llmca.reporting`.
- Scientific plots and frozen process-paper pack: `llmca.figures`, `llmca.paper_pack`.
- GUI: `llmca.gui`.
- Independent review and adjudication: `llmca.annotation`, `llmca.annotation_models`,
  `llmca.annotation_gui`.
- CLI: `llmca.cli`.

## Test map

- Schema acceptance/rejection: `tests/test_models_generation.py`.
- Seeded generation and split integrity: `tests/test_models_generation.py`.
- Perturbation invariants and counterfactual relabeling: `tests/test_models_generation.py`.
- Representation content equivalence: `tests/test_models_generation.py`.
- Offline integration pipeline: `tests/test_pipeline.py`.
- Metric fixture: `tests/test_pipeline.py`.
- Paired differences, confusion counts, split isolation, failure scoring, mock filters,
  unequal-cluster weighting and deterministic exports: `tests/test_statistics.py`.
- Blinding, independent review coverage, ontology validation, disagreement handling,
  immutable source, variant consistency and audited application: `tests/test_annotation.py`.
- Opt-in native form saving and small-window layout: `tests/gui_annotation_smoke.py`.

## Current limitations

- The supervised baseline uses a lightweight nearest-neighbor classifier so it remains offline.
- The frozen release has actual PNG and editable SVG figures; earlier demo runs retain historical artifacts.
- Annotation tooling is implemented; the pilot still requires independent human judgments.
- Real-provider calls are implemented for OpenAI-compatible HTTP endpoints but not exercised in offline tests.

## Process Paper Freeze

`runs/paper_v1` is the complete offline matrix. `paper_artifacts/v1` contains
the writing pack and frozen evidence. Reproduction scripts in `scripts/` retain
test logs, measured coverage, complete-matrix/hash checks and source checksums.
The two model annotation passes are correlated and unadjudicated. This freeze
completes the process-demonstration scope, not every MUST requirement in the SRS.
