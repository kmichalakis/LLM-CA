# LLM-CA Requirements Traceability

Status values:

- Implemented: executable v1 support exists and is covered by at least one automated or demo check.
- Partial: executable support exists but needs broader coverage or research-grade depth.
- Planned: not implemented yet.

| SRS area | MUST requirement summary | Status | Module or artifact | Test or evidence | Remaining work |
|---|---|---:|---|---|---|
| Research boundaries | Distinguish simulation, dataset-derived, and real observations; avoid efficacy claims | Implemented | `llmca.models`, reports, README | `origin_type`, report warning | Add publication checklist before export |
| Evaluation discipline | Compare against labels and acceptable actions, not plausibility | Implemented | `llmca.evaluation`, `llmca.annotation` | `tests/test_pipeline.py`, `tests/test_annotation.py` | Conduct the independent pilot review |
| CLI | `generate`, `validate`, `run`, `evaluate`, `report`, `inspect`, `doctor`; run flags | Implemented | `llmca.cli` | CLI help smoke check | Package install smoke test in CI |
| Outputs | scenario JSONL, raw responses, predictions, metrics, figures, report, manifest | Partial | `llmca.execution`, `llmca.reporting`, `llmca.paper_pack` | full matrix in `runs/paper_v1` and frozen pack | Parquet and some original paper metrics remain unimplemented |
| Mock separation | Mock responses flagged and excluded from filtered test tables | Implemented | `llmca.reasoners`, `llmca.evaluation`, reports | mock-filter tests and separate paper tables | Do not present mock checks as LLM performance |
| Architecture layers | ingestion, normalization, representations, reasoners, policy, evaluation, reporting | Partial | `src/llmca/*` | offline pipeline test | Split ingestion adapters for real datasets |
| Canonical schema | Required scenario, observation, target, annotation fields | Implemented | `llmca.models`, `schemas/scenario.schema.json` | schema/generation tests | Add JSON Schema compatibility tests |
| Timestamps and units | Timezone timestamps and unit consistency | Partial | `llmca.models`, `llmca.validation` | unit mismatch/range tests; generated UTC timestamps | General unit conversion and timezone rejection validation missing |
| Domain ontology | Four domains with labels, actions, variables, ranges, examples | Implemented | `llmca.domains` | generation/validation tests | Expand documented causal assumptions |
| Scenario generator | Seeded, constraint-based, balanced defaults | Implemented | `llmca.generation` | reproducibility tests | Add naturally imbalanced config mode |
| Dataset policy | Dataset-derived provenance, citations, mapping | Planned | docs placeholder | none | Implement dataset adapter contracts |
| Annotation workflow | Blinded tasks, independent reviews, disagreements, explicit adjudication and audited dataset | Implemented | `llmca.annotation`, `llmca.annotation_models`, `llmca.annotation_gui`, `llmca.cli` | Round-trip and rejection tests; native GUI save smoke test | Collect actual human judgments; review protocol quality |
| Perturbations | missingness, irrelevance, noise, conflict, stale, counterfactual metadata | Partial | `llmca.generation` | perturbation tests and full matrix | Target preservation is provisional; counterfactual recomputation need not change label |
| Leakage prevention | Family split integrity; target hidden from prompts | Partial | `llmca.validation`, representations | hardening tests and matrix checks | Offline gate uses reference action sets; provider adapter not benchmarked |
| LLM interface | provider-neutral interface, mock, OpenAI-compatible adapter | Partial | `llmca.reasoners` | parser tests, mock integration | Exercise real/local endpoint in opt-in CI |
| Output validation | schema parsing, citation-key validation, action policy gate | Implemented | `llmca.reasoners`, `llmca.execution` | parser and policy tests | Add logged repair attempt path |
| Retry and budgets | retries, timeout, budget cap, resume | Partial | `llmca.reasoners`, `llmca.execution` | budget/resume test | Add token/USD hard stops for real provider usage |
| Baselines | majority, rules, supervised ML | Partial | `llmca.baselines` | pipeline test | Replace nearest-neighbor with optional scikit-learn baseline |
| Metrics | situation, action, relevance, abstention, safety, canonical confusion matrices and per-class metrics | Partial | `llmca.evaluation`, `llmca.comparisons` | `tests/test_statistics.py`, four-domain statistics demo | Add macro-F1/calibration; planned-task failure accounting |
| Statistics | family-cluster bootstrap CIs, matched model differences, paired errors and matched robustness drops | Partial | `llmca.statistics`, `llmca.comparisons`, `llmca.evaluation` | Hand-calculated fixtures and deterministic export tests | Severity curves, held-out-domain protocol; confirmatory study design |
| Reports | tables, figures, audit trail, threats to validity | Partial | `llmca.reporting`, `llmca.figures`, `llmca.paper_pack` | real PNG/SVG tests and frozen tables/figures | Held-out generalization and provider cost/latency reports require actual studies |
| Security | offline mode, untrusted context, secret redaction | Partial | prompts, `llmca.security` | secret scanner test | Add full log scan command |
| Ethical safeguards | no patient/student identifiers by default | Implemented | domains/generator docs | synthetic generator | Add retention/anonymization policy doc |
| Automated tests | unit and integration tests | Partial | `tests/`, `scripts/validate_release.py` | captured unittest log, JUnit and measured coverage | Ruff/mypy CI gate and broader real-provider failure tests remain incomplete |
| GUI | Start tests and view reports | Implemented | `llmca.gui` | manual launch path | Add integrated report browser and progress bars |

## Next Milestones

1. Add real dataset adapter contracts with explicit license/provenance files.
2. Add optional scikit-learn supervised baseline behind an extra dependency.
3. Extend paired comparison statistics to severity curves and held-out-domain experiments.
4. Conduct independent pilot annotation with the prepared package and audit disagreements.
5. Add CI configuration once this folder is initialized as a git repository.
