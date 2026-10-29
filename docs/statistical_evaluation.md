# Statistical evaluation

Run `llmca evaluate --run runs/statistics_demo --bootstrap-resamples 2000 --seed 42`
after inference. Evaluation writes all artifacts under the run's `metrics/` directory.
`llmca report --run runs/statistics_demo --format md,html` includes paired comparisons
and expandable confusion matrices in the HTML report.

## Outputs

| File | Contents |
| --- | --- |
| `metrics.csv` | Case means and confidence intervals by model, representation, domain, split and perturbation |
| `case_scores.csv` | Auditable per-case scores, canonical labels, failure and mock flags |
| `confusion_matrices.csv` | Dense long-form counts, including zero cells and invalid responses |
| `per_class_metrics.csv` | Canonical-label support, precision, recall and F1 |
| `paired_comparisons.csv` | Matched model differences, intervals, wins, losses, ties and unmatched counts |
| `paired_errors.jsonl` | Matched cases with different labels, scores, failures or policy outcomes |
| `publication_metrics.csv` | Test-split aggregates with mock conditions excluded |
| `publication_comparisons.csv` | Test-split comparisons with either mock condition excluded |
| `evaluation_manifest.json` | Input hashes, seed, resample count, method and coverage |

Filtered publication files are exports for further review, not a claim of research validity.
Training and validation results remain explicitly labeled in the full audit tables.

## Pairing and intervals

The estimator is the case-weighted mean. For each bootstrap draw, sample scenario
families with replacement, retain every case of each sampled family, and recompute
the mean. This retains dependence among perturbations and handles unequal family
sizes consistently with the reported point estimate. The default is 2,000 draws
and a 95% percentile interval, with linearly interpolated 2.5% and 97.5% quantiles.
Family IDs and scenario IDs are sorted before sampling so input order does not
affect the seeded result.

Compare models only within the same representation, domain, split and perturbation
type. Match scenario IDs before calculating differences. In each bootstrap draw,
sample families of the paired differences, keeping both models together.
Differences are A minus B; all three compared metrics are higher-is-better.
`n_only_a` and `n_only_b` count recorded predictions without a counterpart.
No overlap yields blank estimates and zero pairs. Duplicate prediction keys are
rejected rather than silently overwritten. Repeated stochastic trials need an
explicit trial identifier before they can be evaluated as separate conditions.

The implementation follows the resampling principles described in
[Stata's cluster bootstrap guidance](https://www.stata.com/support/faqs/statistics/bootstrap-with-panel-data/)
and [SciPy's paired bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
It uses the standard library and does not require either product.

Intervals are exploratory, with no multiple-comparison adjustment or significance
claims. One family or constant differences can produce a collapsed interval;
this does not establish population certainty. No independent-case McNemar test is
applied to related perturbation cases. Larger reviewed datasets are required for
substantive conclusions.

## Scoring and coverage

Situation accuracy accepts any explicitly acceptable situation label. Confusion
matrices and per-class metrics use the canonical reference label instead, so their
diagonal need not equal acceptable-label accuracy. Classes absent from the sample
have zero support and zero recall/F1. Failure responses appear as `invalid_response`.

Recorded response failures receive zero situation, action and relevance credit.
An abstention receives action credit only when abstention is acceptable. A policy
violation receives no action credit even if the gate replaced it with `no_action`.
Forbidden-action counts use the original action when available; out-of-allowlist
violations remain a separate metric. Older runs without original-action metadata
cannot reconstruct rejected proposals accurately.

Evaluation covers recorded predictions only. It cannot infer requests that never
produced a record; planned-task accounting remains runner work. Malformed prediction
records fail evaluation with a validation error rather than disappearing from the
denominator. Mock flags are read from both prediction metadata and model manifests.

Robustness drops use only matched clean families in the same split and condition.
Unmatched clean cases do not affect the drop. Blank drops mean unavailable or not
applicable. Counterfactuals change labels and are excluded from degradation scores.
Severity levels are still pooled within each perturbation type; severity curves
remain a later reporting task.

## Offline demonstration

```bash
llmca generate --config configs/benchmark_statistics_demo.yaml --seed 42 --out data/scenarios/statistics_demo.jsonl
llmca validate --input data/scenarios/statistics_demo.jsonl --strict
llmca run --config configs/experiment_statistics_demo.yaml --seed 42
llmca evaluate --run runs/statistics_demo --bootstrap-resamples 2000 --seed 42
llmca report --run runs/statistics_demo --format md,html
```

The demonstration includes all four domains, 40 clean cases, their test-family
perturbations, and four offline reasoners. Its small samples exercise the software.
The mock and rule engine share the generator's rules; their scores are not
independent validation of context reasoning. Both use conservative abstention
when canonical inputs are missing.
