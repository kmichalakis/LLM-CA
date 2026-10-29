# LLM-CA Simulation and Benchmark Platform

LLM-CA is a Python framework for testing context reasoning across pervasive-environment domains without claiming real-world effectiveness from simulated data. It generates canonical scenarios, runs mock or OpenAI-compatible reasoners, validates predictions against explicit labels and allowed actions, evaluates metrics, and produces reproducible reports.

## License and Citation

The source code is distributed under the [MIT license](LICENSE). Software
citation metadata is in [CITATION.cff](CITATION.cff); record the version or commit
used in your work. The code release does not include the manuscript, local
datasets, annotation judgments or generated research artifacts.
See [public release and paper citation](docs/open_source_release.md) for scope,
source packaging, versioning and optional DOI archiving.

## Quickstart

Install the package with plotting support before running reports:

```bash
python -m pip install -e '.[reports]'
```

Use Python 3.11 or newer from the repository root. The optional desktop GUI
requires Tkinter and a graphical session; minimal Linux installations may need
the system's Python Tkinter package. The CLI and automated tests are headless.

Use the bundled/mock workflow first. It is offline and does not call any provider.

```bash
python -m llmca.cli doctor
python -m llmca.cli generate --config configs/benchmark.yaml --seed 42 --out data/scenarios/base.jsonl
python -m llmca.cli validate --input data/scenarios/base.jsonl --strict
python -m llmca.cli run --config configs/experiment_mock.yaml --seed 42 --limit 40
python -m llmca.cli evaluate --run runs/demo_mock
python -m llmca.cli report --run runs/demo_mock --format md,html
```

To open the GUI:

```bash
python -m llmca.gui
```

## What is implemented

- Pydantic v2 canonical schema and exported JSON Schema.
- Seeded four-domain scenario generator with split-safe scenario families.
- Label-preserving perturbations and counterfactual perturbations.
- Natural language, key/value, JSON, and semantic JSON representations.
- Mock reasoner, rule baseline, majority baseline, lightweight supervised nearest-neighbor baseline, and OpenAI-compatible HTTP adapter.
- Offline runner with manifests, raw outputs, normalized predictions, CSV metrics with clustered confidence intervals, scientific SVG/PNG figures, and Markdown/HTML reports.
- Tkinter GUI for generating data, running mock tests, evaluating, and opening reports.
- Unit/integration tests using the Python standard library `unittest` runner.

Evaluation now exports confusion matrices, per-label precision/recall/F1, matched
model comparisons with family-clustered confidence intervals, and paired error
cases. Results are grouped by split. Test-only exports exclude mock conditions.
The HTML report includes expandable confusion matrices and readable comparison tables.
See [statistical evaluation](docs/statistical_evaluation.md) for scoring rules,
methodology, limits, and a four-domain offline demonstration.

```bash
llmca evaluate --run runs/statistics_demo --bootstrap-resamples 2000 --seed 42
llmca report --run runs/statistics_demo --format md,html
```

The **Annotations** button opens the independent-review workspace. It exports
blinded pilot tasks, provides review/adjudication forms, flags disagreements and
writes a new reviewed dataset with provenance. A prepared 40-case pilot is in
`annotations/pilot`. Automatically completed model-review copies and their
disagreement report are in `annotations/pilot/automatic/`; the original human
forms remain blank and adjudication remains pending. See the
[annotation workflow](docs/annotation_workflow.md) for the GUI and CLI steps.

## Frozen Process Paper Package

In the authors' research workspace, the completed full demonstration is in
`runs/paper_v1`: 1,440 synthetic records,
four representations and four offline reasoners, producing 23,040 predictions.
`paper_artifacts/v1/index.html` is the report overview; start writing from
`paper_artifacts/v1/START_HERE.md`. The package includes methods/results drafts,
an outline, limitations, tables, six PNG/SVG figure pairs, test evidence, the
source snapshot, dependency pins and file checksums. The adjacent `v1.zip`
contains the complete frozen package.

The demonstrated contribution is the framework/process. A controlled real-model
benchmark and independent reference-label validation remain outside this release.
See [release reproduction](docs/release_reproduction.md) for exact commands.
These generated outputs and the frozen evidence package are not included in the
source-only repository. Generate new synthetic inputs with the quickstart, or
obtain the separately released evidence package for the full recorded study.

## Tests and Contributions

```bash
python -m pip install -e '.[reports,validation]'
python -m unittest discover -s tests -v
```

The GitHub workflow runs the offline suite on Python 3.11, 3.12 and 3.13 with
provider-network calls blocked. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Scientific caution

Mock and synthetic runs are simulations. They are useful for software validation and pilot planning, but they are not evidence of deployed-system effectiveness. Publishable evaluations require independently reviewed labels, provider configuration records, leakage checks, and real experimental validation.

## Real providers

The OpenAI-compatible adapter reads credentials from environment variables only:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` or a configured `base_url`

Real-provider runs are opt-in. The mock configuration is the default and is excluded from publication tables by default.
