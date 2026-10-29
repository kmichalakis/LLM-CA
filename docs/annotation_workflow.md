# Independent annotation and adjudication

The pilot workflow collects independent judgments, exposes disagreements to an
adjudicator, and writes reviewed targets with an audit trail. It defaults to ten
clean validation cases per domain. The source dataset and existing experiment
outputs remain unchanged.

## Desktop workflow

Start `python -m llmca.gui` after installing the package. Open **Annotations**.
The workspace uses the dataset and seed from the main window.

1. Set the source dataset, a new package folder, two or more pseudonymous reviewer
   IDs, cases per domain, and the sampling split. Select **Export Pilot**.
2. Give each reviewer only their own `reviewers/<id>/` folder. Keep `private/`
   with the coordinator. In **Open Review**, select the reviewer's `reviews.jsonl`.
3. Review **Observations** and complete **Judgment**. Choose the situation,
   additional acceptable labels, relevant keys, allowed/preferred actions,
   abstention acceptability, ambiguity and an evidence rationale. Actions left
   unchecked as allowed become forbidden. **Save Completed** saves the current case.
4. After independent review, select both completed review files with **Import Reviews**.
   Choose a new output folder. Inspect its `summary.md` or `review_summary.json`.
5. Open `adjudication_template.jsonl` with **Open Adjudication**. The observations
   tab now includes the independent judgments. A distinct adjudicator reviews each
   case, supplies a decision rationale, and saves an explicit completed decision.
6. Select **Apply Decisions**, the import folder and completed decisions. Choose
   a new output folder containing `reviewed_scenarios.jsonl`,
   `annotation_audit.jsonl` and `annotation_manifest.json`.

The editor prompts before discarding unsaved changes. Saving a completed case
validates the judgment and atomically replaces the editable submission file.
Package exports, review imports and final dataset folders require new paths.

## CLI workflow

```bash
llmca annotate export --input data/scenarios/base.jsonl --out annotations/pilot --reviewers reviewer_a reviewer_b --per-domain 10 --split validation --seed 42
llmca annotate import --package annotations/pilot --reviews annotations/pilot/reviewers/reviewer_a/reviews.jsonl annotations/pilot/reviewers/reviewer_b/reviews.jsonl --out annotations/pilot/imported
llmca annotate apply --input data/scenarios/base.jsonl --package annotations/pilot --imported annotations/pilot/imported --decisions annotations/pilot/imported/adjudication_template.jsonl --out annotations/pilot/final
llmca validate --input annotations/pilot/final/reviewed_scenarios.jsonl --strict
```

Complete reviews between export and import, and adjudication between import and
apply. Empty templates intentionally cannot pass the final application gate.
For a source checkout without an editable install, use
`PYTHONPATH=src python -m llmca.cli` in place of `llmca` with an interpreter that
has the required packages.

## Review protocol

Protocol version: `independent-review-v1`.

Reviewers work separately using context values, timestamps, source quality and
semantic definitions. They must not consult generated labels, state-machine code,
other reviewers or model predictions during independent review. A label should
describe the situation supported by the observations. Use the ontology's uncertain
or insufficient-evidence label when appropriate and record ambiguity explicitly.
Annotations must not invent additional observations or personal information.

Identify the observations needed to justify the label and action decision. More
than one situation label or action may be acceptable. Specify acceptable sets
explicitly; preferred actions must be a subset of allowed actions. Every ontology
action must be classified as allowed or forbidden. Record a short evidence-based
rationale, and distinguish ambiguous evidence from a negative finding. This pilot
collects simulation judgments; assisted-living cases concern non-clinical routine
signals and simulated actions.

All designated reviewers must complete every selected task before application.
The adjudicator must have an ID distinct from those reviewers. Agreement compares
canonical situation labels, acceptable sets, relevant keys, allowed/preferred/
forbidden action sets, abstention and ambiguity. Set ordering and free-text
rationale differences do not create disagreements. Agreement is descriptive,
without chance correction or uncertainty estimates. The import report includes
the number of comparable reviewer pairs and flags missing reviews separately.

An agreement pre-fills a suggested target, but leaves its decision pending.
Disagreements require an explicit target. The adjudicator can override agreement
with a documented decision. The software checks structure and ontology membership;
reviewers and the study lead are responsible for semantic quality and independence.

## Files and provenance

Reviewer tasks contain opaque task IDs, domain, timestamp, observations, label
choices and action vocabulary. They omit scenario IDs, family IDs, split, generated
targets, annotation metadata, tags and perturbation metadata. Existing source
context is shown faithfully; the study lead must check it for accidental label
leakage or identifying information before distributing real-data review sets.
Blinding is achieved by distributing only reviewer folders; it is not a filesystem
access-control boundary on a shared computer.

The coordinator's private manifest binds sampling settings, reviewer IDs, task
content and the original dataset hash. Imports store normalized review snapshots,
JSON schemas, review summaries, and editable adjudication templates. Applying
decisions verifies source, package, review and adjudication-context hashes.
Unknown IDs, duplicate submissions, unassigned reviewers, invalid action sets,
incomplete coverage and pending decisions are rejected before output creation.

The final dataset retains every scenario ID, family ID, split, observation and
origin type. Only selected cases receive reviewed targets and adjudication metadata.
Changes are not propagated to other family members. A target change is rejected
if it would invalidate unreviewed label-invariant variants; prepare a pilot from
an unperturbed dataset instead. Pilot validation labels do not validate the held-out
test set. Use an explicit `--split test` only under a frozen study protocol.

Each changed case records reviewer IDs, reviewer types, independent labels,
review-record hashes, adjudicator identity and package ID. The audit contains the
original target and annotation, full independent judgments and the final decision.
The final manifest records source, review, decision, output and audit hashes.

Human identity and independence are declared by the study team; the software does
not authenticate people. Synthetic test fixtures must use
`reviewer_type=software_fixture` and `adjudicator_type=software_fixture`. Such
outputs are marked `mixed` and explicitly state that they are not human validation.
The prepared pilot in `annotations/pilot` contains pending templates only.
Its `pending_preview/summary.md` shows the initial zero-completion state; import
actual reviews into a new `imported/` folder after the reviewers finish.

## Automatically completed model reviews

The pilot also has 80 completed records under `annotations/pilot/automatic/`,
imported into `automatic/imported/`. Both passes were authored by the Codex model
in the same conversation, using direct and conservative interpretation policies.
They are correlated, and generator rules were visible earlier in the conversation.
Their agreement is descriptive of these two policies, not independent annotator
reliability. The exact model version is unrecorded.

Actual model-generated judgments use `reviewer_type=automated_model` and require
an `automation` provenance block. They cannot be reclassified as human records
while retaining automated provenance. `software_fixture` remains reserved for
synthetic test records. Final datasets containing model reviews are marked `mixed`,
with separate flags for model annotations and test fixtures.

`automatic_judgments.json` preserves the actual recorded model output. Its compact
row fields are task ID, label, acceptable labels, relevance keys, allowed actions,
preferred actions, abstention acceptability, ambiguity, and rationale. Materialize
a recorded batch into a new folder with:

```bash
PYTHONPATH=src python -m llmca.automatic_annotation --package annotations/pilot --judgments annotations/pilot/automatic_judgments.json --out annotations/pilot/automatic_replay
```

This command validates existing judgments and writes submission files; it does not
call a model or synthesize judgments from the generator. Original human forms and
the source dataset are preserved. Adjudication is still pending for all 40 cases.
