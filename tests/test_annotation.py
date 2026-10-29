from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llmca.annotation import apply_adjudication, export_review_package, import_reviews
from llmca.annotation_models import AdjudicationDecision, ReviewRecord, ReviewTask
from llmca.cli import main
from llmca.config import file_hash
from llmca.generation import generate_base_scenarios, perturb_scenario
from llmca.io import read_jsonl, write_jsonl
from llmca.models import Scenario, Target
from llmca.validation import validate_scenarios


class AnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.jsonl"
        scenarios = generate_base_scenarios(per_domain=10, seed=37)
        scenarios.append(perturb_scenario(scenarios[6], "irrelevance", 2, 0, 7))
        self.scenarios = {s.scenario_id: s for s in scenarios}
        write_jsonl(self.source, scenarios)
        self.original = self.source.read_bytes()
        self.package = export_review_package(self.source, self.root / "pilot", ["alice", "bob"], 1, 19)
        self.manifest = json.loads((self.package / "private" / "manifest.json").read_text())

    def complete_reviews(self, disagreement: bool = False) -> list[Path]:
        paths = []
        for reviewer in ("alice", "bob"):
            path = self.package / "reviewers" / reviewer / "reviews.jsonl"
            records = []
            for index, row in enumerate(read_jsonl(path)):
                scenario = self.scenarios[self.manifest["task_to_scenario"][row["task_id"]]]
                target = scenario.target.model_copy(deep=True)
                if disagreement and reviewer == "bob" and index == 0:
                    labels = read_jsonl(self.package / "reviewers" / reviewer / "tasks.jsonl")[index]["situation_labels"]
                    alternative = next(label for label in labels if label != target.situation_label)
                    target.situation_label = alternative
                    target.acceptable_situation_labels = [alternative]
                target.rationale_reference = f"Software test judgment by {reviewer}"
                records.append(ReviewRecord(
                    task_id=row["task_id"], reviewer_id=reviewer, reviewer_type="software_fixture",
                    status="completed", target=target, ambiguity_level="low",
                ))
            write_jsonl(path, records)
            paths.append(path)
        return paths

    def decide(self, imported: Path) -> Path:
        decisions = []
        for row in read_jsonl(imported / "adjudication_template.jsonl"):
            scenario = self.scenarios[self.manifest["task_to_scenario"][row["task_id"]]]
            decisions.append(AdjudicationDecision(
                task_id=row["task_id"], adjudicator_id="adjudicator", adjudicator_type="software_fixture",
                status="completed", target=scenario.target, ambiguity_level="medium",
                rationale="Software fixture decision; not a human review",
            ))
        path = self.root / "decisions.jsonl"
        write_jsonl(path, decisions)
        return path

    def test_export_is_blinded_balanced_seeded_and_split_safe(self) -> None:
        other = export_review_package(self.source, self.root / "other", ["bob", "alice"], 1, 19)
        self.assertEqual((other / "private" / "manifest.json").read_bytes(),
                         (self.package / "private" / "manifest.json").read_bytes())
        tasks = read_jsonl(self.package / "reviewers" / "alice" / "tasks.jsonl")
        self.assertEqual(len(tasks), 4)
        self.assertEqual(len({t["domain"] for t in tasks}), 4)
        for task in tasks:
            self.assertEqual(set(task), set(ReviewTask.model_fields))
            for hidden in ("target", "annotation", "tags", "scenario_id", "scenario_family_id", "split", "perturbation"):
                self.assertNotIn(hidden, task)
            scenario = self.scenarios[self.manifest["task_to_scenario"][task["task_id"]]]
            self.assertEqual(scenario.split.value, "validation")
            self.assertEqual(scenario.perturbation.perturbation_type, "complete")
        self.assertTrue(all(row["target"] is None and row["status"] == "pending" for row in read_jsonl(
            self.package / "reviewers" / "alice" / "reviews.jsonl")))

    def test_export_rejects_single_duplicate_and_unsafe_identities(self) -> None:
        for reviewers in (["alice"], ["alice", "alice"], ["alice", "../escape"]):
            with self.assertRaises(ValueError):
                export_review_package(self.source, self.root / "invalid", reviewers, 1)
        self.assertFalse((self.root / "invalid").exists())

    def test_export_rejects_undersized_split_and_existing_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 2 available"):
            export_review_package(self.source, self.root / "large", ["alice", "bob"], 3)
        with self.assertRaises(FileExistsError):
            export_review_package(self.source, self.package, ["alice", "bob"], 1)

    def test_import_reports_disagreements_and_keeps_decisions_pending(self) -> None:
        imported = import_reviews(self.package, self.complete_reviews(True), self.root / "imported")
        summary = json.loads((imported / "review_summary.json").read_text())
        self.assertEqual(summary["statuses"], {"agreement": 3, "disagreement": 1})
        self.assertEqual(summary["exact_pair_agreement"], 0.75)
        disagreement = next(s for s in summary["cases"] if s["status"] == "disagreement")
        self.assertIn("situation_label", disagreement["disagreement_fields"])
        self.assertTrue(all(r["status"] == "pending" for r in read_jsonl(imported / "adjudication_template.jsonl")))

    def test_set_order_and_different_rationale_do_not_create_disagreement(self) -> None:
        paths = self.complete_reviews()
        rows = read_jsonl(paths[1])
        for row in rows:
            for value in row["target"].values():
                if isinstance(value, list):
                    value.reverse()
        write_jsonl(paths[1], rows)
        imported = import_reviews(self.package, paths, self.root / "imported")
        summary = json.loads((imported / "review_summary.json").read_text())
        self.assertEqual(summary["statuses"], {"agreement": 4})

    def test_incomplete_reviews_cannot_be_applied(self) -> None:
        paths = self.complete_reviews()
        imported = import_reviews(self.package, paths[:1], self.root / "imported")
        summary = json.loads((imported / "review_summary.json").read_text())
        self.assertEqual(summary["statuses"], {"needs_reviews": 4})
        with self.assertRaisesRegex(ValueError, "incomplete independent reviews"):
            apply_adjudication(self.source, self.package, imported, self.decide(imported), self.root / "final")
        self.assertFalse((self.root / "final").exists())

    def test_duplicate_unknown_task_and_unassigned_reviewer_are_rejected(self) -> None:
        paths = self.complete_reviews()
        original = read_jsonl(paths[0])
        invalids = [original + original[:1], [{**original[0], "task_id": "unknown"}],
                    [{**original[0], "reviewer_id": "unassigned"}]]
        for rows in invalids:
            write_jsonl(paths[0], rows)
            with self.assertRaises(ValueError):
                import_reviews(self.package, paths, self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())

    def test_ontology_keys_and_action_partitions_are_validated(self) -> None:
        paths = self.complete_reviews()
        original = read_jsonl(paths[0])
        for field, bad in (("relevant_observation_keys", ["invented"]),
                           ("allowed_actions", ["execute_shell"]),
                           ("acceptable_situation_labels", ["invented"])):
            rows = json.loads(json.dumps(original))
            rows[0]["target"][field] = bad
            write_jsonl(paths[0], rows)
            with self.assertRaises(ValueError):
                import_reviews(self.package, paths, self.root / "invalid")

    def test_pending_decisions_and_reviewer_as_adjudicator_are_rejected(self) -> None:
        imported = import_reviews(self.package, self.complete_reviews(), self.root / "imported")
        with self.assertRaisesRegex(ValueError, "adjudication is pending"):
            apply_adjudication(self.source, self.package, imported,
                               imported / "adjudication_template.jsonl", self.root / "final")
        path = self.decide(imported)
        rows = read_jsonl(path)
        rows[0]["adjudicator_id"] = "alice"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "distinct"):
            apply_adjudication(self.source, self.package, imported, path, self.root / "final")

    def test_round_trip_preserves_source_splits_variants_and_audits_fixtures(self) -> None:
        imported = import_reviews(self.package, self.complete_reviews(True), self.root / "imported")
        decisions = self.decide(imported)
        output = apply_adjudication(self.source, self.package, imported, decisions, self.root / "final")
        self.assertEqual(self.source.read_bytes(), self.original)
        reviewed, errors = validate_scenarios(output / "reviewed_scenarios.jsonl", strict=True)
        self.assertFalse(errors)
        self.assertEqual(len(reviewed), len(self.scenarios))
        selected = set(self.manifest["task_to_scenario"].values())
        for scenario in reviewed:
            original = self.scenarios[scenario.scenario_id]
            self.assertEqual(scenario.context, original.context)
            self.assertEqual(scenario.split, original.split)
            self.assertEqual(scenario.scenario_family_id, original.scenario_family_id)
            self.assertEqual(scenario.origin_type, original.origin_type)
            if scenario.scenario_id in selected:
                self.assertEqual(scenario.annotation.annotator_type, "mixed")
                self.assertEqual(scenario.annotation.adjudication_status, "adjudicated")
                self.assertEqual(len(scenario.annotation.review_evidence), 2)
                self.assertIn("not human validation", scenario.annotation.provenance)
            else:
                self.assertEqual(scenario.model_dump(), original.model_dump())
        manifest = json.loads((output / "annotation_manifest.json").read_text())
        self.assertEqual(manifest["output_hash"], file_hash(output / "reviewed_scenarios.jsonl"))
        self.assertTrue(manifest["includes_software_fixtures"])
        audit = read_jsonl(output / "annotation_audit.jsonl")
        self.assertEqual(len(audit), 4)
        self.assertTrue(all("original" in row and "reviews" in row and "decision" in row for row in audit))
        with self.assertRaises(FileExistsError):
            apply_adjudication(self.source, self.package, imported, decisions, output)

    def test_source_and_import_tampering_are_rejected(self) -> None:
        imported = import_reviews(self.package, self.complete_reviews(), self.root / "imported")
        decisions = self.decide(imported)
        self.source.write_bytes(self.original + b"\n")
        with self.assertRaisesRegex(ValueError, "source dataset changed"):
            apply_adjudication(self.source, self.package, imported, decisions, self.root / "final")
        self.source.write_bytes(self.original)
        with (imported / "submitted_reviews.jsonl").open("a") as fh:
            fh.write("\n")
        with self.assertRaisesRegex(ValueError, "review hash mismatch"):
            apply_adjudication(self.source, self.package, imported, decisions, self.root / "final")

    def test_changed_parent_labels_do_not_silently_invalidate_family_variants(self) -> None:
        imported = import_reviews(self.package, self.complete_reviews(), self.root / "imported")
        path = self.decide(imported)
        rows = read_jsonl(path)
        task_id = rows[0]["task_id"]
        parent_id = self.manifest["task_to_scenario"][task_id]
        # Use a second source/export so the hash binds the newly added family variant.
        scenarios = list(self.scenarios.values())
        scenarios.append(perturb_scenario(self.scenarios[parent_id], "stale", "single", 0, 99))
        write_jsonl(self.source, scenarios)
        package = export_review_package(self.source, self.root / "family_pilot", ["alice", "bob"], 1, 19)
        manifest = json.loads((package / "private" / "manifest.json").read_text())
        task_id = next(task for task, sid in manifest["task_to_scenario"].items() if sid == parent_id)
        submissions = []
        for reviewer in ("alice", "bob"):
            submission = package / "reviewers" / reviewer / "reviews.jsonl"
            completed = []
            for row in read_jsonl(submission):
                target = self.scenarios[manifest["task_to_scenario"][row["task_id"]]].target
                completed.append(ReviewRecord(task_id=row["task_id"], reviewer_id=reviewer,
                                              reviewer_type="software_fixture", status="completed",
                                              target=target, ambiguity_level="low"))
            write_jsonl(submission, completed)
            submissions.append(submission)
        imported = import_reviews(package, submissions, self.root / "family_imported")
        decisions = read_jsonl(imported / "adjudication_template.jsonl")
        for decision in decisions:
            decision.update(adjudicator_id="judge", adjudicator_type="software_fixture", status="completed", rationale="Test decision")
            if decision["task_id"] == task_id:
                decision["target"]["abstain_acceptable"] = not decision["target"]["abstain_acceptable"]
        path = self.root / "family_decisions.jsonl"
        write_jsonl(path, decisions)
        with self.assertRaisesRegex(ValueError, "unreviewed family variants"):
            apply_adjudication(self.source, package, imported, path, self.root / "family_final")
        self.assertFalse((self.root / "family_final").exists())

    def test_package_tampering_is_rejected(self) -> None:
        path = self.package / "private" / "manifest.json"
        data = json.loads(path.read_text())
        data["seed"] = 999
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            import_reviews(self.package, [], self.root / "invalid")

    def test_decision_coverage_and_explicit_rationale_are_required(self) -> None:
        imported = import_reviews(self.package, self.complete_reviews(), self.root / "imported")
        path = self.decide(imported)
        rows = read_jsonl(path)
        write_jsonl(path, rows[:-1])
        with self.assertRaisesRegex(ValueError, "every selected task"):
            apply_adjudication(self.source, self.package, imported, path, self.root / "final")
        rows[0]["rationale"] = " "
        with self.assertRaises(ValueError):
            AdjudicationDecision.model_validate(rows[0])

    def test_cli_export_import_and_apply(self) -> None:
        imported = self.root / "cli_imported"
        paths = self.complete_reviews()
        self.assertEqual(main(["annotate", "import", "--package", str(self.package),
                               "--reviews", *(str(p) for p in paths), "--out", str(imported)]), 0)
        decisions = self.decide(imported)
        self.assertEqual(main(["annotate", "apply", "--input", str(self.source),
                               "--package", str(self.package), "--imported", str(imported),
                               "--decisions", str(decisions), "--out", str(self.root / "cli_final")]), 0)
        self.assertTrue((self.root / "cli_final" / "reviewed_scenarios.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
