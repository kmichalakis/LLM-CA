from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llmca.annotation import apply_adjudication, export_review_package, import_reviews
from llmca.annotation_models import ReviewRecord
from llmca.automatic_annotation import materialize_model_reviews
from llmca.generation import generate_base_scenarios
from llmca.io import read_jsonl, write_jsonl
from llmca.models import Scenario


class AutomaticAnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.jsonl"
        write_jsonl(self.source, generate_base_scenarios(10))
        self.package = export_review_package(self.source, self.root / "pilot", ["a", "b"], 1)
        manifest = json.loads((self.package / "private" / "manifest.json").read_text())
        tasks = read_jsonl(self.package / "reviewers" / "a" / "tasks.jsonl")
        judgments = [[t["task_id"], t["situation_labels"][-1], [t["situation_labels"][-1]],
                      [t["context"][0]["key"]], ["no_action"], ["no_action"], True,
                      "high", "Test fixture judgment; not a human review"] for t in tasks]
        self.batch = {
            "schema_version": "model-judgments-1.0", "package_id": manifest["package_id"],
            "input_tasks_hash": manifest["tasks_hash"], "engine": "test model fixture",
            "model_version": None, "same_session_passes": True,
            "generator_rules_visible_to_session": True,
            "passes": {r: {"method": f"test pass {r}", "judgments": judgments} for r in ("a", "b")},
        }
        self.plan = self.root / "judgments.json"
        self.plan.write_text(json.dumps(self.batch))

    def test_materializer_uses_only_blinded_tasks_and_preserves_original_forms(self) -> None:
        original = (self.package / "reviewers" / "a" / "reviews.jsonl").read_bytes()
        self.source.unlink()
        output = materialize_model_reviews(self.package, self.plan, self.root / "automatic")
        rows = read_jsonl(output / "a" / "reviews.jsonl")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["status"] == "completed" and r["reviewer_type"] == "automated_model" for r in rows))
        self.assertTrue(all(r["automation"]["same_session_passes"] for r in rows))
        self.assertTrue(all(r["automation"]["model_version"] is None for r in rows))
        self.assertEqual((self.package / "reviewers" / "a" / "reviews.jsonl").read_bytes(), original)
        imported = import_reviews(self.package, [output / r / "reviews.jsonl" for r in ("a", "b")], self.root / "imported")
        summary = json.loads((imported / "review_summary.json").read_text())
        self.assertTrue(summary["includes_nonhuman_reviews"])
        self.assertEqual(summary["completed_reviews"], 8)

    def test_provenance_is_required_and_cannot_be_reclassified_as_human(self) -> None:
        output = materialize_model_reviews(self.package, self.plan, self.root / "automatic")
        row = read_jsonl(output / "a" / "reviews.jsonl")[0]
        with self.assertRaisesRegex(ValueError, "provenance"):
            ReviewRecord.model_validate({**row, "automation": None})
        with self.assertRaisesRegex(ValueError, "human review"):
            ReviewRecord.model_validate({**row, "reviewer_type": "human"})

    def test_wrong_package_incomplete_and_duplicate_judgments_are_rejected(self) -> None:
        cases = []
        cases.append({**self.batch, "package_id": "wrong"})
        for rows in (self.batch["passes"]["a"]["judgments"][:-1],
                     self.batch["passes"]["a"]["judgments"] * 2):
            modified = json.loads(json.dumps(self.batch))
            modified["passes"]["a"]["judgments"] = rows
            cases.append(modified)
        for value in cases:
            self.plan.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                materialize_model_reviews(self.package, self.plan, self.root / "invalid")
            self.assertFalse((self.root / "invalid").exists())

    def test_applied_model_reviews_are_not_mislabeled_as_test_fixtures(self) -> None:
        output = materialize_model_reviews(self.package, self.plan, self.root / "automatic")
        imported = import_reviews(self.package, [output / r / "reviews.jsonl" for r in ("a", "b")], self.root / "imported")
        decisions = read_jsonl(imported / "adjudication_template.jsonl")
        for decision in decisions:
            decision.update(adjudicator_id="judge", status="completed", rationale="Explicit test decision")
        path = self.root / "decisions.jsonl"
        write_jsonl(path, decisions)
        final = apply_adjudication(self.source, self.package, imported, path, self.root / "final")
        manifest = json.loads((final / "annotation_manifest.json").read_text())
        self.assertTrue(manifest["includes_automated_model_annotations"])
        self.assertFalse(manifest["includes_software_fixtures"])
        reviewed = [Scenario.model_validate(r) for r in read_jsonl(final / "reviewed_scenarios.jsonl")]
        changed = [s for s in reviewed if s.annotation.adjudication_status == "adjudicated"]
        self.assertEqual(len(changed), 4)
        self.assertTrue(all(s.annotation.annotator_type == "mixed" for s in changed))
        self.assertTrue(all("automated model" in s.annotation.provenance for s in changed))


if __name__ == "__main__":
    unittest.main()
