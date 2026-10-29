"""Opt-in native GUI smoke check; requires a desktop session."""

from __future__ import annotations

import json
import tempfile
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

from llmca.annotation import export_review_package, import_reviews
from llmca.annotation_gui import ReviewEditor
from llmca.annotation_models import ReviewRecord
from llmca.generation import generate_base_scenarios
from llmca.io import read_jsonl, write_jsonl


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        scenarios = generate_base_scenarios(10)
        by_id = {s.scenario_id: s for s in scenarios}
        source = folder / "source.jsonl"
        write_jsonl(source, scenarios)
        package = export_review_package(source, folder / "pilot", ["alice", "bob"], 1)
        manifest = json.loads((package / "private" / "manifest.json").read_text())
        paths = [package / "reviewers" / reviewer / "reviews.jsonl" for reviewer in ("alice", "bob")]
        root = tk.Tk()
        root.withdraw()
        try:
            def fail_dialog(_title: str, text: str, **_kwargs) -> None:
                raise AssertionError(text)

            with patch("llmca.annotation_gui.messagebox.showerror", side_effect=fail_dialog):
                editor = ReviewEditor(root, paths[0])
                editor.geometry("850x650")
                root.update()
                for index, record in enumerate(editor.records):
                    editor._load(index)
                    scenario = by_id[manifest["task_to_scenario"][record.task_id]]
                    target = scenario.target
                    editor.notebook.select(1)
                    editor.actor_type.set("software_fixture")
                    editor.label.set(target.situation_label)
                    editor.abstain.set(target.abstain_acceptable)
                    for mapping, selected in ((editor.acceptable, target.acceptable_situation_labels),
                                              (editor.relevant, target.relevant_observation_keys),
                                              (editor.allowed, target.allowed_actions),
                                              (editor.preferred, target.preferred_actions)):
                        for key, variable in mapping.items():
                            variable.set(key in selected)
                    editor.rationale.delete("1.0", tk.END)
                    editor.rationale.insert("1.0", "Software GUI fixture; not a human review")
                    editor._save()
                    root.update()
                    assert editor.rationale.winfo_width() <= editor.form.winfo_width()
                assert all(r["status"] == "completed" for r in read_jsonl(paths[0]))
                editor.destroy()
                rows = []
                for row in read_jsonl(paths[1]):
                    scenario = by_id[manifest["task_to_scenario"][row["task_id"]]]
                    rows.append(ReviewRecord(
                        task_id=row["task_id"], reviewer_id="bob", reviewer_type="software_fixture",
                        status="completed", target=scenario.target, ambiguity_level="low",
                    ))
                write_jsonl(paths[1], rows)
                imported = import_reviews(package, paths, folder / "imported")
                adjudicator = ReviewEditor(root, imported / "adjudication_template.jsonl")
                adjudicator.identity.set("adjudicator")
                adjudicator.actor_type.set("software_fixture")
                adjudicator.notes.insert("1.0", "Explicit software adjudication fixture")
                adjudicator._save()
                root.update()
                saved = read_jsonl(imported / "adjudication_template.jsonl")[0]
                assert saved["status"] == "completed"
                assert saved["adjudicator_type"] == "software_fixture"
                assert saved["adjudicator_id"] == "adjudicator"
                adjudicator.destroy()
            print("GUI smoke passed: four reviews and an adjudication saved; small-window form fits.")
        finally:
            root.destroy()


if __name__ == "__main__":
    main()
