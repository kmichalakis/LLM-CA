"""Native review forms backed by the annotation JSONL contracts."""

from __future__ import annotations

import json
import os
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .annotation import apply_adjudication, export_review_package, import_reviews, validate_judgment
from .annotation_models import AdjudicationDecision, ReviewRecord, ReviewTask
from .io import read_jsonl, write_jsonl
from .models import Target


def save_review_records(path: Path, records: list[ReviewRecord | AdjudicationDecision]) -> None:
    """Replace an editable submission only after all records serialize successfully."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".review-", suffix=".jsonl")
    os.close(fd)
    try:
        write_jsonl(temporary, records)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ReviewEditor(tk.Toplevel):
    def __init__(self, parent: tk.Misc, path: str | Path) -> None:
        super().__init__(parent)
        self.path = Path(path)
        raw = read_jsonl(self.path)
        if not raw:
            self.destroy()
            raise ValueError("submission contains no records")
        self.adjudicating = "adjudicator_id" in raw[0]
        model = AdjudicationDecision if self.adjudicating else ReviewRecord
        self.records: list[Any] = [model.model_validate(r) for r in raw]
        task_file = "adjudication_tasks.jsonl" if self.adjudicating else "tasks.jsonl"
        task_rows = read_jsonl(self.path.parent / task_file)
        self.task_data = {r["task_id"]: r for r in task_rows}
        self.tasks = {key: ReviewTask.model_validate({
            field: value for field, value in row.items() if field in ReviewTask.model_fields
        }) for key, row in self.task_data.items()}
        if len({r.task_id for r in self.records}) != len(self.records) or any(
            r.task_id not in self.tasks for r in self.records
        ):
            self.destroy()
            raise ValueError("submission contains duplicate or unknown tasks")
        self.current = 0
        self.dirty = False
        self.loading = True
        self.title("LLM-CA Adjudication" if self.adjudicating else "LLM-CA Independent Review")
        self.geometry("1120x850")
        self.minsize(850, 650)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._load(0)

    def _build(self) -> None:
        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = ttk.Frame(pane, width=250)
        right = ttk.Frame(pane)
        pane.add(left, weight=1)
        pane.add(right, weight=4)
        self.task_list = tk.Listbox(left, exportselection=False, width=30)
        self.task_list.pack(fill=tk.BOTH, expand=True)
        self.task_list.bind("<<ListboxSelect>>", self._select)
        self._refresh_labels()
        self.identity = tk.StringVar()
        self.actor_type = tk.StringVar(value="human")
        self.label = tk.StringVar()
        self.ambiguity = tk.StringVar(value="low")
        self.abstain = tk.BooleanVar()
        for variable in (self.identity, self.actor_type, self.label, self.ambiguity, self.abstain):
            variable.trace_add("write", self._mark_dirty)
        actor = ttk.Frame(right)
        actor.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(actor, text="Adjudicator ID" if self.adjudicating else "Reviewer ID").pack(side=tk.LEFT)
        ttk.Entry(actor, textvariable=self.identity, state="normal" if self.adjudicating else "readonly", width=22).pack(side=tk.LEFT, padx=8)
        ttk.Combobox(actor, textvariable=self.actor_type, values=["human", "software_fixture", "automated_model"], state="readonly", width=17).pack(side=tk.LEFT)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        context = ttk.Frame(self.notebook)
        judgment = ttk.Frame(self.notebook)
        canvas = tk.Canvas(judgment, highlightthickness=0)
        form_scroll = ttk.Scrollbar(judgment, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=form_scroll.set)
        form_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True)
        self.form = ttk.Frame(canvas, padding=12)
        form_window = canvas.create_window((0, 0), window=self.form, anchor="nw")
        self.form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(form_window, width=e.width))
        self.notebook.add(context, text="Observations")
        self.notebook.add(judgment, text="Judgment")
        self.context = tk.Text(context, wrap=tk.WORD, state="disabled")
        scroll = ttk.Scrollbar(context, command=self.context.yview)
        self.context.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.context.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(self.form)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Situation").grid(row=0, column=0, sticky="w")
        self.label_combo = ttk.Combobox(top, textvariable=self.label, state="readonly", width=28)
        self.label_combo.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(top, text="Ambiguity").grid(row=1, column=0, sticky="w")
        ttk.Combobox(top, textvariable=self.ambiguity, values=["low", "medium", "high"], state="readonly", width=28).grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        ttk.Checkbutton(top, text="Abstention acceptable", variable=self.abstain).grid(row=2, column=1, sticky="w", padx=8)
        self.acceptable_frame = ttk.LabelFrame(self.form, text="Acceptable situation labels", padding=6)
        self.acceptable_frame.pack(fill=tk.X, pady=8)
        self.relevance_frame = ttk.LabelFrame(self.form, text="Relevant observations", padding=6)
        self.relevance_frame.pack(fill=tk.X, pady=8)
        self.action_frame = ttk.LabelFrame(self.form, text="Actions", padding=6)
        self.action_frame.pack(fill=tk.X, pady=8)
        ttk.Label(self.form, text="Evidence rationale").pack(anchor="w")
        self.rationale = tk.Text(self.form, height=4, wrap=tk.WORD)
        self.rationale.pack(fill=tk.X)
        ttk.Label(self.form, text="Decision rationale" if self.adjudicating else "Reviewer notes").pack(anchor="w", pady=(8, 0))
        self.notes = tk.Text(self.form, height=3, wrap=tk.WORD)
        self.notes.pack(fill=tk.X)
        for widget in (self.rationale, self.notes):
            widget.bind("<KeyRelease>", self._mark_dirty)
        footer = ttk.Frame(right)
        footer.pack(fill=tk.X, pady=10)
        self.status = tk.StringVar()
        ttk.Label(footer, textvariable=self.status).pack(side=tk.LEFT)
        ttk.Button(footer, text="Save Completed", command=self._save).pack(side=tk.RIGHT)
        self.acceptable: dict[str, tk.BooleanVar] = {}
        self.relevant: dict[str, tk.BooleanVar] = {}
        self.allowed: dict[str, tk.BooleanVar] = {}
        self.preferred: dict[str, tk.BooleanVar] = {}

    def _mark_dirty(self, *_args: Any) -> None:
        if not self.loading:
            self.dirty = True

    def _refresh_labels(self) -> None:
        self.task_list.delete(0, tk.END)
        for index, record in enumerate(self.records, 1):
            task = self.tasks[record.task_id]
            self.task_list.insert(tk.END, f"{index}. {task.domain} [{record.status}]")
        self.task_list.selection_set(self.current)

    def _checks(self, frame: ttk.Frame, keys: list[str], selected: list[str]) -> dict[str, tk.BooleanVar]:
        for child in frame.winfo_children():
            child.destroy()
        variables = {}
        for index, key in enumerate(keys):
            variable = tk.BooleanVar(value=key in selected)
            variable.trace_add("write", self._mark_dirty)
            ttk.Checkbutton(frame, text=key, variable=variable).grid(row=index // 2, column=index % 2, sticky="w", padx=6)
            variables[key] = variable
        return variables

    def _load(self, index: int) -> None:
        self.loading = True
        self.current = index
        record = self.records[index]
        task = self.tasks[record.task_id]
        target = record.target
        actor_id = record.adjudicator_id if self.adjudicating else record.reviewer_id
        self.identity.set(actor_id or self.identity.get())
        self.actor_type.set(record.adjudicator_type if self.adjudicating else record.reviewer_type)
        self.label_combo.configure(values=task.situation_labels)
        self.label.set(target.situation_label if target else "")
        self.ambiguity.set(record.ambiguity_level or "low")
        self.abstain.set(target.abstain_acceptable if target else False)
        self.acceptable = self._checks(self.acceptable_frame, task.situation_labels,
                                       target.acceptable_situation_labels if target else [])
        self.relevant = self._checks(self.relevance_frame, [o.key for o in task.context],
                                     target.relevant_observation_keys if target else [])
        for child in self.action_frame.winfo_children():
            child.destroy()
        for col, name in enumerate(("Action", "Allowed", "Preferred")):
            ttk.Label(self.action_frame, text=name).grid(row=0, column=col, sticky="w", padx=6)
        self.allowed, self.preferred = {}, {}
        for row, action in enumerate(task.action_vocabulary, 1):
            ttk.Label(self.action_frame, text=action).grid(row=row, column=0, sticky="w", padx=6)
            for col, mapping, selected in ((1, self.allowed, target.allowed_actions if target else []),
                                            (2, self.preferred, target.preferred_actions if target else [])):
                variable = tk.BooleanVar(value=action in selected)
                variable.trace_add("write", self._mark_dirty)
                mapping[action] = variable
                ttk.Checkbutton(self.action_frame, variable=variable).grid(row=row, column=col)
        self.rationale.delete("1.0", tk.END)
        self.rationale.insert("1.0", target.rationale_reference if target else "")
        self.notes.delete("1.0", tk.END)
        self.notes.insert("1.0", record.rationale if self.adjudicating else record.notes)
        context = json.dumps(task.model_dump(mode="json"), indent=2, ensure_ascii=False)
        if self.adjudicating:
            context += "\n\nIndependent reviews\n" + json.dumps(self.task_data[task.task_id].get("reviews", []), indent=2, ensure_ascii=False)
        self.context.configure(state="normal")
        self.context.delete("1.0", tk.END)
        self.context.insert("1.0", context)
        self.context.configure(state="disabled")
        self.status.set(f"{index + 1}/{len(self.records)}: {record.status}")
        self.task_list.selection_clear(0, tk.END)
        self.task_list.selection_set(index)
        self.loading = False
        self.dirty = False

    def _select(self, _event: Any) -> None:
        selection = self.task_list.curselection()
        if not selection or selection[0] == self.current:
            return
        if self.dirty and not messagebox.askyesno("Unsaved judgment", "Discard unsaved changes?", parent=self):
            self.task_list.selection_clear(0, tk.END)
            self.task_list.selection_set(self.current)
            return
        self._load(selection[0])

    def _save(self) -> None:
        try:
            record = self.records[self.current]
            task = self.tasks[record.task_id]
            selected = lambda mapping: [key for key, value in mapping.items() if value.get()]
            allowed = selected(self.allowed)
            acceptable = sorted(set(selected(self.acceptable) + [self.label.get()]))
            target = Target(
                situation_label=self.label.get(), acceptable_situation_labels=acceptable,
                relevant_observation_keys=selected(self.relevant), allowed_actions=allowed,
                preferred_actions=selected(self.preferred),
                forbidden_actions=[action for action in task.action_vocabulary if action not in allowed],
                abstain_acceptable=self.abstain.get(), rationale_reference=self.rationale.get("1.0", "end-1c"),
            )
            validate_judgment(task, target)
            data = {**record.model_dump(mode="json"), "target": target.model_dump(mode="json"),
                    "ambiguity_level": self.ambiguity.get(), "status": "completed"}
            if self.adjudicating:
                data.update(adjudicator_id=self.identity.get().strip(), adjudicator_type=self.actor_type.get(),
                            rationale=self.notes.get("1.0", "end-1c"))
                completed = AdjudicationDecision.model_validate(data)
            else:
                data.update(reviewer_type=self.actor_type.get(), notes=self.notes.get("1.0", "end-1c"))
                completed = ReviewRecord.model_validate(data)
            updated = self.records.copy()
            updated[self.current] = completed
            save_review_records(self.path, updated)
            self.records = updated
            self.dirty = False
            self._refresh_labels()
            self.status.set(f"{self.current + 1}/{len(self.records)}: completed, saved")
        except Exception as exc:
            messagebox.showerror("Cannot save judgment", str(exc), parent=self)

    def _close(self) -> None:
        if not self.dirty or messagebox.askyesno("Unsaved judgment", "Discard unsaved changes and close?", parent=self):
            self.destroy()


class AnnotationWorkspace(tk.Toplevel):
    def __init__(self, parent: tk.Misc, source: str, seed: int) -> None:
        super().__init__(parent)
        self.title("LLM-CA Annotation Workflow")
        self.geometry("780x430")
        self.source = tk.StringVar(value=source)
        self.package = tk.StringVar(value="annotations/pilot")
        self.reviewers = tk.StringVar(value="reviewer_a reviewer_b")
        self.per_domain = tk.IntVar(value=10)
        self.split = tk.StringVar(value="validation")
        self.seed = seed
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        for row, (label, variable) in enumerate((
            ("Source dataset", self.source), ("Package folder", self.package), ("Reviewer IDs", self.reviewers),
        )):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Label(frame, text="Cases per domain").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1, to=10000, textvariable=self.per_domain, width=8).grid(row=3, column=1, sticky="w", padx=8)
        ttk.Label(frame, text="Split").grid(row=4, column=0, sticky="w", pady=8)
        ttk.Combobox(frame, textvariable=self.split, values=["train", "validation", "test"], state="readonly", width=16).grid(row=4, column=1, sticky="w", padx=8)
        commands = ttk.Frame(frame)
        commands.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        for label, callback in (("Export Pilot", self._export), ("Open Review", self._review),
                                ("Import Reviews", self._import), ("Open Adjudication", self._adjudicate),
                                ("Apply Decisions", self._apply)):
            ttk.Button(commands, text=label, command=lambda cb=callback: self._execute(cb)).pack(side=tk.LEFT, padx=3)
        self.result = tk.Text(frame, height=8, wrap=tk.WORD)
        self.result.grid(row=6, column=0, columnspan=2, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(6, weight=1)

    def _execute(self, callback: Any) -> None:
        try:
            result = callback()
            if result:
                self.result.insert(tk.END, str(result) + "\n")
        except Exception as exc:
            messagebox.showerror("Annotation workflow", str(exc), parent=self)

    def _export(self) -> Path:
        return export_review_package(self.source.get(), self.package.get(), self.reviewers.get().split(),
                                     self.per_domain.get(), self.seed, self.split.get())

    def _review(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Open review submission", initialdir=self.package.get(), filetypes=[("JSONL", "*.jsonl")])
        if path:
            ReviewEditor(self, path)

    def _import(self) -> Path | None:
        paths = filedialog.askopenfilenames(parent=self, title="Select independent review files", initialdir=self.package.get(), filetypes=[("JSONL", "*.jsonl")])
        if not paths:
            return None
        out = filedialog.asksaveasfilename(parent=self, title="New import folder", initialdir=self.package.get(), initialfile="imported")
        return import_reviews(self.package.get(), list(paths), out) if out else None

    def _adjudicate(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Open adjudication template", initialdir=self.package.get(), filetypes=[("JSONL", "*.jsonl")])
        if path:
            ReviewEditor(self, path)

    def _apply(self) -> Path | None:
        imported = filedialog.askdirectory(parent=self, title="Select import folder", initialdir=self.package.get())
        if not imported:
            return None
        decisions = filedialog.askopenfilename(parent=self, title="Completed adjudication decisions", initialdir=imported, filetypes=[("JSONL", "*.jsonl")])
        if not decisions:
            return None
        out = filedialog.asksaveasfilename(parent=self, title="New reviewed dataset folder", initialdir=self.package.get(), initialfile="final")
        return apply_adjudication(self.source.get(), self.package.get(), imported, decisions, out) if out else None
