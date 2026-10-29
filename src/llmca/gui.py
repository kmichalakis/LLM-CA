"""Tkinter GUI for offline LLM-CA simulations."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import load_config
from .evaluation import evaluate_run
from .execution import run_experiment
from .generation import generate_to_file
from .reporting import generate_report
from .validation import validate_scenarios


class LLMCAApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LLM-CA Simulation Benchmark")
        self.geometry("920x640")
        self.events: queue.Queue[str] = queue.Queue()
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X)

        self.config_var = tk.StringVar(value="configs/benchmark.yaml")
        self.experiment_var = tk.StringVar(value="configs/experiment_mock.yaml")
        self.dataset_var = tk.StringVar(value="data/scenarios/base.jsonl")
        self.run_var = tk.StringVar(value="runs/demo_mock")
        self.seed_var = tk.IntVar(value=42)
        self.limit_var = tk.IntVar(value=40)

        self._path_row(controls, "Benchmark config", self.config_var, 0)
        self._path_row(controls, "Experiment config", self.experiment_var, 1)
        self._path_row(controls, "Dataset", self.dataset_var, 2)
        self._path_row(controls, "Run folder", self.run_var, 3)

        numeric = ttk.Frame(controls)
        numeric.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Label(numeric, text="Seed").pack(side=tk.LEFT)
        ttk.Spinbox(numeric, from_=0, to=999999, textvariable=self.seed_var, width=8).pack(side=tk.LEFT, padx=8)
        ttk.Label(numeric, text="Run limit").pack(side=tk.LEFT)
        ttk.Spinbox(numeric, from_=1, to=100000, textvariable=self.limit_var, width=8).pack(side=tk.LEFT, padx=8)

        actions = ttk.Frame(outer)
        actions.pack(fill=tk.X, pady=(10, 8))
        for text, command in [
            ("Generate", self.generate),
            ("Validate", self.validate),
            ("Dry Run", self.dry_run),
            ("Run Mock", self.run_mock),
            ("Evaluate", self.evaluate),
            ("Report", self.report),
            ("Annotations", self.annotations),
        ]:
            ttk.Button(actions, text=text, command=command).pack(side=tk.LEFT, padx=4)

        self.log = tk.Text(outer, wrap=tk.WORD, height=24)
        self.log.pack(fill=tk.BOTH, expand=True)
        self._log("Ready. Use the mock workflow first; it is offline and reproducible.")

    def _path_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label, width=18).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(variable)).grid(row=row, column=2, padx=4)
        parent.columnconfigure(1, weight=1)

    def _browse(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename() or filedialog.askdirectory()
        if path:
            variable.set(path)

    def _log(self, text: str) -> None:
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def _poll(self) -> None:
        while True:
            try:
                self._log(self.events.get_nowait())
            except queue.Empty:
                break
        self.after(100, self._poll)

    def _background(self, name: str, func) -> None:  # noqa: ANN001
        def run() -> None:
            self.events.put(f"Starting {name}...")
            try:
                result = func()
                self.events.put(f"Finished {name}: {result}")
            except Exception as exc:  # noqa: BLE001
                self.events.put(f"Failed {name}: {exc}")
                messagebox.showerror("LLM-CA", str(exc))

        threading.Thread(target=run, daemon=True).start()

    def generate(self) -> None:
        self._background(
            "generation",
            lambda: f"{generate_to_file(load_config(self.config_var.get()), self.seed_var.get(), self.dataset_var.get())} scenarios",
        )

    def validate(self) -> None:
        def work() -> str:
            scenarios, errors = validate_scenarios(self.dataset_var.get(), strict=True)
            if errors:
                return "validation errors:\n" + "\n".join(errors[:20])
            return f"{len(scenarios)} valid scenarios"

        self._background("validation", work)

    def dry_run(self) -> None:
        self._background(
            "dry run",
            lambda: run_experiment(load_config(self.experiment_var.get()), seed=self.seed_var.get(), dry_run=True, limit=self.limit_var.get()),
        )

    def run_mock(self) -> None:
        self._background(
            "mock run",
            lambda: run_experiment(load_config(self.experiment_var.get()), seed=self.seed_var.get(), resume=True, limit=self.limit_var.get()),
        )

    def evaluate(self) -> None:
        self._background("evaluation", lambda: f"{len(evaluate_run(self.run_var.get())[0])} predictions")

    def report(self) -> None:
        self._background("report", lambda: ", ".join(str(p) for p in generate_report(self.run_var.get(), ["md", "html"])))

    def annotations(self) -> None:
        from .annotation_gui import AnnotationWorkspace

        AnnotationWorkspace(self, self.dataset_var.get(), self.seed_var.get())


def main() -> None:
    app = LLMCAApp()
    app.mainloop()


if __name__ == "__main__":
    main()
