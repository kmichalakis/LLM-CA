# Contributing

Use Python 3.11 or newer and work from the repository root:

```sh
python -m venv .venv
python -m pip install -e '.[reports,validation]'
python -m unittest discover -s tests -v
```

Activate the virtual environment before installation, or use its Python
executable explicitly. On Linux, install the system's Tkinter package if you
intend to use the optional desktop GUI. The automated suite is headless; the
desktop smoke test in `tests/gui_annotation_smoke.py` is separate and opt-in.

Keep changes focused and add regression tests for behavior changes. Include
the command, Python version, seed, configuration, and a minimal synthetic
example when reporting a problem. Do not include credentials, personal data,
unpublished paper drafts, live-provider responses, or local environment files.

Real-provider calls must remain opt-in. Do not introduce provider credentials
into the test workflow. Distinguish synthetic reference labels, automated
review judgments, and independent human annotations in both code and reports.
Do not describe simulation agreement as a deployed-system or LLM benefit.

The project uses the MIT license in `LICENSE`. Preserve applicable notices and
review the licenses of any proposed third-party code or datasets separately.

To build the deliberately limited public source archive:

```sh
python scripts/build_open_source_release.py
```

The builder uses an explicit file allowlist and writes a SHA256 manifest.
Review the resulting archive before making a new public release.
