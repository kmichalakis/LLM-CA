"""Record exact installed dependency versions without embedding an editable path."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path


def main() -> None:
    packages = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()
                if d.metadata["Name"].lower() not in {"llmca", "pip"}}
    header = "# Frozen environment for the LLM-CA process demonstration.\n# Python 3.12.14; macOS arm64. Other platforms resolve their matching wheels.\n"
    path = Path(__file__).resolve().parents[1] / "requirements.lock"
    path.write_text(header + "".join(f"{name}=={version}\n" for name, version in sorted(packages.items())), encoding="utf-8")
    print(f"Pinned {len(packages)} dependency versions in {path}")


if __name__ == "__main__":
    main()
