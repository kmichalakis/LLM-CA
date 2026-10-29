"""Configuration loading without mandatory third-party YAML dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not parseable by the built-in loader. Use JSON syntax in .yaml files "
            "or install PyYAML and extend llmca.config."
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("configuration root must be an object")
    return data


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
