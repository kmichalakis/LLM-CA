"""JSONL and filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, BaseModel):
                payload = row.model_dump(mode="json")
            else:
                payload = row
            fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, rows: Iterable[BaseModel | dict]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8") as fh:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(row)
    return rows
