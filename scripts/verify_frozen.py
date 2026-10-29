"""Verify the per-file SHA256 manifest of an extracted paper pack."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def verify(root: Path) -> int:
    root = root.resolve()
    manifest = json.loads((root / "freeze_manifest.json").read_text())
    errors = []
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(relative + ": missing or outside pack")
            continue
        with path.open("rb") as fh:
            digest = hashlib.file_digest(fh, "sha256").hexdigest()
        if digest != expected:
            errors.append(relative + ": hash mismatch")
    if errors:
        raise ValueError("\n".join(errors))
    return len(manifest["files"])


if __name__ == "__main__":
    print(f"Verified {verify(Path(sys.argv[1]))} frozen files")
