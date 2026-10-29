"""Capture a real native GUI smoke check when a desktop session is available."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run([sys.executable, str(root / "tests/gui_annotation_smoke.py")], cwd=root,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "gui_smoke.log").write_text(completed.stdout, encoding="utf-8")
    summary = {"successful": completed.returncode == 0, "returncode": completed.returncode,
               "scope": "native annotation forms; four review saves and one adjudication save in temporary data"}
    (out / "gui_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(completed.returncode)
