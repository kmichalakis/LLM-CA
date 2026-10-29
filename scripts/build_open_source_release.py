"""Build a source-only release, excluding local research inputs and outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
import zipfile
from pathlib import Path


ROOT_FILES = (
    "README.md", "LICENSE", "CITATION.cff", "CONTRIBUTING.md", "pyproject.toml",
    "requirements.lock", ".gitignore", ".env.example",
)
SOURCE_FOLDERS = (
    ("src/llmca", ".py"), ("tests", ".py"), ("scripts", ".py"),
    ("configs", ".yaml"), ("schemas", ".json"), ("prompts", ".txt"),
    ("docs", ".md"), (".github/workflows", ".yml"),
)
TOKEN_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def source_files(root: Path) -> list[Path]:
    root = root.resolve()
    candidates = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for folder, suffix in SOURCE_FOLDERS:
        candidates.extend((root / folder).rglob("*" + suffix))
    files = []
    for path in candidates:
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.is_symlink() or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Source path escapes the project: {relative}")
        files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def build_release(root: Path, destination: Path) -> tuple[Path, Path]:
    root = root.resolve()
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if not re.fullmatch(r"[A-Za-z0-9.+-]+", version):
        raise ValueError("Version is not suitable for an archive filename")
    files = source_files(root)
    hashes = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        if TOKEN_PATTERN.search(content.decode("utf-8")):
            raise ValueError(f"Possible credential in {relative}; review before publishing")
        hashes[relative] = hashlib.sha256(content).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"LLM-CA-{version}-source.zip"
    manifest = destination / f"LLM-CA-{version}-source.manifest.json"
    prefix = f"LLM-CA-{version}/"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(prefix + relative, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive integrity check failed")
    manifest.write_text(json.dumps({
        "name": "LLM-CA", "version": version, "license": "MIT",
        "scope": "Source only; research inputs, annotations, generated reports and manuscript excluded.",
        "credential_check": "Common token/key patterns only; not a comprehensive security audit.",
        "files": hashes, "file_count": len(files),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return archive, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--out", type=Path, default=root / "releases")
    args = parser.parse_args()
    archive, manifest = build_release(root, args.out)
    print(archive)
    print(manifest)


if __name__ == "__main__":
    main()
