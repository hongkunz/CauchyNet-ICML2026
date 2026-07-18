"""Write SHA-256 checksums for every shipped release file."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "SOURCE_MANIFEST_SHA256.txt"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}
EXCLUDED_NAMES = {OUTPUT.name, ".DS_Store"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".log"}
    )
    OUTPUT.write_text(
        "\n".join(f"{digest(path)}  {path.relative_to(ROOT)}" for path in files)
        + "\n"
    )
    print(f"Wrote {OUTPUT} with {len(files)} files")


if __name__ == "__main__":
    main()
