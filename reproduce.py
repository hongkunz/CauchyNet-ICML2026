"""Single entry point for the final CauchyNet paper package."""

from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("verify", "figures", "tables", "all", "uci-sweep"),
        default="verify",
    )
    args = parser.parse_args()

    if not compileall.compile_dir(ROOT, quiet=1):
        raise SystemExit("Python compilation failed")

    subprocess.run(
        [
            sys.executable,
            "run_paper_reproduction.py",
            "--mode",
            args.mode,
        ],
        cwd=ROOT / "paper_reproduction",
        check=True,
    )
    print(f"CauchyNet reproduction mode {args.mode!r} completed successfully.")


if __name__ == "__main__":
    main()
