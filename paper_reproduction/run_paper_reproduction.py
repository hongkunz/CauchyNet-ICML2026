"""Run the final paper-ordered CauchyNet reproduction scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "results"
FIGURE_NUMBERS = list(range(1, 15))
TABLE_NUMBERS = list(range(16, 23))
VERIFY_NUMBERS = [3, 4, 5, 6, 7, 10, 11, *TABLE_NUMBERS]


def resolve(number: int) -> str:
    matches = sorted(ROOT.glob(f"{number:02d}_*.py"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one script for {number:02d}, found {matches}")
    return matches[0].name


def scripts_for(mode: str) -> list[tuple[str, list[str]]]:
    if mode == "verify":
        return [(resolve(number), []) for number in VERIFY_NUMBERS]
    if mode == "figures":
        return [(resolve(number), []) for number in FIGURE_NUMBERS]
    if mode == "tables":
        return [(resolve(number), []) for number in TABLE_NUMBERS]
    if mode == "all":
        return [(resolve(number), []) for number in FIGURE_NUMBERS + TABLE_NUMBERS]
    return [(resolve(23), ["--mode", "sweep"])]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_final_inventory() -> None:
    figure_dir = ROOT / "outputs" / "figures"
    figure_names = {
            1: "rational_spike",
            2: "cauchy_activation",
            3: "theory_roadmap",
            4: "elliptical_poles",
            5: "gap_filling",
            6: "architecture_summary",
            7: "gap_filling_curves",
            8: "missing_disk",
            9: "2d_surface",
            10: "m4_decomposition",
            11: "targets",
            12: "m4_benchmark",
            13: "predictions_residuals",
            14: "imaginary_penalty",
    }
    expected = {
        figure_dir / f"figure{number:02d}_{name}.pdf"
        for number, name in figure_names.items()
    }
    generated_previews = {
        figure_dir / f"figure{number:02d}_{name}.png"
        for number, name in figure_names.items()
    }
    missing = sorted(path for path in expected if not path.is_file())
    allowed = expected | generated_previews
    unexpected = sorted(
        path for path in figure_dir.iterdir() if path.is_file() and path not in allowed
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Final figure inventory mismatch: missing={missing}, unexpected={unexpected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("verify", "figures", "tables", "all", "uci-sweep"),
        default="verify",
    )
    args = parser.parse_args()
    check_final_inventory()

    rows = []
    for script, script_args in scripts_for(args.mode):
        command = [sys.executable, str(ROOT / script), *script_args]
        print(f"\n=== {script} {' '.join(script_args)} ===", flush=True)
        started = time.time()
        process = subprocess.run(command, cwd=ROOT, text=True)
        rows.append(
            {
                "script": script,
                "arguments": script_args,
                "return_code": process.returncode,
                "elapsed_seconds": time.time() - started,
            }
        )
        if process.returncode != 0:
            break

    files = sorted(
        path
        for path in (ROOT / "outputs").rglob("*")
        if path.is_file() and not path.name.startswith("reproduction_manifest_")
    )
    manifest = {
        "mode": args.mode,
        "status": "pass" if all(row["return_code"] == 0 for row in rows) else "fail",
        "runs": rows,
        "output_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / f"reproduction_manifest_{args.mode.replace('-', '_')}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nSaved {manifest_path}")
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
