"""Reproduce Figure 5 from the authoritative saved final-result files.

Figure 5 already has a canonical release implementation:
``code_release/experiments/plot_main_figure5.py``. The reproducibility audit
regenerated the archived PNG with identical RGB pixels. This adapter runs that
exact plotting code while redirecting outputs into ``paper_reproduction`` and
records checksums for the JSON/NPZ inputs.

Use ``--retrain`` only to rebuild the representative reconstruction panel.
The paper figure normally uses the shipped, deterministic reconstruction NPZ.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT.parent
EXPERIMENT_DIR = RELEASE_ROOT / "experiments"
SOURCE_SCRIPT = EXPERIMENT_DIR / "plot_main_figure5.py"
SOURCE_RESULTS = EXPERIMENT_DIR / "results"
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical_module():
    sys.path.insert(0, str(EXPERIMENT_DIR))
    specification = importlib.util.spec_from_file_location(
        "cauchynet_canonical_figure5", SOURCE_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    module = load_canonical_module()

    module.RESULTS = str(SOURCE_RESULTS)
    module.SUMMARY_PATH = str(SOURCE_RESULTS / "best_config_gap_filling.json")
    module.CURVES_PATH = str(SOURCE_RESULTS / "best_config_gap_filling_curves.npz")
    module.RECON_PATH = str(SOURCE_RESULTS / "main_figure5_reconstruction.npz")
    module.FIG_DIR = str(FIGURE_DIR)
    module.PDF_PATH = str(FIGURE_DIR / "figure05_gap_filling.pdf")
    module.PNG_PATH = str(FIGURE_DIR / "figure05_gap_filling.png")
    module.main()

    inputs = [
        SOURCE_RESULTS / "best_config_gap_filling.json",
        SOURCE_RESULTS / "best_config_gap_filling_curves.npz",
        SOURCE_RESULTS / "main_figure5_reconstruction.npz",
    ]
    metadata = {
        "provenance": str(SOURCE_SCRIPT.relative_to(RELEASE_ROOT)),
        "audit": "archived and audit-regenerated PNGs are RGB-pixel identical",
        "inputs": {
            str(path.relative_to(RELEASE_ROOT)): sha256(path) for path in inputs
        },
        "outputs": [module.PDF_PATH, module.PNG_PATH],
    }
    result_path = RESULT_DIR / "figure05_gap_filling.json"
    result_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
