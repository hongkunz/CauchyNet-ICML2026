"""Generate Table 3, the stated legacy 1D trainable-pole defaults."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/results"
ROWS = [
    ("Data Split Ratio", "50% training, 25% validation, 25% testing"),
    ("Architecture", "One hidden layer with 128 neurons"),
    ("Batch Size", "32"),
    ("Optimizer", "Adam, learning rate 0.01, decayed by factor 0.5 every 100 epochs"),
    ("Weight Decay", "1e-4"),
    ("Preprocessing", "Target values normalized via MinMaxScaler"),
    ("Random Seed", "10"),
    ("Imaginary-Part Penalty", "lambda=0.1; final fixed-pole gap filling uses 0"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": [{"parameter": key, "value": value} for key, value in ROWS],
        "scope": "legacy one-dimensional trainable-pole default only",
        "warning": "The removed legacy sensitivity audit uses its own non-default schedules and RNG state.",
    }
    json_path = OUT / "table03_default_setup.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    tex = [r"\begin{tabular}{ll}", r"Parameter & Value \\", r"\hline"]
    for key, value in ROWS:
        escaped_value = value.replace("%", r"\%")
        tex.append(f"{key} & {escaped_value} \\\\")
    tex.append(r"\end{tabular}")
    tex_path = OUT / "table03_default_setup.tex"
    tex_path.write_text("\n".join(tex) + "\n")
    print(f"Saved {json_path}")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
