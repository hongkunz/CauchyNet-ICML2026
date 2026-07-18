"""Reproduce Table 1 from the saved 10-seed gap-filling results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT.parent
SOURCE = RELEASE_ROOT / "experiments/results/best_config_gap_filling.json"
OUT = ROOT / "outputs/results"
MODEL_KEYS = [
    ("CauchyNet", "CauchyNet"),
    ("SIREN", "SIREN"),
    ("FNN", "FNN_ReLU"),
    ("N-BEATS", "NBeats"),
]


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = []
    for label, key in MODEL_KEYS:
        item = source[key]
        per_seed = item["per_seed"]
        recomputed = {
            "mean_mae": float(np.mean([row["mae_mean"] for row in per_seed])),
            # The stored table median is pooled over all held-out errors.  It
            # cannot be reconstructed from the per-seed medians alone.
            "median": float(item["mae_median"]),
            "max": float(np.max([row["mae_max"] for row in per_seed])),
            "params": int(per_seed[0]["params"]),
        }
        stored = {
            "mean_mae": item["mae_mean"],
            "median": item["mae_median"],
            "max": item["mae_max"],
            "params": item["params"],
        }
        if not all(
            np.isclose(recomputed[name], stored[name], rtol=1e-7, atol=1e-12)
            for name in ("mean_mae", "max", "params")
        ):
            raise RuntimeError(f"Stored summary does not match per-seed data: {label}")
        rows.append({"model": label, **recomputed})

    configuration = {
        "hidden_size": 64,
        "pole_ellipse": [2.5, 0.4],
        "trainable_poles": False,
        "learning_rate": 0.05,
        "imaginary_penalty": 0.0,
        "epochs": 3000,
        "seeds": 10,
    }
    result = {
        "configuration": configuration,
        "rows": rows,
        "source": str(SOURCE.relative_to(RELEASE_ROOT)),
        "status": "exact: summaries recomputed from all 10 saved seeds",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "table01_gap_filling.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"Model & Mean MAE & Median & Max & Params \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['mean_mae']:.4f} & {row['median']:.4f} & "
            f"{row['max']:.4f} & {row['params']} \\\\"
        )
    lines.append(r"\end{tabular}")
    tex_path = OUT / "table01_gap_filling.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    for row in rows:
        print(
            f"{row['model']:<10} {row['mean_mae']:.4f}  {row['median']:.4f}  "
            f"{row['max']:.4f}  {row['params']}"
        )
    print(f"Saved {json_path}")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
