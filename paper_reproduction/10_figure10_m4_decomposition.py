"""Reproduce Figure 10: multiplicative decomposition of M4 series H1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from _m4_data import load_m4_h1


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"


def main() -> None:
    series, archived_y = load_m4_h1()
    decomposition = seasonal_decompose(
        series, model="multiplicative", period=24, extrapolate_trend=1
    )
    trend = np.asarray(decomposition.trend, dtype=float)
    trend_difference = trend - archived_y

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 16,
        "axes.labelsize": 12, "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(4, 1, figsize=(10, 11.5), sharex=True)
    components = [
        ("Original Data", np.asarray(decomposition.observed)),
        ("Trend", trend),
        ("Seasonal", np.asarray(decomposition.seasonal)),
        ("Residual", np.asarray(decomposition.resid)),
    ]
    for axis, (title, values) in zip(axes, components):
        axis.plot(np.arange(len(values)), values, color="#0072B2", linewidth=1.5)
        axis.set_title(title)
        axis.grid(True, color="lightblue", linestyle="--", linewidth=0.35)
    axes[-1].set_xlabel("Time")
    figure.suptitle("Time Series Decomposition", fontsize=19, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.98))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figure10_m4_decomposition"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)

    result = {
        "protocol": {
            "series": "M4 Hourly H1",
            "observations": 700,
            "model": "multiplicative",
            "period": 24,
            "extrapolate_trend": 1,
        },
        "data_source": "paper_reproduction/_m4_data.py embedded M4 H1 payload",
        "trend_comparison": {
            "max_abs_difference": float(np.max(np.abs(trend_difference))),
            "mse": float(np.mean(trend_difference ** 2)),
            "exact_within_1e-10": bool(np.max(np.abs(trend_difference)) < 1e-10),
        },
        "provenance": [
            "ICML2026/source/supp.tex Figure 10",
            "code/1d_case_missing_data/mstl_decomposition_M4.ipynb multiplicative cell",
            "reproducibility_audit/notebook_reruns/run_m4_decomp.py and run_m4_decomp.log",
        ],
    }
    path = RESULT_DIR / "figure10_m4_decomposition.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["trend_comparison"], indent=2))
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
