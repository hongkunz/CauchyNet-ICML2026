"""Reproduce Figure 11: synthetic target samples and the M4 trend target."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _m4_data import load_m4_h1


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"


def mixed_target(x: np.ndarray) -> np.ndarray:
    return (
        1.0 / ((x + 0.6) ** 2 + 0.005)
        - 40.0 * np.exp(-2.0 * (x + 0.4) ** 2)
        + 50.0 * np.sign(x) * np.abs(np.sin(4.0 * x) + 0.8) ** 1.5 * np.sin(10.0 * x)
    )


def main() -> None:
    x_left = np.linspace(-1.0, 1.0, 150)
    y_left = mixed_target(x_left)
    _, trend = load_m4_h1()
    x_right = np.linspace(-1.0, 1.0, len(trend))

    plt.rcParams.update({"font.family": "serif", "font.size": 11, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].scatter(x_left, y_left, s=12, color="blue", label=r"$(x_i,y_i)$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].scatter(x_right, trend, s=4, color="blue", label="f(x)")
    axes[1].set_xlabel("x")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figure11_targets"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)

    result = {
        "synthetic": {
            "points": 150,
            "domain": [-1.0, 1.0],
            "formula": (
                "1/((x+0.6)^2+0.005) - 40*exp(-2*(x+0.4)^2) "
                "+ 50*sign(x)*abs(sin(4*x)+0.8)^1.5*sin(10*x)"
            ),
        },
        "m4_trend": {
            "points": len(trend),
            "x_normalization": "linspace(-1,1,700)",
            "source": "paper_reproduction/_m4_data.py embedded M4 H1 payload",
        },
        "provenance": [
            "ICML2026/source/supp.tex Figure 11",
            "code/1d_case_missing_data/7_experiment5_3.ipynb challenging_function",
            "reproducibility_audit/notebook_reruns/run_e53b.py lines 137-150",
            "code/1d_case_missing_data/M4_trend_data.csv",
            "reproducibility_audit/notebook_reruns/run_redo_paper_experiments.py",
        ],
    }
    path = RESULT_DIR / "figure11_targets.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
