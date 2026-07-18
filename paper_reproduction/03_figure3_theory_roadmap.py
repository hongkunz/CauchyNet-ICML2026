"""Reproduce Figure 3: four-stage roadmap of the theoretical analysis.

This is an output-local version of
``code_release/experiments/make_roadmap_figure.py``. The release script was
rerun during the audit and its RGB pixels match the archived paper figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"

STAGES = [
    ("1", "Problem setup", r"$M\subset\mathbb{R}^{N}$ compact;  $\mathcal{F}_{M,D}=C^{0}(M,D)$"),
    ("2", "Cauchy kernel function", r"$K(\boldsymbol{\xi},\mathbf{x})=\prod_{i=1}^{N}(\xi^{i}-x_{i})^{-1}$"),
    ("3", "Cauchy approximation theorem", r"$\sup_{\mathbf{x}\in M}\,|f(\mathbf{x})-\sum_{k=1}^{m}\theta_{k}K(\boldsymbol{\xi}_{k},\mathbf{x})|<\varepsilon$"),
    ("4", "Universal approximation for CauchyNet", r"$\sup_{\mathbf{x}\in M}|f(\mathbf{x})-\Re(N_{\mathbf{B},\mathbf{C}}(\mathbf{x}))|<\varepsilon$"),
]


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(5.6, 3.6))
    axis.set(xlim=(0, 10), ylim=(0, 7.6))
    axis.set_aspect("equal")
    axis.axis("off")

    palette = ["#e8f0fb", "#dceaf6", "#c9dcee", "#b3cce4"]
    edge, number_color = "#3b6ba5", "#1f3b66"
    box_width, box_height, x_left = 8.8, 1.30, 0.8
    y_centers = [6.5, 4.8, 3.1, 1.4]

    for (label, title, body), y, fill in zip(STAGES, y_centers, palette):
        axis.add_patch(FancyBboxPatch(
            (x_left, y - box_height / 2), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.18", linewidth=1.0,
            edgecolor=edge, facecolor=fill, zorder=2,
        ))
        axis.add_patch(plt.Circle(
            (x_left + 0.55, y), 0.34, facecolor=number_color,
            edgecolor="none", zorder=4,
        ))
        axis.text(x_left + 0.55, y, label, ha="center", va="center",
                  color="white", fontsize=11, fontweight="bold", zorder=5)
        axis.text(x_left + 1.20, y + 0.32, title, ha="left", va="center",
                  fontsize=10.5, fontweight="bold", color=number_color, zorder=5)
        axis.text(x_left + 1.20, y - 0.32, body, ha="left", va="center",
                  fontsize=10.5, color="#222222", zorder=5)

    for y_top, y_bottom in zip(y_centers[:-1], y_centers[1:]):
        axis.add_patch(FancyArrowPatch(
            (x_left + box_width / 2, y_top - box_height / 2),
            (x_left + box_width / 2, y_bottom + box_height / 2 + 0.02),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.2,
            color="#5c7fad", zorder=1,
        ))

    figure.tight_layout(pad=0.15)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figure03_theory_roadmap"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    metadata = {
        "stages": [{"number": n, "title": t, "statement": s} for n, t, s in STAGES],
        "provenance": "code_release/experiments/make_roadmap_figure.py",
        "audit": "RGB-pixel match against archived flowchart.png",
    }
    result_path = RESULT_DIR / "figure03_theory_roadmap.json"
    result_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
