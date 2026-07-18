"""Reproduce Figure 4: real and imaginary parts of the 1D Cauchy kernel.

The archived graphic comes from the first plotting cells of
``code/1d_case_missing_data/activation_2.ipynb``; its checkpoint preserves
the original imaginary-part cell. It uses x in [-5, 5], ellipse radii (3, 2),
seven poles in the real panel, and nine in the imaginary panel, with a pi/20
angular offset.

The PDF caption says the semi-major axis is 6. Both the graphic and notebook
use semi-major axis 3 (full width 6). This script reproduces the graphic and
records the caption discrepancy in its JSON metadata.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"


def make_poles(count: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False) + math.pi / 20.0
    return 3.0 * np.cos(angles) + 2.0j * np.sin(angles)


def draw_panel(axis: plt.Axes, component: str, count: int) -> np.ndarray:
    x_values = np.linspace(-5.0, 5.0, 400)
    xi_values = make_poles(count)
    normalization = colors.Normalize(vmin=0, vmax=count - 1)
    colormap = cm.viridis

    for index, xi in enumerate(xi_values):
        kernel = 1.0 / (xi - x_values)
        values = kernel.real if component == "real" else kernel.imag
        axis.plot(x_values, values, color=colormap(normalization(index)), linewidth=2.5)

    axis.axhline(0.0, color="black", linewidth=0.5)
    axis.axvline(0.0, color="black", linewidth=0.5)
    axis.grid(True, linestyle="--", alpha=0.7)
    axis.set_xlim(-5.0, 5.0)
    axis.set_xlabel("x", fontsize=12)

    if component == "real":
        axis.set_title(
            r"Graph of real part of $K_{\xi}(x)=\frac{1}{\xi-x}$ with $\xi$ taken values on an ellipse",
            fontsize=14,
        )
        axis.set_ylabel(r"$Re(K_{\xi}(x))$", fontsize=12)
        inset_bounds = [0.80, 0.65, 0.15, 0.15]
    else:
        axis.set_title(
            r"Graph of imaginary part of $K(\xi,x)=\frac{1}{\xi-x}$ with $\xi$ taken on an ellipse",
            fontsize=14,
        )
        axis.set_ylabel(r"$K(\xi,x)$", fontsize=12)
        inset_bounds = [0.82, 0.35, 0.15, 0.15]

    inset = axis.inset_axes(inset_bounds)
    inset.add_patch(Ellipse((0, 0), width=6.0, height=4.0,
                             edgecolor="black", fill=False))
    inset.set_xlim(-3.5, 3.5)
    inset.set_ylim(-2.5, 2.5)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for index, xi in enumerate(xi_values):
        inset.plot(xi.real, xi.imag, "o", color=colormap(normalization(index)))
    inset.set_title(r"Values of $\xi$ on ellipse", fontsize=10)
    return xi_values


def main() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    real_poles = draw_panel(axes[0], "real", 7)
    imaginary_poles = draw_panel(axes[1], "imaginary", 9)
    figure.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figure04_elliptical_poles"
    figure.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "x_domain": [-5.0, 5.0],
        "ellipse_semi_axes_used_by_graphic": [3.0, 2.0],
        "ellipse_full_width_and_height": [6.0, 4.0],
        "angle_offset": "pi/20",
        "real_panel_poles": [[float(z.real), float(z.imag)] for z in real_poles],
        "imaginary_panel_poles": [[float(z.real), float(z.imag)] for z in imaginary_poles],
        "caption_discrepancy": (
            "PDF says semi-major axis 6; notebook and graphic use semi-major "
            "axis 3, i.e. full ellipse width 6"
        ),
        "provenance": (
            "code/1d_case_missing_data/activation_2.ipynb cells 1-2; "
            "checkpoint preserves the original imaginary cell"
        ),
    }
    result_path = RESULT_DIR / "figure04_elliptical_poles.json"
    result_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
