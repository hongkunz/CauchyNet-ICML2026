"""Reproduce Figure 2: the complex reciprocal Cauchy activation.

The figure evaluates X(z) = 1 / (z - x0) on [-2, 2]^2 with x0=0.01,
matching cell 24 of ``code/1d_case_missing_data/activation_1.ipynb``.
Values with |X(z)| > 1.5 are excluded; equivalently, this removes the circular
disc |z - x0| < 2/3 around the pole. The top row shows real part, imaginary
part, and log10 magnitude. The bottom row shows the corresponding contours.

The original notebook does not force equal aspect on the bottom axes, so the
circular masked set appears elliptical in the published panel. That visual
choice is retained here to reproduce the PDF rather than redesign it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"


def reciprocal_activation(z: np.ndarray, pole: float) -> np.ndarray:
    return 1.0 / (z - pole)


def build_figure(grid_size: int, output_stem: Path) -> dict[str, float | int]:
    pole = 0.01
    magnitude_threshold = 1.5
    disc_radius = 1.0 / magnitude_threshold

    coordinates = np.linspace(-2.0, 2.0, grid_size)
    real_grid, imaginary_grid = np.meshgrid(coordinates, coordinates)
    z = real_grid + 1j * imaginary_grid
    values = reciprocal_activation(z, pole)
    magnitude = np.abs(values)
    excluded = magnitude > magnitude_threshold

    real_part = np.ma.array(values.real, mask=excluded)
    imaginary_part = np.ma.array(values.imag, mask=excluded)
    magnitude_masked = np.ma.array(magnitude, mask=excluded)
    log_magnitude = np.ma.log10(magnitude_masked)

    figure = plt.figure(figsize=(20, 16), dpi=300)
    layout = gridspec.GridSpec(2, 3, height_ratios=[5, 1])

    surface_specs = (
        (real_part, "cividis", "lightgreen", "Re(f(z))", 15, 45),
        (imaginary_part, "magma", "orange", "Im(f(z))", 15, 45),
        (log_magnitude, "plasma", "lightblue", "log(|f(z)|)", 30, 45),
    )
    for column, (surface, cmap, edge, zlabel, elevation, azimuth) in enumerate(
        surface_specs
    ):
        axis = figure.add_subplot(layout[0, column], projection="3d")
        axis.plot_surface(
            real_grid,
            imaginary_grid,
            surface,
            cmap=cmap,
            edgecolor=edge,
            alpha=0.8,
        )
        axis.set_xlabel("Re(z)")
        axis.set_ylabel("Im(z)")
        axis.set_zlabel(zlabel)
        axis.view_init(elev=elevation, azim=azimuth)
        if column == 0:
            axis.grid(False)

    contour_specs = (
        (real_part, "viridis", 15, r"$\Re(1/(z-x))$"),
        (imaginary_part, "magma", 15, r"$\Im(1/(z-x))$"),
        (magnitude_masked, "plasma", 5, r"$|1/(z-x)|$"),
    )
    for column, (surface, cmap, levels, title) in enumerate(contour_specs):
        axis = figure.add_subplot(layout[1, column])
        contour = axis.contourf(
            real_grid, imaginary_grid, surface, levels=levels, cmap=cmap
        )
        axis.set_xlabel("Re(z)")
        axis.set_ylabel("Im(z)")
        axis.set_title(title)
        figure.colorbar(contour, ax=axis)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    return {
        "grid_size": grid_size,
        "domain_min": -2.0,
        "domain_max": 2.0,
        "pole": pole,
        "magnitude_threshold": magnitude_threshold,
        "excluded_disc_radius": disc_radius,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    suffix = "_smoke" if args.smoke_test else ""
    grid_size = 180 if args.smoke_test else 2800
    figure_stem = FIGURE_DIR / f"figure02_cauchy_activation{suffix}"
    result_path = RESULT_DIR / f"figure02_cauchy_activation{suffix}.json"

    metadata = build_figure(grid_size, figure_stem)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {figure_stem.with_suffix('.pdf')}")
    print(f"Saved {figure_stem.with_suffix('.png')}")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
