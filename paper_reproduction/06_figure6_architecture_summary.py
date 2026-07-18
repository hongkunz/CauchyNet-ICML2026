"""Render Figure 6: neural-architecture strengths and weaknesses.

Figure 6 is a framed text summary typeset directly in ``supp.tex``; it has no
experimental data. This deterministic renderer preserves the eight entries
and their exact substantive text in a standalone release figure.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"

ENTRIES = [
    ("LSTM", "Effectively learns long-term temporal dependencies via gating.",
     "Large parameter footprint and sensitive hyperparameters, especially in sparse-data contexts."),
    ("Transformer", "Captures long-range correlations efficiently with abundant data.",
     "Quadratic complexity in sequence length; memory-intensive."),
    ("Informer", "Employs sparse self-attention to enhance scalability for long sequences.",
     "Multi-layer stacks involve thousands of parameters and require large datasets."),
    ("N-BEATS", "Provides accurate forecasting with interpretable decompositions into trend and seasonal components.",
     "High parameter count and data-hungry; may overfit in scenarios with limited data."),
    ("MLP", "Simple and straightforward to implement and train.",
     "Lacks specialized inductive biases for oscillatory or near-singular data."),
    ("RBF Network", "Excels at local approximations and can model a broad class of functions.",
     "Placement of kernel centers is non-trivial; struggles with sharp rational spikes or steep gradients."),
    ("SIREN", "Sinusoidal activations capture smooth and high-frequency signals effectively.",
     "Requires large capacity to model abrupt spikes or extremely limited data; not explicitly designed for missing-data imputation."),
    ("TCN", "Uses dilated convolutions for efficient sequence modeling.",
     "Deeper architectures increase parameter counts and can be computationally slow."),
]


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "pdf.fonttype": 42})
    figure, axis = plt.subplots(figsize=(8.0, 10.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(Rectangle((0.025, 0.025), 0.95, 0.95, fill=False,
                             edgecolor="#555555", linewidth=1.0))

    y = 0.95
    line_height = 0.018
    for model, strength, weakness in ENTRIES:
        axis.text(0.055, y, f"• {model}", fontweight="bold", va="top")
        y -= line_height * 1.15
        for label, statement in (("Strengths", strength), ("Weaknesses", weakness)):
            # Keep a fixed label column and wrap against the actual available
            # width.  The older renderer used 91 characters and started the
            # prose too close to the bold label, which caused visible overlap.
            wrapped = textwrap.wrap(statement, width=55)
            axis.text(0.085, y, f"{label}:", fontweight="bold", va="top")
            axis.text(0.255, y, wrapped[0], va="top")
            for continuation in wrapped[1:]:
                y -= line_height
                axis.text(0.255, y, continuation, va="top")
            y -= line_height
        y -= line_height * 0.30

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figure06_architecture_summary"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)

    result_path = RESULT_DIR / "figure06_architecture_summary.json"
    result_path.write_text(json.dumps({
        "entries": [
            {"model": model, "strengths": strength, "weaknesses": weakness}
            for model, strength, weakness in ENTRIES
        ],
        "provenance": "ICML2026/source/supp.tex, Figure 6 framed itemize block",
        "status": "content-equivalent deterministic rendering; original is typeset directly by LaTeX",
    }, indent=2) + "\n")
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
