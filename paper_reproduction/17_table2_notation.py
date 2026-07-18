"""Generate Table 2, the paper's notation dictionary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/results"
ROWS = [
    ("M", r"M=\prod_{i=1}^N M_i\subset\mathbb{R}^N", "Compact real domain."),
    ("U_i", r"M_i\subset U_i\subset\mathbb{C}", "Planar complex domain enclosing M_i."),
    ("U", r"U=\prod_{i=1}^N U_i\subset\mathbb{C}^N", "Product domain."),
    (r"\Gamma", r"\Gamma=\partial U_1\times\cdots\times\partial U_N", "Product contour for multivariate Cauchy formula."),
    (r"K(\xi,x)", r"\prod_{i=1}^N(\xi^i-x_i)^{-1}", "Multivariate Cauchy kernel atom."),
    ("f", r"f\in C(M;\mathbb{R})", "Target function to approximate."),
    (r"\xi_k", r"\xi_k\in\Gamma", "Pole locations for kernel atoms."),
    (r"\theta_k", r"\theta_k\in\mathbb{C}", "Complex weights in the kernel expansion."),
    (r"N_{B,C}", r"\mathrm{CauchyNet}", r"Network with parameters $B\in\mathbb{C}^{h\times N}$, $C\in\mathbb{C}^h$."),
    ("h", r"\mathrm{integer}", "Hidden-layer width."),
    (r"\lambda", r"\lambda\geq0", "Imaginary-part penalty weight."),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"rows": [dict(zip(("symbol", "definition", "meaning"), row)) for row in ROWS]}
    json_path = OUT / "table02_notation.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    tex = [r"\begin{tabular}{lll}", r"Symbol & Definition & Meaning \\", r"\hline"]
    tex.extend(f"${s}$ & ${d}$ & {m} \\\\" for s, d, m in ROWS)
    tex.append(r"\end{tabular}")
    tex_path = OUT / "table02_notation.tex"
    tex_path.write_text("\n".join(tex) + "\n")
    print(f"Saved {json_path}")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
