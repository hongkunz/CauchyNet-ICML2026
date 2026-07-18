"""Reproduce Table 5 from the exact saved fixed-pole ridge experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT.parent
SOURCE = RELEASE_ROOT / "experiments/results/exp11_fixed_pole_ridge_results.json"
OUT = ROOT / "outputs/results"
EXPECTED_RATIOS = {20: 14.8, 25: 14.4, 30: 17.8, 40: 55.6, 50: 81.7}


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = []
    for n_train, paper_ratio in EXPECTED_RATIOS.items():
        item = source["results"][str(n_train)]
        cn = item["FixedPoleCauchyRidge"]
        fnn = item["FNN"]
        for model in (cn, fnn):
            mse = np.array([run["mse"] for run in model["per_seed"]])
            if not np.isclose(mse.mean(), model["mse_mean"], atol=1e-14):
                raise RuntimeError(f"Mean mismatch at n={n_train}")
            if not np.isclose(mse.std(), model["mse_std"], atol=1e-14):
                raise RuntimeError(f"Std mismatch at n={n_train}")
        ratio = fnn["mse_mean"] / cn["mse_mean"]
        if round(ratio, 1) != paper_ratio:
            raise RuntimeError(f"Ratio mismatch at n={n_train}: {ratio}")
        rows.append(
            {
                "n_train": n_train,
                "cauchy_mse_mean": cn["mse_mean"],
                "cauchy_mse_std": cn["mse_std"],
                "fnn_mse_mean": fnn["mse_mean"],
                "fnn_mse_std": fnn["mse_std"],
                "ratio": ratio,
            }
        )
    payload = {
        "protocol": source["protocol"],
        "rows": rows,
        "status": "exact: per-seed summaries and all printed ratios verified",
        "producer": "code_release/experiments/exp11_fixed_pole_ridge.py",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "table05_fixed_pole_ridge.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    tex = [
        r"\begin{tabular}{rrrr}",
        r"$n_{train}$ & Fixed-$\xi$ CN ridge & FNN & ratio \\",
        r"\hline",
    ]
    for row in rows:
        tex.append(
            f"{row['n_train']} & {row['cauchy_mse_mean']:.6f} $\\pm$ {row['cauchy_mse_std']:.6f} & "
            f"{row['fnn_mse_mean']:.4f} $\\pm$ {row['fnn_mse_std']:.4f} & {row['ratio']:.1f}$\\times$ \\\\"
        )
    tex.append(r"\end{tabular}")
    tex_path = OUT / "table05_fixed_pole_ridge.tex"
    tex_path.write_text("\n".join(tex) + "\n")
    for row in rows:
        print(f"n={row['n_train']:2d}: ratio={row['ratio']:.1f}x")
    print(f"Saved {json_path}")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
