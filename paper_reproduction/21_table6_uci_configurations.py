"""Generate Table 6 from the saved tuned-UCI result record."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT.parent
SOURCE = RELEASE_ROOT / "experiments/results/exp7_uci_tuned_results.json"
OUT = ROOT / "outputs/results"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = []
    for dataset in ("Diabetes", "California", "Wine"):
        item = source[dataset]
        cfg = item["tuned_cfg"]
        rows.append(
            {
                "dataset": dataset,
                **cfg,
                "params": item["tuned_cauchy"]["num_params"],
            }
        )
    expected = [
        ("Diabetes", 32, 0.05, 3.0, 0.5, 704),
        ("California", 32, 0.05, 1.0, 1.5, 576),
        ("Wine", 32, 0.05, 1.0, 1.5, 768),
    ]
    observed = [(r["dataset"], r["h"], r["lr"], r["eps"], r["r_im"], r["params"]) for r in rows]
    if observed != expected:
        raise RuntimeError(f"Table 6 mismatch: {observed}")
    payload = {
        "shared": {"r_re": 2 * math.pi, "weight_decay": 1e-4, "lambda": 0.05, "epochs": 200},
        "rows": rows,
        "status": "printed values match the saved result record exactly",
        "reproducibility_gap": (
            "The original released exp7_sweep.py is only an lr/epoch sweep and "
            "does not produce this 54-configuration h/lr/eps/r_im grid. "
            "paper_reproduction/23_experiment12_uci_tuned_sweep.py is the clean "
            "replacement producer."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "table06_uci_configurations.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
