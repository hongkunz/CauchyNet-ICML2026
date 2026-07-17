# CauchyNet: Compact and Data-Efficient Learning Using Holomorphic Activation Functions

Reference implementation and reproducibility package for the ICML 2026 paper
**"CauchyNet: Compact and Data-Efficient Learning Using Holomorphic Activation
Functions"** by Hong-Kun Zhang, Xin Li, Sikun Yang, and Zhihong Xia.

CauchyNet is a single-hidden-layer complex-valued network whose activation is
the multivariate Cauchy kernel

```
X(z) = prod_{i=1}^{N} z_i^{-1}
```

It is designed for data-scarce regimes and targets with sharp, rational-like
spikes, where standard real-valued networks become width-hungry.

## Repository layout

```text
.
├── README.md                  <- this file
├── paper.pdf                  <- camera-ready paper + supplement (27 pages)
├── LICENSE                    <- MIT
├── requirements.txt
├── experiments/               <- reference implementation + all released experiments
│   ├── shared.py              <- CauchyNet, FNN, SIREN, Transformer, N-BEATS, training utils
│   ├── run_all.py             <- entry point: python run_all.py [1 2 ... 11]
│   ├── verify_table1.py       <- cross-checks shipped JSONs vs. the paper's Table 1 (<1 s)
│   ├── exp1 ... exp11 ...     <- one script per released experiment
│   ├── best_config_gap_filling.py  <- Table 1 headline (gap-filling, 10 seeds)
│   ├── plot_main_figure5.py   <- regenerates the main empirical figure
│   └── results/               <- JSON results + LaTeX tables from our runs
└── reproducibility_audit/     <- reproducibility audit (July 2026)
    ├── README.md              <- full findings report, item-by-item verdicts + caveats
    ├── independent_repro/     <- from-scratch reimplementations of the headline results
    │   ├── repro_gap_filling.py   (Table 1: MAE 0.0200 vs paper's 0.0202)
    │   ├── recover_m4.py          (M4 7-model figure, legacy pipeline, self-contained)
    │   └── repro_fig1/24/25...    (main Fig. 1 and supplement ablations)
    ├── release_rerun/         <- fresh retrain of all 11 experiments + diff tool
    ├── notebook_reruns/       <- legacy-notebook reruns (2D missing disk, M4, heatmaps)
    ├── legacy_m4/             <- minimal legacy modules + data for recover_m4.py
    └── regenerated_figures/   <- pixel-level regenerations of the paper figures
```

## Installation

Requires Python 3.9+ and PyTorch 2.0+.

```bash
pip install -r requirements.txt
```

A GPU is optional; every experiment here runs on a laptop CPU (the largest
single run is a few minutes).

## Quick start

```bash
# 1. Validate the shipped result files against the paper's Table 1 (<1 s)
cd experiments
python verify_table1.py

# 2. Retrain everything (about 4 minutes on an Apple-Silicon CPU)
python run_all.py            # or: python run_all.py 1 4 9

# 3. Reproduce the headline gap-filling result from scratch (~7 minutes)
cd ../reproducibility_audit/independent_repro
python repro_gap_filling.py

# 4. Reproduce the M4 time-series comparison (self-contained, ~10 minutes)
python recover_m4.py
```

## Reproducing the paper's numbers

| Paper result                                     | Script                                     |
| ------------------------------------------------ | ------------------------------------------ |
| Table 1: gap-filling (MAE 0.0202, 4.3x vs SIREN) | `experiments/best_config_gap_filling.py`   |
| Table 1: param-match (20x vs FNN, near-singular) | `experiments/exp4_parameter_matched.py`    |
| Table 1: delta-sweep (up to ~100x vs FNN)        | `experiments/exp9_delta_sweep.py`          |
| Table 1: multi-layer Skip (1.7x vs 1-layer)      | `experiments/exp6_multilayer_cauchy.py`    |
| Table 1: UCI tabular (Hybrid wins 3/3)           | `experiments/exp7_uci_tabular.py`          |
| Table 1: RationalNN (wins 3/3, 1.2-2.0x)         | `experiments/exp8_rational_nn.py`          |
| Supp: fixed-pole ridge (14.4x-81.7x)             | `experiments/exp11_fixed_pole_ridge.py`    |
| Supp: M4 trend, 7-model comparison               | `reproducibility_audit/independent_repro/recover_m4.py` |
| Main figure (Fig. 5)                             | `experiments/plot_main_figure5.py`         |

Each script writes JSON (and, where applicable, LaTeX tables and figures)
into `experiments/results/`.

## Reproducibility audit

The `reproducibility_audit/` folder contains an audit performed on
2026-07-11: every table and quantitative figure in the paper and
supplement was either re-executed from this code or reimplemented from
scratch using only the paper's description. Headline outcomes:

- **Table 1 (gap-filling)**: independently reimplemented; MAE 0.0200 vs the
  paper's 0.0202, identical parameter counts, same ratios.
- **All 11 released experiments**: retrained; headline metrics agree with the
  shipped JSONs to the 4th-6th decimal (one experiment byte-identical).
- **M4 7-model comparison**: recovered with the legacy pipeline
  (`recover_m4.py`): CauchyNet 3.3e-4 vs SIREN 1.5e-3 vs baselines
  5e-3-1.6e-2, matching the paper's figure.
- **Sensitivity notes** (extreme values that vary with seed or protocol,
  one PyTorch-version-dependent parameter count, and two saved-result-only
  tables) are recorded in `reproducibility_audit/README.md`.

## Default hyperparameters

Most non-gap-filling scripts use the defaults in `experiments/shared.py`,
matching Table 2 of the supplement:

| Parameter                         | Value |
| --------------------------------- | ----- |
| Hidden size `h`                   | 64    |
| Batch size                        | 32    |
| Learning rate                     | 1e-2  |
| Weight decay                      | 1e-4  |
| Epochs                            | 200   |
| Imaginary penalty `lambda`        | 0.1   |
| Optimizer                         | Adam  |
| Epsilon (denominator regularizer) | 1e-8  |

The gap-filling headline uses the sweep-selected configuration: fixed poles
on the ellipse `(r_re, r_im) = (2.5, 0.4)`, `lambda_imag = 0`, `lr = 5e-2`,
`h = 64`, `n_train = 200`, 3000 epochs, 10 seeds.

## Minimal usage

```python
import torch
import sys; sys.path.insert(0, "experiments")
import shared

x = torch.linspace(-1, 1, 200).unsqueeze(1)
y = (4.0 / ((x - 0.5) ** 2 + 0.01) + torch.sin(3 * x)).squeeze()

model = shared.CauchyNet1D(hidden_size=64)
metrics = shared.train_and_eval(
    model, x[:160], y[:160], x[160:], y[160:],
    epochs=500, lr=1e-2, imag_penalty=0.1,
)
print(metrics)
```

## Citation

```bibtex
@inproceedings{zhang2026cauchynet,
  title     = {{CauchyNet}: Compact and Data-Efficient Learning Using Holomorphic Activation Functions},
  author    = {Zhang, Hong-Kun and Li, Xin and Yang, Sikun and Xia, Zhihong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).

## Contact

Questions or issues: open a GitHub issue, or email `hongkunz@gmail.com`.
