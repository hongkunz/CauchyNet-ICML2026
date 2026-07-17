"""
Faithful recovery of supp Fig. 19 (M4 trend, 7-model comparison).

Complementing repro_fig19_m4.py (a generic from-scratch protocol), this
script uses the authors' own legacy ingredients, exactly as wired in the
archived notebook 7_experiment5_3.ipynb, and reproduces the shipped ranking:

  - models.py            CauchyNet (tuple output, /h normalisation, std=1 init),
                         SIREN, RBFNetwork, FNN-free comparison set:
                         NBeats, LSTM, MinimalTransformer, MinimalInformer
  - utils2.py            loadData (random 50/25/25 split, batch 256 -- the
                         authors' original configuration) and
                         train_and_evaluate_model (Adam lr=0.005, wd=1e-10,
                         ReduceLROnPlateau(0.9, patience=5), 200 epochs,
                         imaginary penalty a on tuple outputs)
  - M4_trend_data.csv    700-observation M4 trend subset
  - x mapped to [-1, 1], Y MinMax-scaled to [0, 1]

Run from this directory:
    python recover_m4.py            # 5 runs/model (default)
    python recover_m4.py --runs 10

Outputs: recover_m4_losses.png, recover_m4_boxplot.png, stats to stdout.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Self-contained: legacy model/training modules and M4 data ship in
# reproducibility_audit/legacy_m4/ (fallback: the full manuscript workspace).
_CANDIDATES = [
    os.path.abspath(os.path.join(HERE, "..", "legacy_m4")),
    os.path.abspath(os.path.join(HERE, "..", "..", "code", "1d_case_missing_data")),
]
LEGACY = next(p for p in _CANDIDATES if os.path.isdir(p))
sys.path.insert(0, LEGACY)

from sklearn.preprocessing import MinMaxScaler          # noqa: E402
from models import (CauchyNet, SIREN, RBFNetwork, LSTM,  # noqa: E402
                    MinimalTransformer, MinimalInformer, NBeats)
from utils2 import loadData, train_and_evaluate_model    # noqa: E402

# the legacy modules enable text.usetex at import time; keep plotting plain
matplotlib.rcParams.update({"text.usetex": False})
plt.rcParams.update({"text.usetex": False})

H = 128          # hidden size (supp per-model configurations)
EPOCHS = 200
LR = 0.005    # authors' original configuration
IMAG_A = 1       # imaginary penalty for CauchyNet (notebook default a=1)

MODELS = {
    "CauchyNet":   CauchyNet,
    "SIREN":       SIREN,
    "NBeats":      NBeats,
    "LSTM":        LSTM,
    "Transformer": MinimalTransformer,
    "RBF":         RBFNetwork,
    "Informer":    MinimalInformer,
}
COLORS = {"CauchyNet": "tab:orange", "SIREN": "tab:blue", "NBeats": "tab:green",
          "LSTM": "tab:red", "Transformer": "tab:purple", "RBF": "tab:gray",
          "Informer": "tab:olive"}


BATCH = 256  # authors' original configuration; override with --batch

def build_loaders(seed):
    """M4 trend -> (x in [-1,1], Y MinMax-scaled), 50/25/25 random split."""
    df = pd.read_csv(os.path.join(LEGACY, "M4_trend_data.csv"))
    y = df["Y"].to_numpy(dtype=np.float32)[:700]
    x = np.linspace(-1.0, 1.0, len(y)).astype(np.float32)

    scaler = MinMaxScaler()
    y_norm = scaler.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    torch.manual_seed(seed)          # loadData uses torch random_split
    X = torch.tensor(x)
    Y = torch.tensor(y_norm)         # 1-D: the legacy loop unsqueezes itself
    train_loader, val_loader, test_loader, _ = loadData(X, Y, batchSize=BATCH)
    return train_loader, val_loader, test_loader, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.005)
    args = ap.parse_args()
    global BATCH
    BATCH = args.batch

    curves, errors = {}, {}
    for name, ctor in MODELS.items():
        trs, vas, errs = [], [], []
        n_params = None
        for s in range(args.runs):
            torch.manual_seed(s); np.random.seed(s)
            tl, vl, te, scaler = build_loaders(seed=s)
            a = IMAG_A if name == "CauchyNet" else 1
            (model, tr, va, mse, mae, preds, truths,
             ttime, n_params) = train_and_evaluate_model(
                ctor, 1, H, 1, tl, vl, te,
                lr=args.lr, epochs=EPOCHS, device=torch.device("cpu"),
                scaler=scaler, a=a)
            trs.append(np.asarray(tr)); vas.append(np.asarray(va))
            errs.append(np.abs(np.asarray(preds) - np.asarray(truths)).flatten())
        # utils2 counts a complex scalar as one parameter; the paper's
        # convention counts it as two reals, so recount here.
        n_params = sum(p.numel() * (2 if p.is_complex() else 1)
                       for p in model.parameters() if p.requires_grad)
        curves[name] = (np.stack(trs), np.stack(vas))
        errors[name] = np.concatenate(errs)
        print(f"{name:<12} final val MSE median={np.median([v[-1] for v in vas]):.2e}  "
              f"test MAE(unscaled)={errors[name].mean():.4f}  params={n_params}",
              flush=True)

    ep = np.arange(1, EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    for name in MODELS:
        tr, va = curves[name]
        med = np.median(tr, 0)
        ax.plot(ep, med, color=COLORS[name], label=name)
        ax.fill_between(ep, np.maximum(med - tr.std(0), 1e-12), med + tr.std(0),
                        color=COLORS[name], alpha=0.15)
        ax.plot(ep, np.median(va, 0), color=COLORS[name], ls="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title(f"M4 trend, legacy pipeline — Training and Validation "
                 f"(Median +/- Std, {args.runs} runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "recover_m4_losses.png"),
                dpi=130, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(MODELS)
    ax.boxplot([errors[n] for n in names], tick_labels=names, showfliers=False)
    ax.set_ylabel("Absolute Error (unscaled)")
    ax.set_title("M4 trend, legacy pipeline — test absolute errors")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "recover_m4_boxplot.png"),
                dpi=130, bbox_inches="tight")
    print("saved recover_m4_losses.png, recover_m4_boxplot.png")


if __name__ == "__main__":
    main()
