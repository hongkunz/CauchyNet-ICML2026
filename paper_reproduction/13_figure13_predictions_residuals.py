"""Reproduce Figure 13: M4 trend predictions and residuals for seven models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from _legacy_m4 import (
    CauchyNet,
    LSTM,
    MinimalInformer,
    MinimalTransformer,
    NBeats,
    RBFNetwork,
    SIREN,
    loadData,
    train_and_evaluate_model,
)
from _m4_data import load_m4_h1


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"


matplotlib.rcParams.update({"text.usetex": False, "pdf.fonttype": 42})

SEED = 1
HIDDEN_WIDTH = 128
EPOCHS = 200
LEARNING_RATE = 0.005
BATCH_SIZE = 32
IMAGINARY_PENALTY = 0.1

MODELS = {
    "CauchyNet": CauchyNet,
    "SIREN": SIREN,
    "NBeats": NBeats,
    "LSTM": LSTM,
    "Transformer": MinimalTransformer,
    "RBF": RBFNetwork,
    "Informer": MinimalInformer,
}
COLORS = {
    "CauchyNet": "tab:orange",
    "SIREN": "tab:blue",
    "NBeats": "tab:green",
    "LSTM": "tab:red",
    "Transformer": "tab:purple",
    "RBF": "tab:gray",
    "Informer": "tab:olive",
}


def make_data():
    _, truth = load_m4_h1()
    truth = truth.astype(np.float32)
    x = np.linspace(-1.0, 1.0, len(truth), dtype=np.float32)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(truth.reshape(-1, 1)).ravel().astype(np.float32)
    return x, truth, scaled, scaler


def fit_and_predict(
    name: str,
    constructor,
    x: np.ndarray,
    scaled: np.ndarray,
    scaler,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, validation, test, _ = loadData(
        torch.tensor(x), torch.tensor(scaled), batchSize=BATCH_SIZE
    )
    model, _, _, test_mse, test_mae, _, _, _, _ = train_and_evaluate_model(
        constructor,
        1,
        HIDDEN_WIDTH,
        1,
        train,
        validation,
        test,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        device=torch.device("cpu"),
        scaler=scaler,
        a=IMAGINARY_PENALTY if name == "CauchyNet" else 1.0,
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(x).unsqueeze(1))
        if isinstance(output, tuple):
            output = output[0]
    prediction_scaled = output.detach().cpu().numpy().reshape(-1, 1)
    prediction = scaler.inverse_transform(prediction_scaled).ravel()
    return prediction, float(test_mse), float(test_mae)


def main() -> None:
    x, truth, scaled, scaler = make_data()
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {"x": x, "truth": truth}
    for name, constructor in MODELS.items():
        prediction, test_mse, test_mae = fit_and_predict(
            name, constructor, x, scaled, scaler
        )
        predictions[name] = prediction
        arrays[f"{name}_prediction"] = prediction
        arrays[f"{name}_residual"] = prediction - truth
        metrics[name] = {
            "test_mse_unscaled": test_mse,
            "test_mae_unscaled": test_mae,
            "dense_grid_mae_unscaled": float(np.mean(np.abs(prediction - truth))),
        }
        print(
            f"{name:<12} test MAE={test_mae:.4f}; "
            f"dense-grid MAE={metrics[name]['dense_grid_mae_unscaled']:.4f}",
            flush=True,
        )

    figure, axes = plt.subplots(
        len(MODELS),
        2,
        figsize=(8.2, 10.2),
        sharex=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    for row, name in enumerate(MODELS):
        prediction = predictions[name]
        residual = prediction - truth
        color = COLORS[name]
        left, right = axes[row]
        left.plot(x, truth, "k--", linewidth=1.0, label="True")
        left.plot(x, prediction, color=color, linewidth=1.25, label=name)
        left.set_title(name, fontsize=8)
        left.grid(alpha=0.22)
        left.legend(fontsize=5, loc="best", ncol=2)
        right.axhline(0.0, color="0.35", linestyle="--", linewidth=0.7)
        right.plot(x, residual, color=color, linewidth=1.1)
        right.set_title(f"{name} Residual", fontsize=8)
        right.grid(alpha=0.22)
        right.set_ylim(-125.0, 50.0)
    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("x")
    figure.tight_layout(h_pad=0.45, w_pad=0.8)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figure13_predictions_residuals"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    npz_path = RESULT_DIR / "figure13_predictions_residuals.npz"
    np.savez_compressed(npz_path, **arrays)
    result = {
        "protocol": {
            "seed": SEED,
            "hidden_width": HIDDEN_WIDTH,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "imaginary_penalty": IMAGINARY_PENALTY,
            "split": [350, 175, 175],
            "target_scaling": "MinMaxScaler to [0,1] fitted before the split",
        },
        "metrics": metrics,
        "data_source": "paper_reproduction/_m4_data.py embedded M4 H1 payload",
        "provenance": [
            "ICML2026/source/Fig/PredictedVersusTrue.pdf",
            "code/1d_case_missing_data/7_experiment5_3.ipynb",
            "code/1d_case_missing_data/models.py",
            "code/1d_case_missing_data/utils2.py",
            "code_release/paper_reproduction/12_figure12_m4_benchmark.py",
        ],
    }
    json_path = RESULT_DIR / "figure13_predictions_residuals.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
