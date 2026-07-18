"""Reproduce Figure 12: the legacy seven-model M4 trend case study.

This script isolates the runnable protocol recovered from the original
``7_experiment5_3.ipynb`` workflow.  It uses the original model definitions,
50/25/25 split, target scaling, optimizer, scheduler, and imaginary loss.
Unlike the notebook, every run is seeded and all numerical arrays are saved.
"""

from __future__ import annotations

import argparse
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
    CauchyNet1,
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

MODEL_ORDER = [
    "CauchyNet",
    "SIREN",
    "NBeats",
    "LSTM",
    "Transformer",
    "RBF",
    "Informer",
]
COLORS = {
    "CauchyNet": "tab:orange",
    "SIREN": "tab:blue",
    "NBeats": "tab:green",
    "LSTM": "tab:red",
    "Transformer": "tab:purple",
    "RBF": "tab:gray",
    "Informer": "tab:olive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--imag-penalty", type=float, default=0.1)
    parser.add_argument(
        "--cauchy-variant",
        choices=("normal", "ellipse"),
        default="normal",
        help="normal is the audit recovery; ellipse is models.py:CauchyNet1",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Run a subset for protocol tuning; the publication figure uses all models.",
    )
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def constructors(cauchy_variant: str) -> dict[str, type[torch.nn.Module]]:
    return {
        "CauchyNet": CauchyNet1 if cauchy_variant == "ellipse" else CauchyNet,
        "SIREN": SIREN,
        "NBeats": NBeats,
        "LSTM": LSTM,
        "Transformer": MinimalTransformer,
        "RBF": RBFNetwork,
        "Informer": MinimalInformer,
    }


def make_loaders(seed: int, batch_size: int):
    _, y = load_m4_h1()
    y = y.astype(np.float32)
    x = np.linspace(-1.0, 1.0, len(y), dtype=np.float32)
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)
    torch.manual_seed(seed)
    return (*loadData(torch.tensor(x), torch.tensor(y_scaled), batchSize=batch_size)[:3], scaler)


def run_model(
    name: str,
    constructor: type[torch.nn.Module],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    train_runs: list[np.ndarray] = []
    val_runs: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    mse_runs: list[float] = []
    mae_runs: list[float] = []
    parameter_count = 0

    for seed in range(args.runs):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, val_loader, test_loader, scaler = make_loaders(seed, args.batch_size)
        (
            model,
            train_loss,
            val_loss,
            test_mse,
            test_mae,
            predictions,
            truths,
            _,
            _,
        ) = train_and_evaluate_model(
            constructor,
            1,
            128,
            1,
            train_loader,
            val_loader,
            test_loader,
            lr=args.lr,
            epochs=args.epochs,
            device=torch.device("cpu"),
            scaler=scaler,
            a=args.imag_penalty if name == "CauchyNet" else 1.0,
        )
        train_runs.append(np.asarray(train_loss, dtype=float))
        val_runs.append(np.asarray(val_loss, dtype=float))
        errors.append(np.abs(np.asarray(predictions) - np.asarray(truths)).ravel())
        mse_runs.append(float(test_mse))
        mae_runs.append(float(test_mae))
        parameter_count = sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    train = np.stack(train_runs)
    val = np.stack(val_runs)
    pooled_error = np.concatenate(errors)
    summary: dict[str, float | int] = {
        "test_mse_mean": float(np.mean(mse_runs)),
        "test_mse_std": float(np.std(mse_runs)),
        "test_mae_mean": float(np.mean(mae_runs)),
        "test_mae_std": float(np.std(mae_runs)),
        "pooled_abs_error_median": float(np.median(pooled_error)),
        "pooled_abs_error_q1": float(np.quantile(pooled_error, 0.25)),
        "pooled_abs_error_q3": float(np.quantile(pooled_error, 0.75)),
        "real_parameter_count": parameter_count,
    }
    print(
        f"{name:<12} MAE={summary['test_mae_mean']:.4f} +/- "
        f"{summary['test_mae_std']:.4f}; median abs={summary['pooled_abs_error_median']:.4f}",
        flush=True,
    )
    return train, val, pooled_error, summary


def plot_figure(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    errors: dict[str, np.ndarray],
    output_stem: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    epoch = np.arange(1, next(iter(curves.values()))[0].shape[1] + 1)
    for name in MODEL_ORDER:
        if name not in curves:
            continue
        train, val = curves[name]
        train_median = np.median(train, axis=0)
        train_std = np.std(train, axis=0)
        val_median = np.median(val, axis=0)
        val_std = np.std(val, axis=0)
        color = COLORS[name]
        axes[0].plot(epoch, train_median, color=color, linewidth=1.4, label=name)
        axes[0].fill_between(
            epoch,
            np.maximum(train_median - train_std, 1e-4),
            train_median + train_std,
            color=color,
            alpha=0.18,
        )
        axes[0].plot(epoch, val_median, color=color, linestyle="--", alpha=0.55)
        axes[0].fill_between(
            epoch,
            np.maximum(val_median - val_std, 1e-4),
            val_median + val_std,
            color=color,
            alpha=0.10,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-4, 10.0)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (log scale)")
    axes[0].set_title(r"Training & Validation (Median $\pm$ Std)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    names = [name for name in MODEL_ORDER if name in errors]
    boxplot = axes[1].boxplot(
        [errors[name] for name in names],
        tick_labels=names,
        patch_artist=True,
        showfliers=True,
        flierprops={"markersize": 2.5, "markerfacecolor": "none"},
    )
    for patch, name in zip(boxplot["boxes"], names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.42)
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title("Distribution of Absolute Errors (Test Set)")
    axes[1].grid(alpha=0.25)
    axes[1].tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    model_constructors = constructors(args.cauchy_variant)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    errors: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float | int]] = {}
    arrays: dict[str, np.ndarray] = {}
    for name in args.models:
        train, val, pooled_error, summary = run_model(name, model_constructors[name], args)
        curves[name] = (train, val)
        errors[name] = pooled_error
        summaries[name] = summary
        arrays[f"{name}_train"] = train
        arrays[f"{name}_validation"] = val
        arrays[f"{name}_absolute_error"] = pooled_error

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    stem = FIGURE_DIR / f"figure12_m4_benchmark{suffix}"
    plot_figure(curves, errors, stem)
    npz_path = RESULT_DIR / f"figure12_m4_benchmark{suffix}.npz"
    np.savez_compressed(npz_path, **arrays)
    result = {
        "protocol": {
            "runs": args.runs,
            "seeds": list(range(args.runs)),
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "imaginary_penalty": args.imag_penalty,
            "hidden_width": 128,
            "split": [350, 175, 175],
            "target_scaling": "MinMaxScaler to [0,1] fitted before the split",
            "cauchy_variant": args.cauchy_variant,
        },
        "metrics": summaries,
        "data_source": "paper_reproduction/_m4_data.py embedded M4 H1 payload",
        "provenance": [
            "code/1d_case_missing_data/7_experiment5_3.ipynb",
            "code/1d_case_missing_data/models.py",
            "code/1d_case_missing_data/utils2.py",
            "reproducibility_audit/independent_repro/recover_m4.py",
        ],
    }
    json_path = RESULT_DIR / f"figure12_m4_benchmark{suffix}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
