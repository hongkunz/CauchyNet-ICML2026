"""Reproduce Figure 9: limited-data 2D polynomial-rational approximation.

The default is the verified protocol reported in the revised supplement:
seed 10, 300 uniform samples on [-1.5,1.5]^2, a 50/25/25 split, width 128,
ellipse pole initialization (2.5, 0.5), input scaling by 1.5, and 20,000 Adam
epochs. Use ``--legacy-500`` to audit the earlier 500-epoch draft protocol
(normal initialization, no imaginary penalty).

The split is formed first and the target scaler is fitted to the 150 training
targets only. The same transformation is then applied to validation, test,
and plotting-grid predictions.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    return x1.square() - x1 * x2 + 3.0 * x2 + x2.square() + 1.0 / (5.0 + x1.square())


class CauchyNet2D(nn.Module):
    def __init__(
        self, width: int, pole_init: str = "normal", r_re: float = 2 * np.pi,
        r_im: float = 1.0, input_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.input_scale = input_scale
        if pole_init == "ellipse":
            angles = 2.0 * np.pi * torch.rand(2, width)
            pole_real = r_re * torch.cos(angles)
            pole_imag = r_im * torch.sin(angles)
        else:
            pole_real = torch.randn(2, width)
            pole_imag = torch.randn(2, width)
        self.poles = nn.Parameter(torch.complex(pole_real, pole_imag))
        self.coefficients = nn.Parameter(
            torch.complex(0.01 * torch.randn(width, 1), 0.01 * torch.randn(width, 1))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x / self.input_scale
        z = torch.complex(x, torch.zeros_like(x))
        atoms = 1.0 / (
            torch.prod(self.poles.T.unsqueeze(0) - z.unsqueeze(1), dim=2) + 1e-8
        )
        output = atoms @ self.coefficients
        return output.real.squeeze(1), output.imag.squeeze(1)


def predict(model: CauchyNet2D, x: torch.Tensor, device: torch.device) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for chunk in x.split(8192):
            scaled, _ = model(chunk.to(device))
            output.append((scaled * model.y_range + model.y_min).cpu().numpy())  # type: ignore[attr-defined]
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--imaginary-penalty", type=float, default=0.1)
    parser.add_argument("--pole-init", choices=["normal", "ellipse"], default="ellipse")
    parser.add_argument("--pole-real-radius", type=float, default=2.5)
    parser.add_argument("--pole-imag-radius", type=float, default=0.5)
    parser.add_argument("--input-scale", type=float, default=1.5)
    parser.add_argument(
        "--lbfgs-steps", type=int, default=0,
        help="Optional full-batch LBFGS refinement after the best Adam checkpoint."
    )
    parser.add_argument("--step-size", type=int, default=4000)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--legacy-500", action="store_true",
        help="Audit the earlier 500-epoch draft protocol."
    )
    args = parser.parse_args()
    if args.legacy_500:
        args.epochs = 500
        args.imaginary_penalty = 0.0
        args.pole_init = "normal"
        args.input_scale = 1.0
        args.step_size = 100
        args.gamma = 0.5
        if not args.tag:
            args.tag = "legacy_500"
    epochs = 2 if args.smoke else args.epochs
    suffix = "_smoke" if args.smoke else (f"_{args.tag}" if args.tag else "")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x = torch.from_numpy(rng.uniform(-1.5, 1.5, size=(300, 2)).astype(np.float32))
    y = target(x)
    raw_dataset = TensorDataset(x, y)
    raw_train, raw_validation, raw_test = random_split(
        raw_dataset, [150, 75, 75], generator=torch.Generator().manual_seed(args.seed)
    )
    train_indices = torch.tensor(raw_train.indices)
    y_min, y_max = float(y[train_indices].min()), float(y[train_indices].max())
    y_range = y_max - y_min
    y_scaled = (y - y_min) / y_range
    dataset = TensorDataset(x, y_scaled)
    train_set = Subset(dataset, raw_train.indices)
    val_set = Subset(dataset, raw_validation.indices)
    test_set = Subset(dataset, raw_test.indices)
    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    set_seed(args.seed)
    model = CauchyNet2D(
        args.width, args.pole_init, args.pole_real_radius,
        args.pole_imag_radius, args.input_scale
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    criterion = nn.MSELoss()
    train_history: list[float] = []
    validation_history: list[float] = []
    best_value = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            real, imaginary = model(batch_x)
            loss = criterion(real, batch_y) + args.imaginary_penalty * criterion(
                imaginary, torch.zeros_like(imaginary)
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
        train_history.append(total / len(train_set))

        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                real, imaginary = model(batch_x.to(device))
                loss = criterion(real, batch_y.to(device)) + args.imaginary_penalty * criterion(
                    imaginary, torch.zeros_like(imaginary)
                )
                total += float(loss) * len(batch_x)
        validation_history.append(total / len(val_set))
        if validation_history[-1] < best_value:
            best_value = validation_history[-1]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)

    lbfgs_validation = None
    if args.lbfgs_steps > 0:
        train_indices = torch.tensor(train_set.indices)
        full_train_x = x[train_indices].to(device)
        full_train_y = y_scaled[train_indices].to(device)
        refinement = torch.optim.LBFGS(
            model.parameters(), lr=0.5, max_iter=args.lbfgs_steps,
            history_size=100, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            refinement.zero_grad(set_to_none=True)
            real, imaginary = model(full_train_x)
            loss = criterion(real, full_train_y) + args.imaginary_penalty * criterion(
                imaginary, torch.zeros_like(imaginary)
            )
            loss.backward()
            return loss

        refinement.step(closure)
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                real, imaginary = model(batch_x.to(device))
                loss = criterion(real, batch_y.to(device)) + args.imaginary_penalty * criterion(
                    imaginary, torch.zeros_like(imaginary)
                )
                total += float(loss) * len(batch_x)
        lbfgs_validation = total / len(val_set)
        if lbfgs_validation > best_value:
            model.load_state_dict(best_state)
    model.y_min = y_min  # type: ignore[attr-defined]
    model.y_range = y_range  # type: ignore[attr-defined]

    test_indices = torch.tensor(test_set.indices)
    test_x = x[test_indices]
    test_truth = y[test_indices].numpy()
    test_prediction = predict(model, test_x, device)

    axis = torch.linspace(-1.5, 1.5, 180)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    grid = torch.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_truth = target(grid).numpy().reshape(grid_x.shape)
    grid_prediction = predict(model, grid, device).reshape(grid_x.shape)
    grid_error = grid_prediction - grid_truth

    metrics = {
        "test_mse": float(np.mean((test_prediction - test_truth) ** 2)),
        "test_mae": float(np.mean(np.abs(test_prediction - test_truth))),
        "grid_mse": float(np.mean(grid_error ** 2)),
        "grid_mae": float(np.mean(np.abs(grid_error))),
        "grid_error_min": float(grid_error.min()),
        "grid_error_max": float(grid_error.max()),
        "grid_max_abs_error": float(np.max(np.abs(grid_error))),
    }

    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(10.5, 8.2))
    panel = figure.add_subplot(111, projection="3d")
    panel.plot_wireframe(
        grid_x.numpy(), grid_y.numpy(), grid_truth,
        rstride=9, cstride=9, color="#2389ff", linewidth=1.2, alpha=0.75
    )
    panel.plot_surface(
        grid_x.numpy(), grid_y.numpy(), grid_prediction,
        cmap="viridis", alpha=0.72, linewidth=0
    )
    floor = float(min(grid_truth.min(), grid_prediction.min()) - 0.8)
    norm = Normalize(vmin=float(grid_error.min()), vmax=float(grid_error.max()))
    panel.contourf(
        grid_x.numpy(), grid_y.numpy(), grid_error, zdir="z", offset=floor,
        levels=45, cmap="viridis", norm=norm
    )
    scalar = cm.ScalarMappable(norm=norm, cmap="viridis")
    scalar.set_array(grid_error)
    colorbar = figure.colorbar(scalar, ax=panel, shrink=0.55, pad=0.08)
    colorbar.set_label("Error Value", fontweight="bold")
    panel.set_zlim(floor, float(max(grid_truth.max(), grid_prediction.max()) + 0.5))
    panel.set_title("3D View of Model Output vs Target", fontsize=20, fontweight="bold")
    panel.set_xlabel("x1", fontsize=13, fontweight="bold", labelpad=8)
    panel.set_ylabel("x2", fontsize=13, fontweight="bold", labelpad=8)
    panel.set_zlabel("Value", fontsize=13, fontweight="bold", labelpad=8)
    panel.view_init(elev=27, azim=-60)
    panel.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="Target Plane", markerfacecolor="#2389ff", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Prediction Surface", markerfacecolor="green", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Error", markerfacecolor="grey", markersize=9),
        ],
        loc="upper left",
    )
    figure.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / f"figure09_2d_surface{suffix}"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)

    result = {
        "protocol": {
            "seed": args.seed,
            "domain": [-1.5, 1.5],
            "sample_count": 300,
            "split_counts": {"train": 150, "validation": 75, "test": 75},
            "width": args.width,
            "epochs": epochs,
            "batch_size": 32,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "imaginary_penalty": args.imaginary_penalty,
            "pole_init": args.pole_init,
            "pole_radii": [args.pole_real_radius, args.pole_imag_radius],
            "input_scale_divisor": args.input_scale,
            "step_lr": {"step_size": args.step_size, "gamma": args.gamma},
            "target_scaler_fit": "150 training targets only",
            "checkpoint": "lowest validation loss",
            "best_epoch": best_epoch,
            "best_validation_loss_scaled": best_value,
            "lbfgs_steps": args.lbfgs_steps,
            "lbfgs_validation_loss_scaled": lbfgs_validation,
            "lbfgs_accepted": lbfgs_validation is not None and lbfgs_validation <= best_value,
        },
        "metrics": metrics,
        "training_seconds": elapsed,
        "final_train_loss_scaled": train_history[-1],
        "final_validation_loss_scaled": validation_history[-1],
        "provenance": [
            "ICML2026/source/supp.tex, Figure 9 protocol",
            "code/2d_case/2d_f1*.ipynb legacy plotting/model branches",
            "code/1d_case_missing_data/3_Redo_PaperExperiments_CauchyNet.ipynb",
            "reproducibility_audit/notebook_reruns/run_redo_paper_experiments.py and run_redo.log",
        ],
        "audit_note": "The earlier 500-epoch draft protocol does not recover the displayed residual range. Full-grid and held-out test metrics for the verified 20,000-epoch protocol are both reported here.",
    }
    result_path = RESULT_DIR / f"figure09_2d_surface{suffix}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
