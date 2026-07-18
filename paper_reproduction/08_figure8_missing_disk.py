"""Reproduce Figure 8: two-dimensional missing-disk imputation.

The default configuration reconstructs the archived panel using the internally
consistent legacy branch: width 128, seed 10, 1,000 epochs, no weight decay,
full imaginary-part penalty, the best-validation checkpoint, and the plot's
documented +1 display-only surface offset.  Use ``--paper-stated`` to run the
literal 200-epoch common setup stated in ``ICML2026/source/supp.tex``:

* 3,000 uniform samples on [-0.8, 0.8]^2, seed 10;
* radius-0.3 disk around the origin held out for testing;
* the remaining points split 60/40 into train/validation;
* a width-128 trainable-pole CauchyNet, batch size 32, Adam at 0.01,
  weight decay 1e-4, StepLR(100, 0.5), 200 epochs, and lambda_imag=0.1;
* target MinMax scaling fitted to train+validation, matching the valid legacy
  missing-disk cell in ``code/2d_case/2d_example.ipynb``.

The legacy notebook also contains a later, incompatible [-0.2, 0.2]^2 cell.
Because its radius-0.3 holdout consumes every sample and the cell crashes, it
is deliberately excluded here.
"""

from __future__ import annotations

import argparse
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
    """g(x,y)=3-x^2+xy-y^2-1/(5+(x-1)^2)."""
    x1, x2 = x[:, 0], x[:, 1]
    return 3.0 - x1.square() + x1 * x2 - x2.square() - 1.0 / (5.0 + (x1 - 1.0).square())


class CauchyNet2D(nn.Module):
    """One hidden layer of product Cauchy atoms with trainable complex poles."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        pole_real = torch.randn(2, width)
        pole_imag = torch.randn(2, width)
        self.poles = nn.Parameter(torch.complex(pole_real, pole_imag))
        coeff_real = 0.01 * torch.randn(width, 1)
        coeff_imag = 0.01 * torch.randn(width, 1)
        self.coefficients = nn.Parameter(torch.complex(coeff_real, coeff_imag))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.complex(x, torch.zeros_like(x))
        differences = self.poles.T.unsqueeze(0) - z.unsqueeze(1)
        atoms = 1.0 / (torch.prod(differences, dim=2) + 1e-8)
        output = atoms @ self.coefficients
        return output.real.squeeze(1), output.imag.squeeze(1)


def make_split(seed: int) -> dict[str, torch.Tensor]:
    set_seed(seed)
    points = 1.6 * torch.rand(3000, 2) - 0.8
    values = target(points)
    in_disk = torch.linalg.vector_norm(points, dim=1) < 0.3
    outside_points = points[~in_disk]
    outside_values = values[~in_disk]
    n_train = int(0.6 * len(outside_points))
    return {
        "train_x": outside_points[:n_train],
        "train_y": outside_values[:n_train],
        "val_x": outside_points[n_train:],
        "val_y": outside_values[n_train:],
        "test_x": points[in_disk],
        "test_y": values[in_disk],
    }


def train_model(
    split: dict[str, torch.Tensor], seed: int, epochs: int, width: int,
    learning_rate: float, weight_decay: float, imaginary_penalty: float,
    step_size: int, gamma: float, device: torch.device
) -> tuple[CauchyNet2D, list[float], list[float], float, float]:
    # The valid legacy cell fits the target scaler on train+validation.
    fit_y = torch.cat([split["train_y"], split["val_y"]])
    y_min, y_max = float(fit_y.min()), float(fit_y.max())
    y_range = y_max - y_min

    def scale(y: torch.Tensor) -> torch.Tensor:
        return (y - y_min) / y_range

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(split["train_x"], scale(split["train_y"])),
        batch_size=32,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(split["val_x"], scale(split["val_y"])),
        batch_size=32,
        shuffle=False,
    )

    set_seed(seed)
    model = CauchyNet2D(width=width).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    mse = nn.MSELoss()
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        running = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_real, pred_imag = model(batch_x)
            loss = mse(pred_real, batch_y) + imaginary_penalty * mse(
                pred_imag, torch.zeros_like(pred_imag)
            )
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(batch_x)
        train_losses.append(running / len(train_loader.dataset))

        model.eval()
        running = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_real, pred_imag = model(batch_x)
                loss = mse(pred_real, batch_y) + imaginary_penalty * mse(
                    pred_imag, torch.zeros_like(pred_imag)
                )
                running += float(loss) * len(batch_x)
        val_losses.append(running / len(val_loader.dataset))
        if val_losses[-1] < best_validation:
            best_validation = val_losses[-1]
            best_epoch = len(val_losses)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        scheduler.step()

    elapsed = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    model.y_min = y_min  # type: ignore[attr-defined]
    model.y_range = y_range  # type: ignore[attr-defined]
    model.best_epoch = best_epoch  # type: ignore[attr-defined]
    model.best_validation_loss = best_validation  # type: ignore[attr-defined]
    return model, train_losses, val_losses, y_min, elapsed


def predict_unscaled(model: CauchyNet2D, points: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in points.split(8192):
            pred, _ = model(chunk.to(device))
            pred = pred * model.y_range + model.y_min  # type: ignore[attr-defined]
            predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions)


def make_figure(
    split: dict[str, torch.Tensor], model: CauchyNet2D, device: torch.device,
    stem: Path, display_offset: float
) -> dict[str, float]:
    grid_axis = torch.linspace(-0.8, 0.8, 180)
    grid_x, grid_y = torch.meshgrid(grid_axis, grid_axis, indexing="xy")
    grid_points = torch.column_stack([grid_x.ravel(), grid_y.ravel()])
    truth = target(grid_points).numpy().reshape(grid_x.shape)
    prediction = predict_unscaled(model, grid_points, device).reshape(grid_x.shape)
    error = truth - prediction

    test_prediction = predict_unscaled(model, split["test_x"], device)
    test_truth = split["test_y"].numpy()
    metrics = {
        "test_mse": float(np.mean((test_truth - test_prediction) ** 2)),
        "test_mae": float(np.mean(np.abs(test_truth - test_prediction))),
        "test_max_abs_error": float(np.max(np.abs(test_truth - test_prediction))),
        "grid_error_min": float(error.min()),
        "grid_error_max": float(error.max()),
        "disk_grid_max_abs_error": float(np.max(np.abs(error[(grid_x.numpy() ** 2 + grid_y.numpy() ** 2) < 0.3 ** 2]))),
    }

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12.0, 5.0))
    left = fig.add_subplot(121, projection="3d")
    left.plot_surface(grid_x, grid_y, truth, cmap="viridis", alpha=0.72, linewidth=0)
    train_x = split["train_x"].numpy()
    train_z = split["train_y"].numpy() + 0.01
    test_x = split["test_x"].numpy()
    test_z = split["test_y"].numpy() + 0.012
    left.scatter(train_x[:, 0], train_x[:, 1], train_z, s=3, c="#2454d6", alpha=0.45, label="train")
    left.scatter(test_x[:, 0], test_x[:, 1], test_z, s=5, c="#d62728", alpha=0.8, label="held-out disk")
    left.set_title("Data split on the target surface")
    left.set_xlabel("x")
    left.set_ylabel("y")
    left.set_zlabel("g(x,y)")
    left.view_init(elev=28, azim=55)
    left.legend(loc="upper left", fontsize=8)

    right = fig.add_subplot(122, projection="3d")
    displayed_prediction = prediction + display_offset
    right.plot_surface(
        grid_x, grid_y, displayed_prediction, cmap="Spectral", alpha=0.95,
        linewidth=0
    )
    # The archived panel offsets only the predicted surface by +1 and leaves
    # the signed-error projection near z=0.  Preserve that visual convention.
    floor = -0.2 if display_offset else float(displayed_prediction.min()) - 0.15
    norm = Normalize(vmin=float(error.min()), vmax=float(error.max()))
    right.contourf(grid_x.numpy(), grid_y.numpy(), error, zdir="z", offset=floor,
                   levels=40, cmap="viridis", norm=norm)
    scalar = cm.ScalarMappable(norm=norm, cmap="viridis")
    scalar.set_array(error)
    fig.colorbar(scalar, ax=right, shrink=0.58, pad=0.08, label="Signed error (true - predicted)")
    right.set_zlim(floor, float(displayed_prediction.max()) + 0.1)
    right.set_title("CauchyNet prediction and signed error")
    right.set_xlabel("x")
    right.set_ylabel("y")
    right.set_zlabel("prediction")
    right.view_init(elev=28, azim=55)

    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--imaginary-penalty", type=float, default=1.0)
    parser.add_argument("--step-size", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--tag", default="", help="Suffix for a tuning run; does not overwrite the paper-protocol output.")
    parser.add_argument(
        "--display-offset", type=float, default=1.0,
        help="Vertical display-only offset. The archived right panel used +1.0."
    )
    parser.add_argument(
        "--paper-stated", action="store_true",
        help="Use the literal common setup: 200 epochs, wd=1e-4, lambda=0.1, StepLR(100,0.5), no display offset."
    )
    parser.add_argument("--smoke", action="store_true", help="Run two epochs and use a separate output stem.")
    args = parser.parse_args()
    if args.paper_stated:
        args.epochs = 200
        args.weight_decay = 1e-4
        args.imaginary_penalty = 0.1
        args.step_size = 100
        args.gamma = 0.5
        args.display_offset = 0.0
        if not args.tag:
            args.tag = "paper_stated"
    epochs = 2 if args.smoke else args.epochs
    suffix = "_smoke" if args.smoke else (f"_{args.tag}" if args.tag else "")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    split = make_split(args.seed)
    model, train_losses, val_losses, y_min, elapsed = train_model(
        split, args.seed, epochs, args.width, args.learning_rate,
        args.weight_decay, args.imaginary_penalty, args.step_size, args.gamma,
        device,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / f"figure08_missing_disk{suffix}"
    metrics = make_figure(split, model, device, stem, args.display_offset)
    result = {
        "protocol": {
            "seed": args.seed,
            "domain": [-0.8, 0.8],
            "sample_count": 3000,
            "missing_disk_radius": 0.3,
            "train_count": len(split["train_x"]),
            "validation_count": len(split["val_x"]),
            "test_count": len(split["test_x"]),
            "width": args.width,
            "batch_size": 32,
            "epochs": epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "step_lr": {"step_size": args.step_size, "gamma": args.gamma},
            "imaginary_penalty": args.imaginary_penalty,
            "target_scaler_fit": "train+validation (valid legacy missing-disk cell)",
            "checkpoint": "lowest validation loss",
            "best_epoch": model.best_epoch,  # type: ignore[attr-defined]
            "best_validation_loss_scaled": model.best_validation_loss,  # type: ignore[attr-defined]
            "display_only_vertical_offset": args.display_offset,
        },
        "metrics": metrics,
        "training_seconds": elapsed,
        "final_train_loss_scaled": train_losses[-1],
        "final_validation_loss_scaled": val_losses[-1],
        "target_min": y_min,
        "provenance": [
            "ICML2026/source/supp.tex, Figure 8 protocol",
            "code/2d_case/2d_example.ipynb valid [-0.8,0.8]^2 missing-disk cells",
            "reproducibility_audit/notebook_reruns/run_2d_example.py (legacy audit and failure trace)",
        ],
        "excluded_legacy_cell": "[-0.2,0.2]^2 with radius 0.3: zero train/validation samples",
    }
    result_path = RESULT_DIR / f"figure08_missing_disk{suffix}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
