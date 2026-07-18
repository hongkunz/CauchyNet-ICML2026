"""Faithfully reimplement Figure 1: rational-spike fitting and loss curves.

Paper specification
-------------------
Target:
    g(x) = sin(3x) + 4 / ((x - 0.5)^2 + 0.01), x in [-1, 1].

Comparison:
    A trainable-pole CauchyNet and a width-matched one-hidden-layer ReLU
    feed-forward network under identical data and optimization budgets.

Figure:
    Left: target and model predictions.
    Right: training and validation real-output MSE over 500 epochs. Curves are
    medians across 10 independent model-initialization runs and shaded bands
    are plus/minus one standard deviation, matching the PDF caption.

The archived manuscript contains the final vector graphic but not its original
generating script or per-seed curves. This implementation therefore follows
the caption and the closest archived legacy implementation. Choices not stated
in the caption use supplementary Table 3: width 128, batch size 32, Adam at
0.01, StepLR factor 0.5 every 100 epochs, weight decay 1e-4, split seed 10,
and imaginary-output penalty 0.1. The legacy pipeline fits MinMaxScaler before
the random split, so this script does the same. It is a protocol-faithful
reimplementation, not a claim of bitwise recovery of the missing source data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"


@dataclass(frozen=True)
class Config:
    hidden_size: int = 128
    data_points: int = 200
    train_points: int = 150
    batch_size: int = 32
    epochs: int = 500
    runs: int = 10
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    imaginary_penalty: float = 0.1
    split_seed: int = 10
    pole_real_radius: float = 1.5
    pole_imag_radius: float = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def target(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(3.0 * x) + 4.0 / ((x - 0.5) ** 2 + 0.01)


class TrainablePoleCauchyNet(nn.Module):
    """One-dimensional CauchyNet with trainable poles and coefficients."""

    def __init__(self, hidden_size: int, real_radius: float, imag_radius: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.coefficients = nn.Parameter(
            torch.normal(0.0, 1.0, size=(hidden_size, 1), dtype=torch.cfloat)
        )
        angles = 2.0 * math.pi * torch.rand(hidden_size)
        initial_poles = torch.complex(
            real_radius * torch.cos(angles),
            imag_radius * torch.sin(angles),
        )
        self.poles = nn.Parameter(initial_poles)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_complex = torch.complex(x, torch.zeros_like(x))
        features = 1.0 / (self.poles.unsqueeze(0) - x_complex + 1e-8)
        output = features @ self.coefficients / self.hidden_size
        return output.real, output.imag


class ReLUFeedForwardNet(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_data(config: Config) -> dict[str, torch.Tensor | float]:
    x_all = torch.linspace(-1.0, 1.0, config.data_points).unsqueeze(1)
    y_all_raw = target(x_all)

    generator = torch.Generator().manual_seed(config.split_seed)
    permutation = torch.randperm(config.data_points, generator=generator)
    train_idx = permutation[: config.train_points]
    validation_idx = permutation[config.train_points :]

    # Match the archived legacy helpers: normalize the full deterministic
    # target grid before splitting.
    y_min = float(y_all_raw.min())
    y_range = float(y_all_raw.max() - y_all_raw.min())
    if y_range <= 0:
        raise ValueError("Training targets have zero range; cannot MinMax-scale.")

    y_all_scaled = (y_all_raw - y_min) / y_range
    return {
        "x_all": x_all,
        "y_all_raw": y_all_raw,
        "x_train": x_all[train_idx],
        "y_train": y_all_scaled[train_idx],
        "x_validation": x_all[validation_idx],
        "y_validation": y_all_scaled[validation_idx],
        "y_min": y_min,
        "y_range": y_range,
    }


def train_one_run(
    model: nn.Module,
    data: dict[str, torch.Tensor | float],
    config: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(data["x_train"], data["y_train"]),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step,
        gamma=config.scheduler_gamma,
    )

    training_mse: list[float] = []
    validation_mse: list[float] = []

    for _ in range(config.epochs):
        model.train()
        squared_error_sum = 0.0
        observation_count = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            if isinstance(output, tuple):
                prediction, imaginary = output
                real_mse = criterion(prediction, y_batch)
                loss = real_mse + config.imaginary_penalty * criterion(
                    imaginary, torch.zeros_like(imaginary)
                )
            else:
                prediction = output
                real_mse = criterion(prediction, y_batch)
                loss = real_mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = len(x_batch)
            squared_error_sum += float(real_mse.detach()) * batch_size
            observation_count += batch_size

        scheduler.step()
        training_mse.append(squared_error_sum / observation_count)

        model.eval()
        with torch.no_grad():
            validation_output = model(data["x_validation"])
            if isinstance(validation_output, tuple):
                validation_output = validation_output[0]
            validation_mse.append(
                float(criterion(validation_output, data["y_validation"]))
            )

    model.eval()
    with torch.no_grad():
        prediction = model(data["x_all"])
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction_raw = (
            prediction.flatten() * float(data["y_range"]) + float(data["y_min"])
        )

    return (
        np.asarray(training_mse),
        np.asarray(validation_mse),
        prediction_raw.cpu().numpy(),
    )


def summarize_curves(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(curves)
    return np.median(stacked, axis=0), np.std(stacked, axis=0)


def run_experiment(config: Config) -> dict[str, object]:
    data = build_data(config)
    records: dict[str, dict[str, list[np.ndarray]]] = {
        "CauchyNet": {"train": [], "validation": [], "prediction": []},
        "FFN": {"train": [], "validation": [], "prediction": []},
    }

    constructors = {
        "CauchyNet": lambda: TrainablePoleCauchyNet(
            config.hidden_size,
            config.pole_real_radius,
            config.pole_imag_radius,
        ),
        "FFN": lambda: ReLUFeedForwardNet(config.hidden_size),
    }

    for model_name, constructor in constructors.items():
        for run_seed in range(config.runs):
            set_seed(run_seed)
            model = constructor()
            train_curve, validation_curve, prediction = train_one_run(
                model, data, config, run_seed
            )
            records[model_name]["train"].append(train_curve)
            records[model_name]["validation"].append(validation_curve)
            records[model_name]["prediction"].append(prediction)

    result: dict[str, object] = {
        "config": asdict(config),
        "target": "sin(3*x) + 4/((x-0.5)^2+0.01)",
        "models": {},
    }
    for model_name in records:
        train_median, train_std = summarize_curves(records[model_name]["train"])
        validation_median, validation_std = summarize_curves(
            records[model_name]["validation"]
        )
        predictions = np.stack(records[model_name]["prediction"])
        final_validation = np.asarray(
            [curve[-1] for curve in records[model_name]["validation"]]
        )
        representative_run = int(
            np.argmin(np.abs(final_validation - np.median(final_validation)))
        )
        result["models"][model_name] = {
            "train_median": train_median,
            "train_std": train_std,
            "validation_median": validation_median,
            "validation_std": validation_std,
            "prediction_representative": predictions[representative_run],
            "prediction_std": np.std(predictions, axis=0),
            "representative_run_seed": representative_run,
            "final_train_mse_median": float(train_median[-1]),
            "final_validation_mse_median": float(validation_median[-1]),
        }

    result["x"] = data["x_all"].flatten().numpy()
    result["target_values"] = data["y_all_raw"].flatten().numpy()
    result["provenance_status"] = (
        "protocol-faithful reimplementation; original per-seed Figure 1 "
        "curves are absent from the archive"
    )
    return result


def plot_result(result: dict[str, object], output_stem: Path) -> None:
    x = result["x"]
    target_values = result["target_values"]
    models = result["models"]
    epochs = np.arange(1, result["config"]["epochs"] + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(14, 6.2))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.0, 1.7])

    for row, model_name, color in (
        (0, "CauchyNet", "tab:orange"),
        (1, "FFN", "tab:blue"),
    ):
        axis = figure.add_subplot(grid[row, 0])
        axis.plot(x, target_values, "k--", linewidth=2.1, label="True")
        axis.plot(
            x,
            models[model_name]["prediction_representative"],
            color=color,
            linewidth=2.2,
            label=model_name,
        )
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(0.0, 420.0)
        axis.set_xticks([-1.0, 0.0, 1.0])
        axis.set_yticks([0.0, 250.0])
        axis.set_title(model_name, fontsize=14)
        axis.legend()

    axis = figure.add_subplot(grid[:, 1])
    for model_name, color in (("CauchyNet", "tab:orange"), ("FFN", "tab:blue")):
        model_result = models[model_name]
        for split, linestyle, alpha in (
            ("train", "-", 1.0),
            ("validation", "--", 0.58),
        ):
            median = model_result[f"{split}_median"]
            standard_deviation = model_result[f"{split}_std"]
            # Keep the band inside the published log-axis window. Directly
            # subtracting one standard deviation from a tiny positive loss
            # can otherwise create meaningless near-zero vertical spikes.
            lower = np.maximum(median - standard_deviation, 3e-6)
            upper = median + standard_deviation
            label = model_name if split == "train" else None
            axis.plot(
                epochs,
                median,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                alpha=alpha,
                label=label,
            )
            axis.fill_between(epochs, lower, upper, color=color, alpha=0.10)

    axis.set_yscale("log")
    axis.set_ylim(3e-6, 2e-1)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE Loss (log scale)")
    axis.set_title("Training & Validation (Median ± Std)", fontsize=15)
    axis.legend()
    figure.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two seeds for five epochs and write separate smoke outputs.",
    )
    args = parser.parse_args()

    config = Config()
    suffix = ""
    if args.smoke_test:
        config = Config(epochs=5, runs=2)
        suffix = "_smoke"

    result = run_experiment(config)
    figure_stem = FIGURE_DIR / f"figure01_rational_spike{suffix}"
    result_path = RESULT_DIR / f"figure01_rational_spike{suffix}.json"
    plot_result(result, figure_stem)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(json_ready(result), indent=2) + "\n")

    for model_name, model_result in result["models"].items():
        print(
            f"{model_name}: final train MSE median="
            f"{model_result['final_train_mse_median']:.6g}, "
            f"final validation MSE median="
            f"{model_result['final_validation_mse_median']:.6g}"
        )
    print(f"Saved {figure_stem.with_suffix('.pdf')}")
    print(f"Saved {figure_stem.with_suffix('.png')}")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
