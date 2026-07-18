"""Self-contained legacy model and training definitions for Figures 12--14.

These definitions preserve the runnable behavior recovered from the original
notebook helpers while removing the clean release's dependency on ``code/``.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class ReciprocalActivation(nn.Module):
    def __init__(self, eps: float = 1e-20) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return 1.0 / (value + self.eps)


class CauchyNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lambda_ = nn.Parameter(
            torch.normal(0.0, 1.0, (hidden_size, output_size), dtype=torch.cfloat)
        )
        self.xi = nn.Parameter(
            torch.normal(0.0, 1.0, (hidden_size,), dtype=torch.cfloat)
        )
        self.activation = ReciprocalActivation()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        batch_size = x.size(0)
        complex_x = torch.complex(x, torch.zeros_like(x))
        features = self.activation(
            self.xi.unsqueeze(0).expand(batch_size, -1)
            - complex_x.expand(batch_size, self.hidden_size)
        )
        output = features @ self.lambda_ / self.hidden_size
        return output.real, output.imag


class CauchyNet1(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lambda_ = nn.Parameter(
            torch.normal(0.0, 1.0, (hidden_size, output_size), dtype=torch.cfloat)
        )
        angles = 2.0 * torch.pi * torch.rand(hidden_size)
        self.xi = nn.Parameter(
            torch.complex(1.5 * torch.cos(angles), 0.5 * torch.sin(angles))
        )
        self.activation = ReciprocalActivation()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        batch_size = x.size(0)
        complex_x = torch.complex(x, torch.zeros_like(x))
        features = self.activation(
            self.xi.unsqueeze(0).expand(batch_size, -1)
            - complex_x.expand(batch_size, self.hidden_size)
        )
        output = features @ self.lambda_ / self.hidden_size
        return output.real, output.imag


class SIREN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        w0_initial: float = 30.0,
        w0_hidden: float = 30.0,
    ) -> None:
        super().__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.w0_initial = w0_initial
        self.w0_hidden = w0_hidden
        with torch.no_grad():
            nn.init.uniform_(
                self.linear_in.weight,
                -1.0 / self.linear_in.in_features,
                1.0 / self.linear_in.in_features,
            )
            nn.init.zeros_(self.linear_in.bias)
            self.linear_in.weight *= self.w0_initial
            bound = math.sqrt(6.0 / self.linear_out.in_features)
            nn.init.uniform_(
                self.linear_out.weight,
                -bound / self.w0_hidden,
                bound / self.w0_hidden,
            )
            nn.init.zeros_(self.linear_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.linear_out(torch.sin(self.linear_in(x)))


class RBFNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(hidden_size, input_size))
        self.log_sigmas = nn.Parameter(torch.zeros(hidden_size))
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        distance = (x.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(dim=2)
        sigma = torch.exp(self.log_sigmas).unsqueeze(0)
        return self.linear(torch.exp(-distance / (2.0 * sigma.pow(2))))


class LSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


class MinimalTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        nhead: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        embedded = self.dropout(self.embedding(x).unsqueeze(1) + self.positional_encoding)
        return self.fc_out(self.encoder(embedded).squeeze(1))


class MinimalInformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        nhead: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        embedded = self.embedding(x)
        gated = self.dropout((torch.sigmoid(self.gate(embedded)) * embedded).unsqueeze(1))
        return self.fc_out(self.encoder(gated).squeeze(1))


class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.relu(self.fc1(x)))


class NBeats(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_blocks: int = 1
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            NBeatsBlock(input_size, hidden_size, output_size) for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        residual = x
        output = None
        for block in self.blocks:
            output = block(residual)
            residual = residual - output
        assert output is not None
        return output


def loadData(
    x: torch.Tensor, y: torch.Tensor, batchSize: int = 256
) -> tuple[DataLoader, DataLoader, DataLoader, torch.utils.data.Subset]:
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    dataset = TensorDataset(x, y)
    test_size = int(len(dataset) * 0.25)
    validation_size = test_size
    training_size = len(dataset) - test_size - validation_size
    training, validation, test = random_split(
        dataset, [training_size, validation_size, test_size]
    )
    return (
        DataLoader(training, batch_size=batchSize, shuffle=True, num_workers=0),
        DataLoader(validation, batch_size=batchSize, shuffle=False, num_workers=0),
        DataLoader(test, batch_size=batchSize, shuffle=False, num_workers=0),
        test,
    )


def train_and_evaluate_model(
    model_constructor,
    input_size: int,
    hidden_size: int,
    output_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    lr: float = 0.01,
    epochs: int = 200,
    device: torch.device | None = None,
    scaler=None,
    a: float = 1.0,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_constructor(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=5
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation = float("inf")
    best_state = None
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    started = time.time()
    for _ in range(epochs):
        model.train()
        total = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            if isinstance(output, tuple):
                real, imaginary = output
                loss = criterion(real, y_batch.unsqueeze(-1)) + a * criterion(
                    imaginary, torch.zeros_like(imaginary)
                )
            else:
                loss = criterion(output, y_batch.unsqueeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss)
        train_losses.append(total / len(train_loader))
        model.eval()
        total = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                if isinstance(output, tuple):
                    real, imaginary = output
                    loss = criterion(real, y_batch.unsqueeze(-1)) + a * criterion(
                        imaginary, torch.zeros_like(imaginary)
                    )
                else:
                    loss = criterion(output, y_batch.unsqueeze(-1))
                total += float(loss)
        validation_loss = total / len(val_loader)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            # Preserve the original helper's state_dict behavior exactly.
            best_state = model.state_dict()
        scheduler.step(validation_loss)
    training_time = time.time() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    predictions, truths = [], []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch.to(device))
            scaled_prediction = output[0] if isinstance(output, tuple) else output
            scaled_prediction = scaled_prediction.cpu().numpy().ravel()
            scaled_truth = y_batch.numpy().ravel()
            if scaler is not None:
                prediction = scaler.inverse_transform(scaled_prediction[:, None]).ravel()
                truth = scaler.inverse_transform(scaled_truth[:, None]).ravel()
            else:
                prediction, truth = scaled_prediction, scaled_truth
            predictions.append(prediction)
            truths.append(truth)
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    return (
        model,
        train_losses,
        validation_losses,
        mean_squared_error(truth, prediction),
        mean_absolute_error(truth, prediction),
        prediction,
        truth,
        training_time,
        parameter_count,
    )
