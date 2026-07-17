
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import math


from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1) Data Loading
##############################################################################
def loadData(X, Y, batchSize=64):
    """
    Splits dataset into 50%/25%/25% for train/val/test.
    Returns train_loader, val_loader, test_loader, ds_test.
    """
    # if X.dim() == 1:
    #     X = X.unsqueeze(-1)
    dataset    = TensorDataset(X, Y)
    total_size = len(dataset)
    testSize   = int(total_size * 0.25)
    valSize    = testSize
    trainSize  = total_size - testSize - valSize

    ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
    train_loader = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=batchSize, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=batchSize, shuffle=False)
    return train_loader, val_loader, test_loader, ds_test

##############################################################################
# 0) Global Plot Style
##############################################################################
def configure_plot_style():
    """
    Configure global matplotlib + LaTeX-based styles for a JMLR-like submission.
    We'll keep alpha on lines/fills for a 'fancy' look.
    """
    mpl.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts,amsmath,amssymb}',
        'font.family': 'sans-serif',
        'figure.dpi': 300,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'grid.alpha': 0.3,
        'legend.fontsize': 9,
        'lines.linewidth': 1.6,
        'lines.markersize': 6,
    })

configure_plot_style()

# 2) Example CauchyNet with fixed xi, trainable lambda
##############################################################################
class ReciprocalActivation(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (x + self.epsilon)

class CauchyNet(nn.Module):
    """
    Only lambda_ is trainable; xi is fixed (angles).
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Trainable lambda
        self.lambda_ = nn.Parameter(
            torch.randn(hidden_size, output_size, dtype=torch.cfloat)
        )

        # Fix xi in a buffer (not trained)
        angles = 2*math.pi*torch.rand(hidden_size)
        real_part = 3.0*torch.cos(angles)
        imag_part = 0.5*torch.sin(angles)

        # register_buffer(...) => NOT a Parameter => not trainable
        self.register_buffer("xi_fixed", torch.complex(real_part, imag_part))

        self.activation = ReciprocalActivation()

    def forward(self, x: torch.Tensor):
        # x shape [B,1]
        B = x.shape[0]
        x_flat = x.view(B)  # [B]
        x_c    = torch.complex(x_flat, torch.zeros_like(x_flat))  # [B]
        x_expanded  = x_c.unsqueeze(1).expand(B, self.hidden_size)     # => [B, hidden_size]
        xi_expanded = self.xi_fixed.unsqueeze(0).expand(B, self.hidden_size)

        activated = self.activation(xi_expanded - x_expanded)  # [B, hidden_size]
        out_c     = torch.matmul(activated, self.lambda_) / self.hidden_size  # => [B, out_features]
        return out_c.real, out_c.imag

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch
import torch.nn as nn

class MinimalTransformer(nn.Module):
    """
    Minimal Transformer for scalar input -> scalar output, returning (y_real, y_imag).
    The imaginary part is zero to match CauchyNet's (real, imag) interface.
    """
    def __init__(self, input_size, hidden_size, output_size,
                 nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)

        # Positional encoding as a learned parameter of shape (1, seq_len=1, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # A basic Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Expects x to have shape:
          - (batch_size,)       -> unsqueezed to (batch_size,1)
          - (batch_size,1)
        Returns (y_real, y_imag), each of shape (batch_size, output_size).
        """
        # 1) If x is 1D, reshape to (batch_size,1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)    # => (batch_size,1)

        # 2) Now x is (batch_size, input_size). For scalar input_size=1 -> shape (batch_size,1)
        #    Embedding => (batch_size, hidden_size)
        x = self.embedding(x)

        # 3) Insert seq_len=1 dimension => (batch_size, seq_len=1, hidden_size)
        x = x.unsqueeze(1)

        # 4) Add positional encoding => shape still (batch_size, 1, hidden_size)
        x = x + self.positional_encoding

        # 5) Dropout
        x = self.dropout(x)

        # 6) Transformer encoder => (batch_size, 1, hidden_size)
        x = self.encoder(x)    # with batch_first=True

        # 7) Squeeze out seq_len => (batch_size, hidden_size)
        x = x.squeeze(1)

        # 8) Final FC layer => (batch_size, output_size)
        y = self.fc_out(x)


        return y
        
# Define the FNN model
class FNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Adjust the size as needed
        #self.fc2 = nn.Linear(400, 400)  # Hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)   # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out




# Define the RNN model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.fc(x)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        return self.fc(x)

import torch
import torch.nn as nn
class SIREN(nn.Module):
    """SIREN: Implicit Neural Representations with Periodic Activation Functions."""
    def __init__(self, input_size, hidden_size, output_size, w0_initial=30.0, w0_hidden=30.0):
        super().__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.w0_initial = w0_initial
        self.w0_hidden = w0_hidden
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.uniform_(self.linear_in.weight, -1.0 / self.linear_in.in_features, 1.0 / self.linear_in.in_features)
            nn.init.zeros_(self.linear_in.bias)
            self.linear_in.weight *= self.w0_initial
            bound = (6.0 / self.linear_out.in_features) ** 0.5
            nn.init.uniform_(self.linear_out.weight, -bound / self.w0_hidden, bound / self.w0_hidden)
            nn.init.zeros_(self.linear_out.bias)

    def sine(self, x, w0):
        return torch.sin(w0 * x)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = self.sine(self.linear_in(x), w0=1.0)
        out = self.linear_out(h)
        return out
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class NBeatsBlock(nn.Module):
    """
    A single minimal N-BEATS block:
      - One or more fully-connected layers with ReLU
      - An output layer that predicts part of the forecast
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Uncomment or add more layers if you like:
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x shape: (batch_size, input_size)
        returns shape: (batch_size, output_size)
        """
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        out = self.fc_out(x)
        return out

class NBeats(nn.Module):
    """
    Minimal N-BEATS model for 1D inputs and outputs:
      - Stacks multiple blocks
      - Uses a "residual" trick: each block refines the forecast
    """
    def __init__(self, input_size, hidden_size, output_size, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size, output_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        x shape: [batch_size] or [batch_size, input_size].
        
        If x is shape [batch_size], we'll unsqueeze so it becomes [batch_size, 1].
        Then each block produces an 'out', and we update:
            residual = residual - out
        The last block's 'out' is returned as final forecast.
        """
        # Ensure x has 2D shape: (batch_size, input_size)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # shape => [batch_size, 1]

        residual = x
        final_out = None
        for block in self.blocks:
            out = block(residual)
            residual = residual - out
            final_out = out  # keep track of the last block's output

        # final_out shape => (batch_size, output_size)
        return final_out


class RBFNetwork(nn.Module):
    """Radial Basis Function Network."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(hidden_size, input_size))
        self.log_sigmas = nn.Parameter(torch.zeros(hidden_size))
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_expanded = x.unsqueeze(1)
        centers_expanded = self.centers.unsqueeze(0)
        dist_sq = (x_expanded - centers_expanded).pow(2).sum(dim=2)
        sigma = torch.exp(self.log_sigmas).unsqueeze(0)
        rbf_activations = torch.exp(-dist_sq / (2 * sigma.pow(2)))
        out = self.linear(rbf_activations)
        return out

import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model(
    net,
    train_loader,
    val_loader,
    test_loader,
    model_name,
    epochs=2000,
    lr=0.01,
    scheduler_step_size=500,
    scheduler_gamma=0.8
):
    """
    Trains 'net' for up to 'epochs' using MSELoss + Adam + StepLR.
    Handles both standard nets returning a single Tensor
    and 'CauchyNet' returning (y_real, y_imag).
    
    Parameters
    ----------
    net : nn.Module
        The model to be trained (e.g. standard or CauchyNet).
    train_loader, val_loader, test_loader : DataLoader
        DataLoaders for training, validation, and testing sets.
    model_name : str
        Name for logging and saving best model.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate for Adam.
    scheduler_step_size : int
        Step size for StepLR.
    scheduler_gamma : float
        Gamma factor for StepLR.

    Returns
    -------
    loss_history : list of float
        Per-epoch average training loss.
    val_loss_history : list of float
        Per-epoch average validation loss.
    test_data : list of float
        X-values from test set (in un-squeezed form).
    test_predictions : list of float
        Predictions on test set, shape matching test_data.
    test_targets : list of float
        Ground-truth values from test set.
    """
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
    )
    loss_function = nn.MSELoss()

    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_model_state = None

    start_time = time.time()

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0

        # === Training loop (standard PyTorch approach) ===
        for inputs, labels in train_loader:
            # For standard networks, we often want shape [batch_size, 1].
            # If your net expects [B,1], do unsqueeze.
            inputs = inputs.unsqueeze(-1)   # => shape [B,1]
            labels = labels.unsqueeze(-1)   # => shape [B,1]

            optimizer.zero_grad()

            outputs = net(inputs)  # Could be a single Tensor or (real,imag)

            if isinstance(outputs, tuple):
                # CauchyNet => outputs = (y_real, y_imag)
                y_real, y_imag = outputs
                zeros_imag = torch.zeros_like(y_imag)
                loss = (
                    loss_function(y_real, labels) +
                    loss_function(y_imag, zeros_imag)
                )
            else:
                # Standard model => outputs = Tensor
                loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Step the scheduler (no val_loss usage here, just stepping every epoch)
        scheduler.step()


        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # === Validation loop ===
        net.eval()
        val_loss_acc = 0.0
        with torch.no_grad():
            for v_in, v_lb in val_loader:
                v_in = v_in.unsqueeze(-1)
                v_lb = v_lb.unsqueeze(-1)

                out_val = net(v_in)
                if isinstance(out_val, tuple):
                    vr, vi = out_val
                    zeros_im = torch.zeros_like(vi)
                    vloss = (
                        loss_function(vr, v_lb) +
                        loss_function(vi, zeros_im)
                    )
                else:
                    vloss = loss_function(out_val, v_lb)

                val_loss_acc += vloss.item()

        avg_val_loss = val_loss_acc / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # Best model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = net.state_dict()

        if (epoch + 1) % 500 == 0:
            print(
                f"{model_name} Epoch {epoch+1}/{epochs}, "
                f"Train Loss={avg_loss:.6f}, Val Loss={avg_val_loss:.6f}"
            )

    end_time = time.time()
         
    training_time = end_time - start_time
    print(f"\nFinished Training {model_name}. Total Time: {end_time - start_time:.2f} seconds")

    # === Load best model if found ===
    if best_model_state is not None:
        best_model_filename = f"{model_name}_best.pth"
        net.load_state_dict(best_model_state)
        torch.save(best_model_state, best_model_filename)
        print(f"Best {model_name} model saved with val_loss={best_val_loss:.6f}.\n")
    else:
        print("No improvement found, best model not saved.\n")

    # === Testing phase ===
    net.eval()
    test_predictions = []
    test_targets = []
    test_data = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(-1)  # shape [B,1]
            labels = labels.unsqueeze(-1)  # shape [B,1]

            out_test = net(inputs)
            if isinstance(out_test, tuple):
                y_r, y_i = out_test
                preds = y_r.cpu().numpy().flatten()
            else:
                preds = out_test.cpu().numpy().flatten()

            test_predictions.extend(preds)
            test_targets.extend(labels.cpu().numpy().flatten())
            test_data.extend(inputs.cpu().numpy().flatten())

    test_mse = mean_squared_error(test_targets, test_predictions)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    print(
        f"{model_name} => Test MSE={test_mse:.6f}, "
        f"Test MAE={test_mae:.6f}"
    )

    return (loss_history, val_loss_history, test_data, test_predictions, test_targets, training_time,
            Num_Params)

import numpy as np
import torch
import matplotlib.pyplot as plt

# If you use Seaborn styling:
import seaborn as sns
sns.set_style("whitegrid")  # or "ticks"
sns.set_context("talk")     # e.g. "paper", "talk", or "poster"

############################################################
# A) Plot the Train/Val/Test Sets
############################################################

def plot_data_sets(train_X, train_Y, val_X, val_Y, test_X, test_Y):
    """
    train_X, train_Y, val_X, val_Y, test_X, test_Y => 1D arrays or PyTorch Tensors
    We'll convert them to NumPy if needed, then produce a single figure.
    """

    # 1) Convert to NumPy if PyTorch Tensors
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    train_x_np = to_np(train_X)
    train_y_np = to_np(train_Y)
    val_x_np   = to_np(val_X)
    val_y_np   = to_np(val_Y)
    test_x_np  = to_np(test_X)
    test_y_np  = to_np(test_Y)

    # 2) Create the figure
    plt.figure(figsize=(7, 4.5))  # slightly wide, good ratio
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    # 3) Plot each set with distinct color/marker
    plt.scatter(train_x_np, train_y_np,
                color='blue',  edgecolor='white', s=50, alpha=0.8, label='Train')
    plt.scatter(val_x_np,   val_y_np,
                color='green', edgecolor='white', s=50, alpha=0.8, label='Val')
    plt.scatter(test_x_np,  test_y_np,
                color='red',   edgecolor='white', s=50, alpha=0.8, label='Test')

    # 4) Cosmetic improvements
    plt.title("Train, Validation, and Test Sets", fontsize=14, pad=10)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)

    # Position legend slightly outside or in upper-left
    # e.g. bounding box => (1.02, 1) if we want outside
    plt.legend(loc="best", fontsize=11, frameon=True)

    plt.grid(True, alpha=0.2)  # subtle grid
    plt.tight_layout()
    plt.show()


############################################################
# B) Plot the Original f(x) + Each Model's Predictions
############################################################

def plot_model_predictions(X_dense, Y_dense_true, model_results, 
                           train_X=None, train_Y=None, test_X=None, test_Y=None):
    """
    X_dense, Y_dense_true => define the 'continuous' function to compare against
    model_results => dict of {model_name: {"test_data":..., "test_predictions":...}, ...}
    train_X, train_Y => optional, to highlight training points
    test_X, test_Y   => optional, to highlight test points
    """
    # 1) Convert to NP if Tensors
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    Xd_np = to_np(X_dense)
    Yd_np = to_np(Y_dense_true)

    # Create figure
    plt.figure(figsize=(7, 4.5))
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    # Plot the original function
    plt.plot(Xd_np, Yd_np, color='gray', lw=2.5, alpha=0.8, label="f(x) original")

    # # (Optional) highlight training points if available
    # if train_X is not None and train_Y is not None:
    #     train_x_np = to_np(train_X)
    #     train_y_np = to_np(train_Y)
    #     plt.scatter(train_x_np, train_y_np, color='blue', marker='o', s=40, alpha=0.5,
    #                 label="Train (optional)")

    # # (Optional) highlight test points if available
    # if test_X is not None and test_Y is not None:
    #     test_x_np = to_np(test_X)
    #     test_y_np = to_np(test_Y)
    #     plt.scatter(test_x_np, test_y_np, color='red', marker='^', s=40, alpha=0.5,
    #                 label="Test (optional)")

    # Distinct colors for up to 5 models
    # (Use many more if you have many models, possibly from a bigger palette)
    color_cycle = ['orange', 'limegreen', 'royalblue', 'purple', 'darkcyan', 
                   'gold', 'magenta', 'brown', 'pink', 'olive']
    
    # 2) For each model, plot test_data vs. test_predictions
    for i, (model_name, results_dict) in enumerate(model_results.items()):
        # Extract arrays
        test_data_i  = np.array(results_dict['test_data'])
        test_preds_i = np.array(results_dict['test_predictions'])

        # Sort them by X => produce a line in order
        idx_sort = np.argsort(test_data_i)
        x_sorted = test_data_i[idx_sort]
        y_sorted = test_preds_i[idx_sort]

        c = color_cycle[i % len(color_cycle)]
        plt.plot(x_sorted, y_sorted, '--', color=c, lw=2,
                 label=f"{model_name}: predictions")

    # Cosmetics
    plt.title("Model Predictions vs. Original Function", fontsize=14, pad=10)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(loc="best", fontsize=11, frameon=True)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))

# Optionally define a color cycle
    color_cycle = ["red", "blue", "green", "magenta", "orange"]

    for i, (model_name, results_dict) in enumerate(model_results.items()):
    # Extract NumPy arrays
        test_data_i  = np.array(results_dict['test_data'])
        test_preds_i = np.array(results_dict['test_predictions'])

    # Sort by X if you want them to appear "left-to-right"
        idx_sort = np.argsort(test_data_i)
        x_sorted = test_data_i[idx_sort]
        y_sorted = test_preds_i[idx_sort]

    # Pick a color
        c = color_cycle[i]
    
    # Scatter plot: use marker style, remove line style
        plt.scatter(x_sorted, y_sorted,
                color=c, 
                s=10,             # marker size
                alpha=0.8,        # transparency
                edgecolor='black',# black edge for good contrast
                label=f"{model_name} predictions")
    plt.plot(Xd_np, Yd_np, color='gray', lw=2.5, alpha=0.8, label="f(x) original")

    plt.title("Comparison of Test Predictions (Scatter)", fontsize=12)
    plt.xlabel("X")
    plt.ylabel("Predicted Y")
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_test_predictions_comparison(
    X_dense, Y_dense_true, 
    model_results, 
    figsize=(7, 4.5), 
    title="Comparison of Test Predictions (Scatter)"
):
    """
    Args:
      X_dense (np.ndarray or torch.Tensor): 1D dense grid for x
      Y_dense_true (np.ndarray or torch.Tensor): corresponding f(x) ground truth
      model_results (dict): { "ModelName": {"test_data": [...], "test_predictions": [...]} }
      figsize (tuple): figure size
      title (str): main title
    
    The function displays a single scatter-plot figure showing each model's 
    test predictions. The ground-truth function is plotted as a line in the 
    background for reference.
    """

    # 1) Convert inputs to NumPy if needed
    if hasattr(X_dense, "cpu"): 
        X_dense = X_dense.cpu().numpy()
    if hasattr(Y_dense_true, "cpu"):
        Y_dense_true = Y_dense_true.cpu().numpy()
    
    # 2) Initialize figure
    plt.figure(figsize=figsize)
    
    # Some styling for top ML journals
    # - Reasonable font sizes
    # - Removing top/right spines
    # - Light gridlines
    plt.rcParams.update({
        "font.size": 10,            # Smaller label font
        "axes.labelsize": 10,       # Axis-label font
        "axes.titlesize": 11,       # Title font
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,         # Subtle grid
        "grid.linestyle": "--"
    })
    
    # 3) Plot the “true” function as a gray line
    plt.plot(X_dense, Y_dense_true, '--',
             color='black', lw=2, 
             label="Ground truth $f(x)$")

    # 4) A colorblind-friendly palette, e.g. from Paul Tol or similar
    color_cycle = [
        "#0072B2",  # a nice "blue"
        "#D55E00",  # "orange"
        "#009E73",  # "green"
        "#CC79A7",  # "magenta"
        "#F0E442",  # "yellow"
        "#56B4E9",  # "light-blue"
        "#E69F00",  # "dark-orange"
        "#000000"   # black
    ]

    # 5) For each model, scatter-plot test_data vs. test_predictions
    for i, (model_name, results_dict) in enumerate(model_results.items()):
        # Retrieve arrays
        test_data_i  = np.array(results_dict['test_data'])
        test_preds_i = np.array(results_dict['test_predictions'])

        # Sort them by X => helps if you want left->right lines or a consistent scatter
        idx_sort = np.argsort(test_data_i)
        x_sorted = test_data_i[idx_sort]
        y_sorted = test_preds_i[idx_sort]

        # Pick a color from the cycle
        c = color_cycle[i % len(color_cycle)]

        # We do a scatter (not a line) to emphasize discrete predictions
        plt.scatter(
            x_sorted, y_sorted,
            color=c, s=20, alpha=0.8, 
            #edgecolor='k', 
            linewidth=0.5,    # for crisp markers
            label=f"{model_name} predictions"
        )

    # 6) Final touches
    plt.title(title, fontsize=11, pad=6)  # Slightly smaller, more “journal style”
    plt.xlabel("X")
    plt.ylabel("Predicted Y")
    plt.legend(loc="best")  # or "upper left", etc.
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
# Usage example (assuming you have X_dense, Y_dense_true, model_results):
# 
# plot_test_predictions_comparison(X_dense, Y_dense_true, model_results)
#
# For best results, ensure that:
# - model_results = {
#     "ModelA": {"test_data": [...], "test_predictions": [...]},
#     "ModelB": {"test_data": [...], "test_predictions": [...]},
#     ...
#   }
# where test_data / test_predictions are Python lists or NumPy arrays of the 
# same length. If you have them as torch.Tensors, convert to np.array first.
# ----------------------------------------------------------------------------
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot_abs_errors_with_mean(
    model_names, 
    abs_errors_all, 
    color_map=None,
    title="Absolute Error Distribution (Test Set)",
    filename=None
):
    """
    Plots a boxplot of absolute errors for multiple models, each with a custom color
    if color_map is provided. The boxplot will show a line at the *mean* (instead of the median).
    We also annotate each box with its mean, and print out stats (mean, std, min, max) 
    sorted by ascending mean error.

    Args:
      model_names     : list of str (model names)
      abs_errors_all  : list of arrays, each containing absolute errors for one model
      color_map       : dict { model_name : color }, optional custom facecolors
      title           : str, overall plot title
      filename        : str, optional path to save figure
    """

    # 1) Compute stats for each model
    mean_vals = [np.mean(errs) for errs in abs_errors_all]
    std_vals  = [np.std(errs)  for errs in abs_errors_all]
    min_vals  = [np.min(errs)  for errs in abs_errors_all]
    max_vals  = [np.max(errs)  for errs in abs_errors_all]

    # 2) Sort by ascending mean error
    sorted_idx = np.argsort(mean_vals)
    model_names_sorted   = [model_names[i]    for i in sorted_idx]
    abs_errors_sorted    = [abs_errors_all[i] for i in sorted_idx]
    mean_vals_sorted     = [mean_vals[i]      for i in sorted_idx]
    std_vals_sorted      = [std_vals[i]       for i in sorted_idx]
    min_vals_sorted      = [min_vals[i]       for i in sorted_idx]
    max_vals_sorted      = [max_vals[i]       for i in sorted_idx]

    # 3) Print stats in ascending mean order
    print("=== Absolute Error Statistics (Test Set) ===")
    for mname, mean_, std_, mn_, mx_ in zip(model_names_sorted, 
                                            mean_vals_sorted,
                                            std_vals_sorted, 
                                            min_vals_sorted, 
                                            max_vals_sorted):
        print(f"{mname:12s} => Mean={mean_:.5f}, Std={std_:.5f}, "
              f"Min={mn_:.5f}, Max={mx_:.5f}")

    # 4) Configure style
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 5) Create the boxplot with *mean* lines instead of medians
    box = ax.boxplot(
        abs_errors_sorted,
        labels=model_names_sorted,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops=dict(color='darkred', linewidth=1.5),
        medianprops=dict(visible=False),  # Hide the median line
        showfliers=True, 
        notch=False,
        widths=0.6
    )

    # 6) Helper function to place text annotations for the mean above each box
    def annotate_means(ax, boxplot_obj, stats_list):
        """
        Place text annotations for each box with the numeric mean value.
        stats_list is assumed to match the sorted box order.
        """
        for i, box_patch in enumerate(boxplot_obj['boxes']):
            x_center = i + 1  # Box i is at x = i+1
            path_data = box_patch.get_path().vertices
            y_top = max(path_data[:, 1])  # top edge of the box
            model_mean = stats_list[i]
            ax.text(
                x_center, 
                y_top + 0.01,        # slightly above the box
                f"{model_mean:.3f}", # e.g. 0.012
                ha='center', va='bottom',
                fontsize=9, color='darkblue'
            )

    # 7) If color_map is provided, color each box
    if color_map is not None:
        for patch, m_name in zip(box["boxes"], model_names_sorted):
            c = color_map.get(m_name, "gray")
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

    # 8) Annotate each box with the mean
    annotate_means(ax, box, mean_vals_sorted)

    ax.set_title(title, pad=5)
    ax.set_ylabel("Absolute Error", labelpad=5)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# ----------------- Example usage below -----------------
# if __name__ == "__main__":
#     # Suppose you have something like:
#     # model_results = {
#     #     "CauchyNet":   {"test_data": [...], "test_predictions":[...]},
#     #     "SIREN":       {"test_data": [...], "test_predictions":[...]},
#     #     "FNN":         {"test_data": [...], "test_predictions":[...]},
#     #     "NBeats":      {"test_data": [...], "test_predictions":[...]},
#     # }
#     # And a function f(x) that returns ground truth

#     # 1) Gather absolute errors
#     model_names    = []
#     abs_errors_all = []
#     for model_name, results in model_results.items():
#         x_np   = np.array(results["test_data"])
#         preds  = np.array(results["test_predictions"])
#         x_torch= torch.from_numpy(x_np).float()
#         y_torch= f(x_torch)  # ground truth
#         y_np   = y_torch.numpy()  # CPU assumed
#         abs_err= np.abs(preds - y_np)
#         model_names.append(model_name)
#         abs_errors_all.append(abs_err)

#     # 2) Define color map
#     color_map = {
#         "CauchyNet": "#F2C649",
#         "SIREN":     "#5DA5DA",
#         "FNN":       "#60BD68",
#         "NBeats":    "#FAA43A"
#     }

#     # 3) Finally, call the plot function
#     plot_boxplot_abs_errors_with_mean(
#         model_names=model_names,
#         abs_errors_all=abs_errors_all,
#         color_map=color_map,
#         title="Absolute Error Distribution (Test Set) - Show Mean Lines",
#         filename="abs_err_boxplot_with_mean.png"
#     )

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot_with_stats_table(
    model_names,
    abs_errors_all,
    color_map=None,
    title="Absolute Error Distribution (Test Set)",
    filename=None
):
    """
    Creates a figure with two subplots:
      - Left: a boxplot of absolute errors (sorted by mean).
      - Right: a small stats table (mean, std, min, max) 
        that aligns row-by-row with the boxplot’s model order.
    """

    # 1) Compute stats for each model
    mean_vals = [np.mean(errs) for errs in abs_errors_all]
    std_vals  = [np.std(errs)  for errs in abs_errors_all]
    min_vals  = [np.min(errs)  for errs in abs_errors_all]
    max_vals  = [np.max(errs)  for errs in abs_errors_all]

    # 2) Sort by ascending mean
    sorted_idx = np.argsort(mean_vals)
    models_sorted  = [model_names[i]    for i in sorted_idx]
    errors_sorted  = [abs_errors_all[i] for i in sorted_idx]
    means_sorted   = [mean_vals[i]      for i in sorted_idx]
    stds_sorted    = [std_vals[i]       for i in sorted_idx]
    mins_sorted    = [min_vals[i]       for i in sorted_idx]
    maxs_sorted    = [max_vals[i]       for i in sorted_idx]

    # 3) Configure style
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    # 4) Create figure with 2 subplots: left (boxplot), right (table)
    fig = plt.figure(figsize=(8, 4))
    gs  = fig.add_gridspec(ncols=2, width_ratios=[0.65, 0.35])  
    ax_box   = fig.add_subplot(gs[0])  # left
    ax_table = fig.add_subplot(gs[1])  # right

    # 5) Build the boxplot in ax_box
    box = ax_box.boxplot(
        errors_sorted,
        labels=models_sorted,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops=dict(color='darkred', linewidth=1.3),
        medianprops=dict(visible=False),  # optionally hide median line
        showfliers=True, 
        notch=False,
        widths=0.4
    )

    # 6) Apply custom facecolors if provided
    if color_map:
        for box_patch, m_name in zip(box["boxes"], models_sorted):
            c = color_map.get(m_name, "gray")
            box_patch.set_facecolor(c)
            box_patch.set_alpha(0.6)

    ax_box.set_title(title, pad=5)
    ax_box.set_ylabel("Absolute Error", labelpad=5)

    # 7) Construct table data 
    #    We'll show columns: Mean, Std, Min, Max (all with some formatting).
    #    Each row aligns with the sorted model order.
    table_rows = []
    for (m, mean_, std_, mn_, mx_) in zip(
            models_sorted, means_sorted, stds_sorted, mins_sorted, maxs_sorted):
        table_rows.append([
            f"{m}", 
            f"{mean_:.4f}",
            f"{std_:.4f}",
            f"{mn_:.4f}",
            f"{mx_:.4f}"
        ])

    # column labels
    col_labels = ["Model", "Mean", "Std", "Min", "Max"]

    # 8) Turn off the axis for ax_table (we only want the table)
    ax_table.axis("off")
    ax_table.set_title("Error Stats", fontsize=10, pad=4)

    # 9) Create the table
    #    cellLoc='center' to center text, loc='upper left' is table’s anchor inside the axis
    the_table = ax_table.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc='center right',
        cellLoc='center',
        colLoc='center',
        edges='horizontal'
    )
    # Some styling (optional):
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    the_table.scale(1.4, 2)  # maybe stretch row height

    # 10) Adjust layout
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
