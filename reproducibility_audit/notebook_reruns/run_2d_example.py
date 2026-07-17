import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
_shown = [0]
def _show(*a, **k):
    plt.gcf().savefig(f'fig_2d_example_{_shown[0]:02d}.png', dpi=110, bbox_inches='tight'); _shown[0]+=1; plt.close('all')
plt.show = _show



# 1. Import Necessary Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
# import seaborn (unused)
import pandas as pd
import random

# Import custom utilities from utils2.py
from utils2 import loadData, train_and_evaluate_model, generate_latex_table, scatter_predictions_vs_truth_improved

# For reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# 2. Define the CauchyNet Model
class ReciprocalActivation(nn.Module):
    def __init__(self):
        super(ReciprocalActivation, self).__init__()

    def forward(self, z_diff):
        """
        Compute the reciprocal of the product of (xi_{i,j} - z_i) across input dimensions.

        Args:
            z_diff (torch.Tensor): Tensor of shape [batch_size, N, m] representing (xi_{i,j} - z_i)

        Returns:
            torch.Tensor: Reciprocal tensor of shape [batch_size, N]
        """
        # Compute the product across the 'm' dimension
        product = torch.prod(z_diff, dim=2)  # Shape: [batch_size, N]
        
        # To prevent division by zero, add a small epsilon
        epsilon = 1e-8
        reciprocals = 1.0 / (product + epsilon)  # Shape: [batch_size, N]
        
        return reciprocals

class CauchyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the CauchyNet.

        Args:
            input_size (int): Dimensionality of the input vector (m).
            hidden_size (int): Number of hidden units (N).
            output_size (int): Dimensionality of the output.
        """
        super(CauchyNet, self).__init__()
        self.input_size = input_size  # m
        self.hidden_size = hidden_size  # N
        self.output_size = output_size  # typically 1 for regression

        # Initialize xi parameters: [m, N]
        # Each xi_{i,j} is a complex number
        # Initialize with a normal distribution for both real and imaginary parts
        xi_real = torch.randn(input_size, hidden_size)
        xi_imag = torch.randn(input_size, hidden_size)
        self.xi = nn.Parameter(torch.complex(xi_real, xi_imag))  # Shape: [m, N]

        # Initialize theta parameters: [N, output_size]
        theta_real = torch.randn(hidden_size, output_size) * 0.01
        theta_imag = torch.randn(hidden_size, output_size) * 0.01
        self.theta = nn.Parameter(torch.complex(theta_real, theta_imag))  # Shape: [N, output_size]

        # Activation Function
        self.activation = ReciprocalActivation()

    def forward(self, x):
        """
        Forward pass of the CauchyNet.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, m]

        Returns:
            tuple: (y_real, y_imag) each of shape [batch_size, output_size]
        """
        batch_size = x.size(0)

        # Step 1: Extend real inputs to complex
        z = torch.complex(x, torch.zeros_like(x))  # Shape: [batch_size, m]

        # Step 2: Compute (xi - z) for each input dimension and hidden unit
        # xi: [m, N], z: [batch_size, m] -> need to compute [batch_size, N, m]
        # Expand z to [batch_size, 1, m] and xi to [1, N, m]
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, m]
        xi_expanded = self.xi.transpose(0, 1).unsqueeze(0)  # [1, N, m]

        # Compute (xi - z): [batch_size, N, m]
        z_diff = xi_expanded - z_expanded  # Broadcasting

        # Step 3: Apply reciprocal activation
        # activated: [batch_size, N]
        activated = self.activation(z_diff)  # [batch_size, N]

        # Step 4: Multiply by theta and sum over N to get output
        # theta: [N, output_size], activated: [batch_size, N]
        # Perform element-wise multiplication and sum over N
        # First, reshape activated to [batch_size, N, 1]
        activated = activated.unsqueeze(2)  # [batch_size, N, 1]

        # Broadcast theta to [1, N, output_size]
        theta = self.theta.unsqueeze(0)  # [1, N, output_size]

        # Multiply: [batch_size, N, output_size]
        multiplied = activated * theta  # [batch_size, N, output_size]

        # Sum over N: [batch_size, output_size]
        y = multiplied.sum(dim=1)  # [batch_size, output_size]

        # Separate real and imaginary parts
        y_real = y.real  # [batch_size, output_size]
        y_imag = y.imag  # [batch_size, output_size]

        return y_real, y_imag

# 3. Generate Synthetic Data
def f(X):
    """
    Compute the target function.

    Args:
        X (torch.Tensor): Input tensor of shape [num_samples, 2]

    Returns:
        torch.Tensor: Output tensor of shape [num_samples]
    """
    y = (X[:, 0] - 1)**2 - (X[:, 0] - 1) * (X[:, 1] - 2) + 3 * (X[:, 1] - 2) + (X[:, 1] - 2)**2 + 1 / (5 + (X[:, 0] - 1)**2)
    return -y

# Generate synthetic data
num_samples = 1000
input_size = 2  # m = 2
hidden_size = 128  # N = 10 (you can adjust this)
output_size = 1  # Single output

# Generate input features uniformly distributed between [0, 4]
X_np = np.random.uniform(low=0, high=4, size=(num_samples, input_size)).astype(np.float32)
X_tensor = torch.from_numpy(X_np)

# Compute targets
Y_np = f(X_tensor).numpy().astype(np.float32)
Y_tensor = torch.from_numpy(Y_np)

# 4. Load Data Using loadData
# Load data
train_loader, val_loader, test_loader, ds_test = loadData(X_tensor, Y_tensor, batchSize=64)

# 5. Train the CauchyNet Model
# Define a model constructor
def CauchyNet_constructor(input_size, hidden_size, output_size):
    return CauchyNet(input_size, hidden_size, output_size)

# Train the model
model, train_losses, val_losses, test_mse, test_mae, preds_unscaled, truths_unscaled, training_time, num_params = train_and_evaluate_model(
    model_constructor=CauchyNet_constructor,
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    lr=0.01,
    epochs=200,
    device=None,  # Auto-detect
    scaler=None   # No scaling
)

print(f"Training completed in {training_time:.2f} seconds.")
print(f"Number of Parameters: {num_params}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")


# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()


# Create a dictionary for predictions
preds_dict = {
    "CauchyNet": preds_unscaled
}

# Define color map
color_map = {
    "CauchyNet": (0.12156862745098039, 0.4666666666666667,  0.7058823529411765),  # blue
}

# Plot scatter predictions vs. true
scatter_predictions_vs_truth_improved(
    y_true=truths_unscaled,
    preds_dict=preds_dict,
    color_map=color_map,
    marker_size=30,
    alpha_val=0.9,
    add_stats=True,
    filename=None  # Set to a path string to save the figure
)






import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import vispy

import numpy as np


import numpy as np


def f(X):
    y= (X[:, 0] - 1)**2 - (X[:, 0] - 1) * (X[:, 1] - 2) + 3 * (X[:, 1] - 2) + (X[:, 1] - 2)**2 + 1 / (5 + (X[:, 0] - 1)**2)
    return -y






import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Define critical points near the bumps and other parameters
critical_points = [(0., 0.)]
total_points = 3000
critical_radius = 0.3  # Radius around the bumps
a, b = 0.8,0.8  # Semi-major and semi-minor axes for the ellipse


# Initialize points within a square bounding box
all_points = torch.rand((total_points, 2)) * 2 * max(a, b) - max(a, b)
# ellipse_mask = ((all_points[:, 0])**2) / a**2 + ((all_points[:, 1])**2) / b**2 <= 1
# ellipse_points = all_points[ellipse_mask]
ellipse_mask = (torch.abs(all_points[:, 0]) <= a) & (torch.abs(all_points[:, 1]) <= a)

# Apply the mask to get points inside the square
ellipse_points = all_points[ellipse_mask]

# Calculate function values
all_values = f(ellipse_points)

# Identify points near the critical points
near_critical_mask = torch.zeros(len(ellipse_points), dtype=torch.bool)
for cp in critical_points:
    cp_tensor = torch.tensor(cp, dtype=torch.float32)
    distances = torch.norm(ellipse_points - cp_tensor, dim=1)
    near_critical_mask |= distances < critical_radius

# Split points into dataset and test set
dataset_points = ellipse_points[~near_critical_mask]
dataset_values = all_values[~near_critical_mask]
test_points = ellipse_points[near_critical_mask]
test_values = all_values[near_critical_mask]

# Split the dataset into training and validation sets
train_val_ratio = 0.6
num_train_samples = int(train_val_ratio * len(dataset_points))
train_dataset = TensorDataset(dataset_points[:num_train_samples], dataset_values[:num_train_samples])
val_dataset = TensorDataset(dataset_points[num_train_samples:], dataset_values[num_train_samples:])
test_dataset = TensorDataset(test_points, test_values)

# Print the shapes of the datasets
print("Training data shape:", train_dataset.tensors[0].shape, train_dataset.tensors[1].shape)
print("Validation data shape:", val_dataset.tensors[0].shape, val_dataset.tensors[1].shape)
print("Test data shape:", test_dataset.tensors[0].shape, test_dataset.tensors[1].shape)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define your function f(X) here

def plot_test_dataset(ax, dataset, marker, color):
    X, Y = dataset.tensors
    X = X.numpy()
    Y = Y.numpy()
    Z = f(X) + 0.001
    ax.scatter(X[:, 0], X[:, 1], Z, c=color, s=20, marker=marker, edgecolors='blue', depthshade=False)

def plot_train_dataset(ax, dataset, marker, color):
    X, Y = dataset.tensors
    X = X.numpy()
    Y = Y.numpy()
    Z = f(X) + 0.0001
    ax.scatter(X[:, 0], X[:, 1], Z, c=color, s=10, marker=marker, alpha=0.5, depthshade=False)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-0.8, 0.8, 500)
y = np.linspace(-0.8, 0.8, 500)
X1, Y1 = np.meshgrid(x, y)
Z = f(np.vstack((X1.ravel(), Y1.ravel())).T)
Z = Z.reshape(X1.shape)

ax.plot_surface(X1, Y1, Z, cmap='viridis', alpha=0.8)

# Plot the training dataset without a label
plot_train_dataset(ax, train_dataset, 'o', 'blue')

# Remove axis labels, titles, and legend
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.grid(False)
ax.xaxis._axinfo['juggled'] = (0,0,0)
ax.yaxis._axinfo['juggled'] = (0,0,0)
ax.zaxis._axinfo['juggled'] = (0,0,0)

# Adjust the plot to minimize empty spaces
plt.tight_layout()

# Set view angle
ax.view_init(elev=30, azim=60)

plt.savefig('2f1_training1.png', dpi=300)
plt.show()




import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Define your function f(X) here
def f(X):
    """
    Compute the target function.

    Args:
        X (np.ndarray or torch.Tensor): Input array of shape [N, 2]

    Returns:
        np.ndarray or torch.Tensor: Output array of shape [N]
    """
    X = torch.tensor(X, dtype=torch.float32) if isinstance(X, np.ndarray) else X
    y = (X[:, 0] - 1)**2 - (X[:, 0] - 1) * (X[:, 1] - 2) + \
        3 * (X[:, 1] - 2) + (X[:, 1] - 2)**2 + \
        1 / (5 + (X[:, 0] - 1)**2)
    return -y

# ------------------------
# Define the CauchyNet Model
# ------------------------

class ReciprocalActivation(nn.Module):
    def __init__(self):
        super(ReciprocalActivation, self).__init__()

    def forward(self, z_diff):
        """
        Compute the reciprocal of the product of (xi_{i,j} - z_i) across input dimensions.

        Args:
            z_diff (torch.Tensor): Tensor of shape [batch_size, N, m] representing (xi_{i,j} - z_i)

        Returns:
            torch.Tensor: Reciprocal tensor of shape [batch_size, N]
        """
        # Compute the product across the 'm' dimension
        product = torch.prod(z_diff, dim=2)  # Shape: [batch_size, N]
        
        # To prevent division by zero, add a small epsilon
        epsilon = 1e-8
        reciprocals = 1.0 / (product + epsilon)  # Shape: [batch_size, N]
        
        return reciprocals

class CauchyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the CauchyNet.

        Args:
            input_size (int): Dimensionality of the input vector (m).
            hidden_size (int): Number of hidden units (N).
            output_size (int): Dimensionality of the output.
        """
        super(CauchyNet, self).__init__()
        self.input_size = input_size  # m
        self.hidden_size = hidden_size  # N
        self.output_size = output_size  # typically 1 for regression

        # Initialize xi parameters: [m, N]
        # Each xi_{i,j} is a complex number
        # Initialize with a normal distribution for both real and imaginary parts
        xi_real = torch.randn(input_size, hidden_size)
        xi_imag = torch.randn(input_size, hidden_size)
        self.xi = nn.Parameter(torch.complex(xi_real, xi_imag))  # Shape: [m, N]

        # Initialize theta parameters: [N, output_size]
        theta_real = torch.randn(hidden_size, output_size) * 0.01
        theta_imag = torch.randn(hidden_size, output_size) * 0.01
        self.theta = nn.Parameter(torch.complex(theta_real, theta_imag))  # Shape: [N, output_size]

        # Activation Function
        self.activation = ReciprocalActivation()

    def forward(self, x):
        """
        Forward pass of the CauchyNet.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, m]

        Returns:
            tuple: (y_real, y_imag) each of shape [batch_size, output_size]
        """
        batch_size = x.size(0)

        # Step 1: Extend real inputs to complex
        z = torch.complex(x, torch.zeros_like(x))  # Shape: [batch_size, m]

        # Step 2: Compute (xi - z) for each input dimension and hidden unit
        # xi: [m, N], z: [batch_size, m] -> need to compute [batch_size, N, m]
        # Expand z to [batch_size, 1, m] and xi to [1, N, m]
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, m]
        xi_expanded = self.xi.transpose(0, 1).unsqueeze(0)  # [1, N, m]

        # Compute (xi - z): [batch_size, N, m]
        z_diff = xi_expanded - z_expanded  # Broadcasting

        # Step 3: Apply reciprocal activation
        # activated: [batch_size, N]
        activated = self.activation(z_diff)  # [batch_size, N]

        # Step 4: Multiply by theta and sum over N to get output
        # theta: [N, output_size], activated: [batch_size, N]
        # Perform element-wise multiplication and sum over N
        # First, reshape activated to [batch_size, N, 1]
        activated = activated.unsqueeze(2)  # [batch_size, N, 1]

        # Broadcast theta to [1, N, output_size]
        theta = self.theta.unsqueeze(0)  # [1, N, output_size]

        # Multiply: [batch_size, N, output_size]
        multiplied = activated * theta  # [batch_size, N, output_size]

        # Sum over N: [batch_size, output_size]
        y = multiplied.sum(dim=1)  # [batch_size, output_size]

        # Separate real and imaginary parts
        y_real = y.real  # [batch_size, output_size]
        y_imag = y.imag  # [batch_size, output_size]

        return y_real, y_imag

# ------------------------
# Prediction Function
# ------------------------

def predict(model, grid_points, device='cpu'):
    """
    Predicts outputs for a grid of input points using the trained model.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        grid_points (np.ndarray): Input points of shape [num_points, 2].
        device (str): Device to perform computation ('cpu' or 'cuda').

    Returns:
        np.ndarray: Predicted values of shape [num_points].
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
        pred_real, pred_imag = model(X_tensor)
        preds = pred_real.cpu().numpy().flatten()
    return preds

# ------------------------
# Main Workflow
# ------------------------

# Define critical points near the bumps and other parameters
critical_points = [(0., 0.)]
total_points = 3000
critical_radius = 0.3  # Radius around the bumps
a, b = 0.2, 0.2  # Semi-major and semi-minor axes for the ellipse

# Initialize points within a square bounding box
all_points = torch.rand((total_points, 2)) * 2 * max(a, b) - max(a, b)

# Define an ellipse mask (currently using square, uncomment for ellipse)
# ellipse_mask = ((all_points[:, 0])**2) / a**2 + ((all_points[:, 1])**2) / b**2 <= 1
ellipse_mask = (torch.abs(all_points[:, 0]) <= a) & (torch.abs(all_points[:, 1]) <= b)

# Apply the mask to get points inside the ellipse
ellipse_points = all_points[ellipse_mask]

# Calculate function values
all_values = f(ellipse_points)

# Identify points near the critical points
near_critical_mask = torch.zeros(len(ellipse_points), dtype=torch.bool)
for cp in critical_points:
    cp_tensor = torch.tensor(cp, dtype=torch.float32)
    distances = torch.norm(ellipse_points - cp_tensor, dim=1)
    near_critical_mask |= distances < critical_radius

# Split points into dataset and test set
dataset_points = ellipse_points[~near_critical_mask]
dataset_values = all_values[~near_critical_mask]
test_points = ellipse_points[near_critical_mask]
test_values = all_values[near_critical_mask]

# Split the dataset into training and validation sets
train_val_ratio = 0.6
num_train_samples = int(train_val_ratio * len(dataset_points))
train_dataset = TensorDataset(dataset_points[:num_train_samples], dataset_values[:num_train_samples])
val_dataset = TensorDataset(dataset_points[num_train_samples:], dataset_values[num_train_samples:])
test_dataset = TensorDataset(test_points, test_values)

# Print the shapes of the datasets
print("Training data shape:", train_dataset.tensors[0].shape, train_dataset.tensors[1].shape)
print("Validation data shape:", val_dataset.tensors[0].shape, val_dataset.tensors[1].shape)
print("Test data shape:", test_dataset.tensors[0].shape, test_dataset.tensors[1].shape)

# =============== 5) Extract and scale data ===============
# Extract datasets
train_X = train_dataset.tensors[0]
train_Y = train_dataset.tensors[1]

val_X = val_dataset.tensors[0]
val_Y = val_dataset.tensors[1]

test_X = test_dataset.tensors[0]
test_Y = test_dataset.tensors[1]

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on training and validation data for consistency
Y_train_val_np = torch.cat((train_Y, val_Y)).numpy().reshape(-1, 1)
scaler.fit(Y_train_val_np)

# Transform Y data
Y_train_scaled = scaler.transform(train_Y.numpy().reshape(-1, 1)).flatten()
Y_val_scaled = scaler.transform(val_Y.numpy().reshape(-1, 1)).flatten()
Y_test_scaled = scaler.transform(test_Y.numpy().reshape(-1, 1)).flatten()

# Convert scaled Y back to tensors
Y_train_scaled = torch.tensor(Y_train_scaled, dtype=torch.float32)
Y_val_scaled = torch.tensor(Y_val_scaled, dtype=torch.float32)
Y_test_scaled = torch.tensor(Y_test_scaled, dtype=torch.float32)

# =============== 6) Create Dataset and DataLoader instances ===============
# Create Dataset instances
train_dataset_scaled = TensorDataset(train_X, Y_train_scaled)
val_dataset_scaled = TensorDataset(val_X, Y_val_scaled)
test_dataset_scaled = TensorDataset(test_X, Y_test_scaled)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset_scaled, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset_scaled, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset_scaled, batch_size=batch_size, shuffle=False)

# =============== 7) Initialize the CauchyNet model ===============
hidden_size = 128
output_size = 1
input_size = train_X.shape[1]  # Should be 2

# Initialize the model
model = CauchyNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =============== 8) Define the loss function and optimizer ===============
# Define the loss function (Mean Squared Error for regression)
criterion = nn.MSELoss()

# Define the optimizer (Adam)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# =============== 9) Training loop ===============
num_epochs = 100

# Lists to store loss values
train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device).float()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        pred_real, pred_imag = model(batch_X)

        # Compute loss on the real part only
        loss = criterion(pred_real, batch_Y.unsqueeze(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for val_X_batch, val_Y_batch in val_loader:
            val_X_batch = val_X_batch.to(device)
            val_Y_batch = val_Y_batch.to(device).float()

            pred_real, pred_imag = model(val_X_batch)
            val_loss = criterion(pred_real, val_Y_batch.unsqueeze(-1))

            val_running_loss += val_loss.item() * val_X_batch.size(0)

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}')

# =============== 10) Plot the loss curves ===============
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss_cauchynet.png', dpi=300)
plt.show()

# =============== 11) Evaluate on the test set ===============
model.eval()
test_loss = 0.0
all_preds = []
all_truths = []
with torch.no_grad():
    for test_X_batch, test_Y_batch in test_loader:
        test_X_batch = test_X_batch.to(device)
        test_Y_batch = test_Y_batch.to(device).float()

        pred_real, pred_imag = model(test_X_batch)
        loss = criterion(pred_real, test_Y_batch.unsqueeze(-1))
        test_loss += loss.item() * test_X_batch.size(0)

        # Collect predictions and truths for further analysis
        all_preds.extend(pred_real.cpu().numpy().flatten())
        all_truths.extend(test_Y_batch.cpu().numpy().flatten())

test_loss /= len(test_loader.dataset)
print(f'Test Loss (MSE): {test_loss:.6f}')

# Inverse transform the predictions and truths
all_preds_unscaled = scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
all_truths_unscaled = scaler.inverse_transform(np.array(all_truths).reshape(-1, 1)).flatten()

# Compute additional metrics
test_mse = mean_squared_error(all_truths_unscaled, all_preds_unscaled)
test_mae = mean_absolute_error(all_truths_unscaled, all_preds_unscaled)
print(f'Test MSE: {test_mse:.6f}')
print(f'Test MAE: {test_mae:.6f}')

# =============== 12) Plot Predictions vs. True Values and Error Surface ===============
def plot_surfaces(model, scaler, device, a=0.8, b=0.8, grid_resolution=100):
    """
    Plots the original data surface, the learned surface from the model, and the error surface.

    Args:
        model (torch.nn.Module): Trained model.
        scaler (MinMaxScaler): Fitted scaler for inverse transforming predictions.
        device (torch.device): Device where the model is located.
        a (float): Semi-major axis for the grid.
        b (float): Semi-minor axis for the grid.
        grid_resolution (int): Number of points along each axis for the grid.
    """
    model.eval()
    
    # Create a grid of points
    x = np.linspace(-a, a, grid_resolution)
    y = np.linspace(-b, b, grid_resolution)
    X1, Y1 = np.meshgrid(x, y)
    grid_points = np.vstack([X1.ravel(), Y1.ravel()]).T

    # Compute true function values
    Z_true = f(grid_points).numpy().reshape(X1.shape)

    # Predict using the model
    Z_pred_scaled = predict(model, grid_points, device=device)
    Z_pred_unscaled = scaler.inverse_transform(Z_pred_scaled.reshape(-1, 1)).flatten()
    Z_pred = Z_pred_unscaled.reshape(X1.shape)

    # Compute error
    error = Z_true - Z_pred

    # Plotting
    fig = plt.figure(figsize=(18, 6))

    # Original Surface
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X1, Y1, Z_true, cmap='viridis', alpha=0.7)
    ax1.set_title('Original Data Surface', fontsize=14)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Y')

    # Learned Surface
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X1, Y1, Z_pred, cmap='Spectral', alpha=0.7)
    ax2.set_title('Learned Surface by CauchyNet', fontsize=14)
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Y')

    # Error Surface
    ax3 = fig.add_subplot(133, projection='3d')
    # Normalize error for color mapping
    norm = plt.Normalize(np.min(error), np.max(error))
    colors = plt.cm.seismic(norm(error))
    ax3.plot_surface(X1, Y1, error, facecolors=colors, alpha=0.7)
    m = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    m.set_array(error)
    fig.colorbar(m, ax=ax3, shrink=0.5, aspect=10, label='Error (True - Predicted)')
    ax3.set_title('Error Surface', fontsize=14)
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.set_zlabel('Error')

    plt.tight_layout()
    plt.savefig('original_learned_error_surfaces.png', dpi=300)
    plt.show()

# Call the plotting function
plot_surfaces(model, scaler, device, a=0.8, b=0.8, grid_resolution=100)















