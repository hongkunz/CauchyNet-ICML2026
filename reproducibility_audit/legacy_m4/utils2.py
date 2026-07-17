############################################
# utils.py
############################################

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

##############################################################################
# 1) Data Loading
##############################################################################
def loadData(X, Y, batchSize=256):
    """
    Splits dataset into 50%/25%/25% for train/val/test.
    Returns train_loader, val_loader, test_loader, ds_test.
    """
    if X.dim() == 1:
        X = X.unsqueeze(-1)
    dataset    = TensorDataset(X, Y)
    total_size = len(dataset)
    testSize   = int(total_size * 0.25)
    valSize    = testSize
    trainSize  = total_size - testSize - valSize

    ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
    train_loader = DataLoader(ds_train, batch_size=batchSize, shuffle=True,num_workers=0,     pin_memory=False)
    val_loader   = DataLoader(ds_val,   batch_size=batchSize, shuffle=False,num_workers=0,    pin_memory=False)
    test_loader  = DataLoader(ds_test,  batch_size=batchSize, shuffle=False,num_workers=0,    pin_memory=False)
    return train_loader, val_loader, test_loader, ds_test

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_and_evaluate_model(
    model_constructor, 
    input_size, hidden_size, output_size,
    train_loader, val_loader, test_loader,
    lr=0.01, epochs=200, device=None,
    scaler=None,            # <-- ADD THIS
    a =1
):
    """
    Simple training loop with Adam + MSE. Returns model + logs + test metrics,
    where test errors are computed on the ORIGINAL (unscaled) data.
    
    Params:
    -------
    scaler : MinMaxScaler or similar
        A fitted scaler for Y. Must implement inverse_transform().
        If None, we'll assume no scaling is used.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_constructor(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=5
    )
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state    = None
    Num_Params    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    start_time = time.time()

    # ============ Training Loop ============
    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0.0

        for x_batch, y_batch_scaled in train_loader:
            x_batch, y_batch_scaled = x_batch.to(device), y_batch_scaled.to(device)
            optimizer.zero_grad()




            
            out = model(x_batch)


            # If model outputs (real, imag):
            if isinstance(out, tuple):
                y_real, y_imag = out
                # Expand y_batch_scaled shape for MSE
                loss = (criterion(y_real, y_batch_scaled.unsqueeze(-1)) 
                      + a*criterion(y_imag, torch.zeros_like(y_imag)))
            else:
                loss = criterion(out, y_batch_scaled.unsqueeze(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_accum += loss.item()

        train_loss = train_loss_accum / len(train_loader)
        train_losses.append(train_loss)

        # ============ Validation Loop ============
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for x_batch, y_batch_scaled in val_loader:
                x_batch, y_batch_scaled = x_batch.to(device), y_batch_scaled.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    val_loss = (criterion(y_real, y_batch_scaled.unsqueeze(-1))
                              + a*criterion(y_imag, torch.zeros_like(y_imag)))
                else:
                    val_loss = criterion(out, y_batch_scaled.unsqueeze(-1))

                val_loss_accum += val_loss.item()

        val_loss = val_loss_accum / len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = model.state_dict()

        scheduler.step(val_loss)

    end_time = time.time()
    training_time = end_time - start_time

    # ============ Load Best Model ============
    if best_state is not None:
        model.load_state_dict(best_state)

    # ============ Final Evaluation on Test Set ============
    model.eval()
    preds_list_unscaled = []
    truths_list_unscaled = []

    with torch.no_grad():
        for x_batch, y_batch_scaled in test_loader:
            x_batch = x_batch.to(device)
            y_batch_scaled = y_batch_scaled.to(device)

            # Forward pass => scaled predictions
            out = model(x_batch)
            if isinstance(out, tuple):
                y_real, _ = out
                preds_scaled = y_real.cpu().numpy().flatten()
            else:
                preds_scaled = out.cpu().numpy().flatten()

            truths_scaled = y_batch_scaled.cpu().numpy().flatten()

            # If we have a scaler, inverse-transform both preds & truths
            if scaler is not None:
                preds_unscaled  = scaler.inverse_transform(
                                    preds_scaled.reshape(-1, 1)).flatten()
                truths_unscaled = scaler.inverse_transform(
                                    truths_scaled.reshape(-1, 1)).flatten()
            else:
                # If no scaler, assume data is already unscaled
                preds_unscaled  = preds_scaled
                truths_unscaled = truths_scaled

            preds_list_unscaled.append(preds_unscaled)
            truths_list_unscaled.append(truths_unscaled)

    preds_all_unscaled  = np.concatenate(preds_list_unscaled)
    truths_all_unscaled = np.concatenate(truths_list_unscaled)

    # Compute test metrics in the ORIGINAL domain
    test_mse = mean_squared_error(truths_all_unscaled, preds_all_unscaled)
    test_mae = mean_absolute_error(truths_all_unscaled, preds_all_unscaled)

    return (model,
            train_losses,
            val_losses,
            test_mse,
            test_mae,
            preds_all_unscaled,   # return unscaled predictions
            truths_all_unscaled,  # return unscaled truths
            training_time,
            Num_Params)

##############################################################################

def get_model_colors(model_names, emphasize_model="CauchyNet"):
    """
    Create a color map for each model using a custom 'Tableau 10' palette
    that is generally pleasing and balanced.
    Then use a gold highlight for `emphasize_model`.
    """
    # Tableau 10 palette in RGB tuples (normalized to 0..1)
    # These match matplotlib's default color cycle for better consistency.
    tableau10 = [
        (0.12156862745098039, 0.4666666666666667,  0.7058823529411765),  # blue
        (0.17254901960784313,  0.6274509803921569, 0.17254901960784313), # green
        (0.8392156862745098,   0.15294117647058825,0.1568627450980392),  # red
        (0.5803921568627451,   0.403921568627451,  0.7411764705882353),  # purple
        #(0.8901960784313725,   0.4666666666666667, 0.7607843137254902),  # pink
        (0.4980392156862745,   0.4980392156862745, 0.4980392156862745),  # gray
         (0.7372549019607844,   0.7411764705882353, 0.13333333333333333), # olive
        # (0.09019607843137255,  0.7450980392156863, 0.8117647058823529),  # cyan
    ]

    # A gold highlight color (goldenrod) for emphasize_model
    highlight_color = (1.0,                  0.4980392156862745, 0.054901960784313725)# (0.85, 0.65, 0.13)

    color_map = {}
    i_pal = 0
    for mname in model_names:
        if mname == emphasize_model:
            color_map[mname] = highlight_color
        else:
            color_map[mname] = tableau10[i_pal % len(tableau10)]
            i_pal += 1

    return color_map

##############################################################################
# 4) Plot Training Curves
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # for manual legend handles


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename=None):
    """
    Plots training (solid) and validation (dashed) curves with median ± IQR 
    for multiple models, but uses only one legend entry per model.

    Parameters
    ----------
    interval_logs : dict
        Dictionary containing 'train' and 'val' logs for each model.
        Structure example:
        {
            'ModelA': {
                'train': [list_of_training_losses_per_run],
                'val':   [list_of_validation_losses_per_run],
            },
            'ModelB': {
                'train': [...],
                'val':   [...],
            }
        }
    model_names : list
        List of model names (keys of interval_logs) to plot.
    color_map : dict
        Mapping from model_name to a color string or RGB tuple 
        (e.g., {'ModelA': 'red', 'ModelB': 'blue'}).
    filename : str, optional
        If provided, the plot is saved to this filename (with dpi=300).
    """

    # Use a Seaborn style that looks neat for publication
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.1)

    plt.figure(figsize=(8, 6))
    epochs_range = None

    for model_name in model_names:
        train_runs = interval_logs[model_name]['train']
        val_runs = interval_logs[model_name]['val']

        # Align runs by minimum length across all runs
        min_train_len = min(len(run) for run in train_runs)
        min_val_len = min(len(run) for run in val_runs)
        min_len = min(min_train_len, min_val_len)

        train_array = np.array([run[:min_len] for run in train_runs])
        val_array   = np.array([run[:min_len] for run in val_runs])

        # Calculate median and IQR
        median_train = np.median(train_array, axis=0)
        q1_train     = np.percentile(train_array, 25, axis=0)
        q3_train     = np.percentile(train_array, 75, axis=0)

        median_val   = np.median(val_array, axis=0)
        q1_val       = np.percentile(val_array, 25, axis=0)
        q3_val       = np.percentile(val_array, 75, axis=0)

        if epochs_range is None:
            epochs_range = np.arange(1, min_len + 1)

        # Color for this model
        c = color_map.get(model_name, 'blue')

        # Plot training curve (solid) with legend
        # (this line gets the label)
        plt.plot(epochs_range, median_train,
                 color=c, linewidth=2, alpha=1.0,
                 label=f"{model_name}")

        # Fill between for training
        plt.fill_between(epochs_range, q1_train, q3_train,
                         color=c, alpha=0.2)

        # Plot validation curve (dashed) with NO legend entry
        plt.plot(epochs_range, median_val,
                 color=c, linestyle='--', linewidth=2, alpha=0.4)
        plt.fill_between(epochs_range, q1_val, q3_val,
                         color=c, alpha=0.15)

    # Log scale on y-axis
    plt.yscale("log")

    # Labels and Title
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("MSE Loss (log scale)", fontsize=13)
    plt.title(r"Training \& Validation (Median $\pm$ Std)", 
              fontsize=15, fontweight='bold')

    # Legend: single entry per model
    plt.legend(loc='best', fontsize=11, frameon=True)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Optionally save plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
##############################################################################
# 5) Box Plot of Errors
##############################################################################
def plot_box_errors(error_dict, color_map, filename=None):
    """
    Show absolute errors distribution for each model as a fancy boxplot with alpha.
    filename (str): optional path to save figure
    """
    model_names = list(error_dict.keys())
    errors_data = [error_dict[m] for m in model_names]

    plt.figure(figsize=(8, 6))
    box = plt.boxplot(errors_data, labels=model_names, patch_artist=True)

    for patch, name in zip(box['boxes'], model_names):
        c = color_map.get(name, 'blue')
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    plt.title("Distribution of Absolute Errors (Test Set)", fontsize=13, fontweight='bold')
    plt.ylabel("Absolute Error", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

##############################################################################
# 6) Scatter
##############################################################################
def scatter_predictions_vs_truth(truths, preds_dict, color_map, filename=None):
    """
    Plot predictions vs. truths for all models with fancy alpha + same color_map.
    """
    plt.figure(figsize=(6,6))

    for model_name, pred in preds_dict.items():
        c = color_map.get(model_name, 'blue')
        plt.scatter(pred, truths, s=50, alpha=0.6, label=model_name, color=c)

    min_val = min(truths.min(), *(p.min() for p in preds_dict.values()))
    max_val = max(truths.max(), *(p.max() for p in preds_dict.values()))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)

    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("True", fontsize=11)
    plt.title("Predictions vs. True (Test Set)", fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

##############################################################################
# 7) Error Curves
##############################################################################

def plot_error_curves(
    x_plot,
    y_true,
    model_preds_dict,
    color_map,
    emphasize_model="CauchyNet",
    filename=None
):
    """
    Plot error = (predicted - true) for each model vs. x.

    - We highlight the emphasize_model (e.g., "CauchyNet") with a thicker, more opaque line.
    - Other models are plotted in a slightly thinner, more transparent line.
    - This style is often used in ML/physics/engineering references to highlight the best model's error.

    Args:
      x_plot (1D array): x-values
      y_true (1D array): ground truth function
      model_preds_dict (dict): { model_name: array_of_predictions }
      color_map (dict): { model_name: (R,G,B) } for consistent plotting
      emphasize_model (str): which model name to highlight (default "CauchyNet")
      filename (str or None): if given, the figure is saved at 300 DPI
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Sort x for a clean line plot
    x_np = np.array(x_plot)
    sort_idx = np.argsort(x_np)
    x_sorted = x_np[sort_idx]
    y_sorted = np.array(y_true)[sort_idx]

    # 2) Prepare figure
    plt.figure(figsize=(8, 5))

    # 3) Plot each model’s error curve
    for model_name, preds in model_preds_dict.items():
        # Sort predictions
        preds_sorted = np.array(preds)[sort_idx]
        error = preds_sorted - y_sorted

        c = color_map.get(model_name, 'blue')

        if model_name == emphasize_model:
            # Use a thicker line, full alpha, so it stands out
            plt.plot(
                x_sorted, error, '-',
                linewidth=3.5, alpha=0.95,
                label=f"{model_name} Error",
                color=c
            )
        else:
            # Other models: slightly thinner + more transparent
            plt.plot(
                x_sorted, error, '-',
                linewidth=2.0, alpha=0.5,
                label=f"{model_name} Error",
                color=c
            )

    # 4) Reference horizontal line y=0 (perfect prediction)
    plt.axhline(0.0, color='k', linestyle='--', linewidth=2.0, alpha=0.8)

    # 5) Title, labels, legend
    plt.title("Error Curves (Predicted - True)", fontsize=13, fontweight='bold')
    plt.xlabel("x", fontsize=11)
    plt.ylabel("Error", fontsize=11)
    plt.legend(fontsize=9, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 6) Optional save
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
##############################################################################

def subplot_predictions(
    x_plot,
    y_true,
    model_preds_dict,
    color_map,
    plot_residual=True,
    share_y=True,
    filename=None
):
    """
    Create subplots comparing each model's predicted curve with the true curve.
    Optionally, add a second row that shows the residual (true - predicted).

    Args:
        x_plot (1D array): The x-values.
        y_true (1D array): The true function values (same length as x_plot).
        model_preds_dict (dict): {model_name: 1D array of predictions}.
        color_map (dict): {model_name: color tuple}.
        plot_residual (bool): If True, generate a second row for residual plots.
        share_y (bool): If True, enforce the same y-limits across all top subplots.
                        Also enforces consistent y-limits on the residual row.
        filename (str): If provided, save figure to this path.

    Returns:
        None. Displays (and optionally saves) the figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Sort x for a clean line plot
    sort_idx = np.argsort(x_plot)
    x_sorted = x_plot[sort_idx]
    y_sorted = y_true[sort_idx]

    # 2) Prepare subplots. Each model gets 1 subplot (if plot_residual=False)
    #    or 2 subplots stacked vertically (if plot_residual=True).
    n_models = len(model_preds_dict)
    n_cols   = 2 if plot_residual else 1
    n_rows   = n_models
    fig_height = 2.5 * n_models if not plot_residual else 2.5 * n_models
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8 if not plot_residual else 9, fig_height),
                             dpi=300,
                             squeeze=False)

    # 3) Collect data for consistent y-limits if share_y=True
    #    We'll gather the min/max from all models top-row predictions + true
    all_top_values = [y_sorted]  # always include the true curve
    for _, pred_vals in model_preds_dict.items():
        pred_sorted = pred_vals[sort_idx]
        all_top_values.append(pred_sorted)
    y_min_global = min(np.min(arr) for arr in all_top_values)
    y_max_global = max(np.max(arr) for arr in all_top_values)

    # For residual
    if plot_residual:
        # We'll gather min/max of residuals to unify axes
        all_resid_values = []
        for _, pred_vals in model_preds_dict.items():
            pred_sorted = pred_vals[sort_idx]
            resid = y_sorted - pred_sorted  # true - pred
            all_resid_values.append(resid)
        resid_min_global = min(np.min(rr) for rr in all_resid_values)
        resid_max_global = max(np.max(rr) for rr in all_resid_values)

    # Flatten axes so we can index easily: top row is axes[i, 0], optional bottom row axes[i, 1]
    # Actually we have shape => (n_models, n_cols). We'll do a double index.
    # But let's define a helper to get the "top subplot" for model i and "resid subplot" if plot_residual
    def top_ax(i):
        return axes[i, 0] if plot_residual else axes[i, 0]
    def resid_ax(i):
        if plot_residual:
            return axes[i, 1]
        return None  # not used if plot_residual=False

    # 4) Plot each model
    for i, (model_name, y_pred) in enumerate(model_preds_dict.items()):
        # Sort predictions
        y_pred_sorted = y_pred[sort_idx]

        # -- Top Subplot: predicted vs. true
        ax_top = axes[i, 0] if not plot_residual else axes[i, 0]
        ax_top.plot(x_sorted, y_sorted, '--', color='black',
                    linewidth=2.0, alpha=1, label="True")
        c = color_map.get(model_name, 'blue')
        ax_top.plot(x_sorted, y_pred_sorted, color=c, linewidth=2.0,
                    alpha=0.8, label=model_name)

        # Title & grid
        ax_top.set_title(model_name, fontsize=11, fontweight='bold')
        ax_top.grid(alpha=0.3)

        # Optionally unify y-limits
        if share_y:
            ax_top.set_ylim([y_min_global, y_max_global])

        # Minimal legend: If you prefer no legend in each subplot, remove these lines:
        # or keep them if each subplot is small and self-contained
        ax_top.legend(fontsize=8, loc='best')

        # -- Residual Subplot: (true - pred) if plot_residual=True
        if plot_residual:
            ax_resid = axes[i, 1]
            residual = y_sorted - y_pred_sorted
            ax_resid.axhline(0.0, color='gray', linestyle='--', linewidth=2, alpha=0.8)
            ax_resid.plot(x_sorted, residual, color=c, linewidth=2.5, alpha=1)
            ax_resid.set_title(model_name + " Residual", fontsize=10)
            ax_resid.grid(alpha=0.3)
            if share_y:
                ax_resid.set_ylim([resid_min_global, resid_max_global])

            # If you want a legend or label here, you can do so
            # ax_resid.legend(["(True - Pred)"], fontsize=8, loc='best')

    # Hide unused subplots if the total #models < n_rows * n_cols
    # e.g. if you have leftover subplots. 
    for row_idx in range(n_models):
        for col_idx in range(n_cols):
            if row_idx*n_cols + col_idx >= 2*n_models if plot_residual else n_models:
                axes[row_idx, col_idx].axis('off')

    # Optionally add x/y labels to left/bottom axes only, or each axis as you prefer
    # e.g.:
    # for row_idx in range(n_models):
    #     axes[row_idx, 0].set_ylabel("Y", fontsize=9)
    # if plot_residual:
    #     for row_idx in range(n_models-1, n_models):
    #         axes[row_idx, 1].set_xlabel("X", fontsize=9)

    # Adjust spacing
    fig.tight_layout()

    # Overall figure title
    # Could do:
    # fig.suptitle("Predicted vs. True (Subplots)", fontsize=13, fontweight='bold', y=1.02)

    # Save if requested
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
##############################################################################
# 9) Generate LaTeX Table
##############################################################################
# def generate_latex_table(results_summary, filename=None):
#     header = r"\begin{table}[ht]\centering" + "\n"
#     header += r"\begin{tabular}{lcccc}" + "\n"
#     header += r"\toprule" + "\n"
#     header += r"Model & MSE & MAE & Time (s) & \# Params \\" + "\n"
#     header += r"\midrule" + "\n"

#     rows = ""
#     for model_name, stats in results_summary.items():
#         mse_str  = f"{stats['MSE_mean']:.4f}±{stats['MSE_std']:.4f}"
#         mae_str  = f"{stats['MAE_mean']:.4f}±{stats['MAE_std']:.4f}"
#         time_str = f"{stats['Time_mean']:.2f}±{stats['Time_std']:.2f}"
#         params   = f"{stats['Num_Params']}"
#         row = f"{model_name} & {mse_str} & {mae_str} & {time_str} & {params} \\\\ \n"
#         rows += row

#     footer = (
#         r"\bottomrule" + "\n" +
#         r"\end{tabular}" + "\n" +
#         r"\caption{Comparison of Models}" + "\n" +
#         r"\end{table}"
#     )

#     latex_str = header + rows + footer

#     if filename:
#         with open(filename, "w") as f:
#             f.write(latex_str)

#     return latex_str


import numpy as np
import matplotlib.pyplot as plt

def scatter_predictions_vs_truth_improved(
    y_true,
    preds_dict,
    color_map,
    marker_size=30,
    alpha_val=0.9,
    add_stats=True,
    filename=None
):
    """
    Create a single scatter plot of Predicted vs. True for all models,
    each model in a distinct color from 'color_map', with:
      - diagonal y=x reference,
      - optional alpha blending to reduce clutter,
      - optional stats annotation (e.g. R^2 or correlation) in the corner,
      - one global legend.

    Args:
        y_true (1D array): shape (N, ), the ground-truth values (unscaled).
        preds_dict (dict): { model_name: predicted_values (1D array, length N) }
        color_map (dict): { model_name: (R,G,B) }, consistent with other plots
        marker_size (int): size of scatter markers
        alpha_val (float): alpha for scatter points
        add_stats (bool): if True, compute & annotate correlation or R^2
        filename (str): if given, save the figure to 'filename' (png/pdf/svg)
    """
    plt.figure(figsize=(6,6))
    min_val = np.min(y_true)
    max_val = np.max(y_true)

    # We'll collect correlation or R^2 data (optional)
    stats_info = []

    for model_name, pred_array in preds_dict.items():
        c = color_map.get(model_name, 'blue')
        plt.scatter(
            pred_array, y_true,
            s=marker_size, alpha=alpha_val,
            label=model_name, color=c
        )

        if add_stats:
            # e.g. compute correlation (Pearson's r) or R^2
            # - correlation:
            corr = np.corrcoef(pred_array, y_true)[0,1]
            # - R^2 (coefficient of determination):
            #   1 - sum((y_true - pred)^2)/sum((y_true - mean(y_true))^2)
            ss_res = np.sum((y_true - pred_array)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1 - (ss_res/ss_tot)

            stats_info.append((model_name, corr, r2))

        # Update min_val, max_val for the entire set of points
        min_val = min(min_val, pred_array.min())
        max_val = max(max_val, pred_array.max())

    # 1) Plot reference diagonal y=x
    plt.plot([min_val, max_val], [min_val, max_val],
             '--', color='black', linewidth=1.5, alpha=0.8)

    # 2) Axes labels & title
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("True", fontsize=11)
    plt.title("Predictions vs. True (All Models)", fontsize=13, fontweight='bold')

    # 3) Legend
    plt.legend(fontsize=9, loc="best")

    # 4) Optionally add stats annotation in corner
    if add_stats:
        # Build a multiline string with correlation or R^2 for each model
        # e.g. "ModelA: R^2=0.93, corr=0.96\nModelB: R^2=0.91, corr=0.94 ..."
        lines = []
        for (mname, corr, r2) in stats_info:
            lines.append(f"{mname}: $R^2$={r2:.2f}, corr={corr:.2f}")
        text_str = "\n".join(lines)
        plt.gca().text(
            0.05, 0.95, text_str,
            transform=plt.gca().transAxes,
            va='top', ha='left',
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5)
        )

    # 5) Grid, layout
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 6) Save if needed
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
import pandas as pd

import pandas as pd

import pandas as pd

# def generate_latex_table(results_summary, filename=None):
#     """
#     Manual 'booktabs' approach for older pandas versions.
#     """
#     rows = []
#     for model_name, stats in results_summary.items():
#         mse_str  = f"{stats['MSE_mean']:.4f}±{stats['MSE_std']:.6f}"
#         mae_str  = f"{stats['MAE_mean']:.4f}±{stats['MAE_std']:.6f}"
#         time_str = f"{stats['Time_mean']:.2f}±{stats['Time_std']:.2f}"
#         params   = stats['Num_Params']
#         rows.append({
#             'Model': model_name,
#             'MSE (mean±std)': mse_str,
#             'MAE (mean±std)': mae_str,
#             'Time (s)': time_str,
#             '# Params': params
#         })
#     df = pd.DataFrame(rows)

#     # 1) Generate basic LaTeX table (no booktabs param)
#     table_latex = df.to_latex(
#         index=False,
#         column_format='lcccc',  # left + 4 centers
#         escape=False,
#         bold_rows=False,
#         header=True
#     )

#     lines = table_latex.split('\n')
#     # Find the first \hline (the one right after column headers):
#     # The second \hline is after column names, the last is at the end.

#     # We'll do a small search approach:
#     for i, line in enumerate(lines):
#         if r'\hline' in line:
#             # Replace the *first* occurrence with \toprule
#             lines[i] = line.replace(r'\hline', r'\toprule', 1)
#             break

#     # Then find the *next* \hline => replace with \midrule
#     for j in range(i+1, len(lines)):
#         if r'\hline' in lines[j]:
#             lines[j] = lines[j].replace(r'\hline', r'\midrule', 1)
#             break

#     # Finally, find the *last* \hline => replace with \bottomrule
#     # We'll search from the bottom
#     for k in range(len(lines)-1, -1, -1):
#         if r'\hline' in lines[k]:
#             lines[k] = lines[k].replace(r'\hline', r'\bottomrule', 1)
#             break

#     # Now we can insert our \caption + \label lines, if we want:
#     # after \begin{tabular}{lcccc} for instance, or at the end
#     # For a minimal approach, let's do it after \begin{table} (you'll need to add manually).
#     # We'll just wrap the final text in \begin{table}...\end{table} if you want:

#     # Reconstruct the table string
#     table_latex = "\n".join(lines)

#     # Optional: add a simple caption/label after \end{tabular} or before it
#     # We'll do it in the minimal approach:
#     table_latex = (
#         "\\begin{table}[ht]\n"
#         "\\centering\n"
#         + table_latex
#         + "\n\\caption{Comparison of Models (Manual Booktabs)}\n"
#         "\\label{tab:manual_booktabs}\n"
#         "\\end{table}"
#     )

#     if filename:
#         with open(filename, 'w') as f:
#             f.write(table_latex)

#     return table_latex
def generate_latex_table(results_summary, filename=None):
    r"""
    Generates a TMLR-style table in LaTeX format using 'booktabs' conventions.
    The function returns the table string, and optionally writes it to a file.

    :param results_summary: Dictionary containing model stats with keys:
                           {
                             'Model': str,
                             'MSE_mean': float,
                             'MSE_std': float,
                             'MAE_mean': float,
                             'MAE_std': float,
                             'Time_mean': float,
                             'Time_std': float,
                             'Num_Params': int
                           }
    :param filename: Optional path to save the .tex file
    :return: A string representing the LaTeX table.
    """
    rows = []
    for model_name, stats in results_summary.items():
        mse_str  = f"{stats['MSE_mean']:.4f}±{stats['MSE_std']:.6f}"
        mae_str  = f"{stats['MAE_mean']:.4f}±{stats['MAE_std']:.6f}"
        time_str = f"{stats['Time_mean']:.2f}±{stats['Time_std']:.2f}"
        params   = stats['Num_Params']
        rows.append({
            'Model': model_name,
            'MSE (mean±std)': mse_str,
            'MAE (mean±std)': mae_str,
            'Time (s)': time_str,
            '# Params': params
        })

    df = pd.DataFrame(rows)

    # Generate initial LaTeX table
    table_latex = df.to_latex(
        index=False,
        column_format='lcccc',  # left + 4 center columns
        escape=False,
        bold_rows=False,
        header=True
    )

    lines = table_latex.split('\n')

    # Replace the first \hline with \toprule
    replaced_top = False
    for i, line in enumerate(lines):
        if r'\hline' in line and not replaced_top:
            lines[i] = line.replace(r'\hline', r'\toprule', 1)
            replaced_top = True
            break

    # Replace the second \hline with \midrule
    replaced_mid = False
    for j in range(i+1, len(lines)):
        if r'\hline' in lines[j] and not replaced_mid:
            lines[j] = lines[j].replace(r'\hline', r'\midrule', 1)
            replaced_mid = True
            break

    # Replace the last \hline with \bottomrule
    for k in range(len(lines)-1, -1, -1):
        if r'\hline' in lines[k]:
            lines[k] = lines[k].replace(r'\hline', r'\bottomrule', 1)
            break

    # Reassemble the table
    table_latex = "\n".join(lines)

    # Wrap in a table environment with TMLR-style caption above
    # (TMLR generally prefers the caption above the table, though not strictly required.)
    table_latex = (
        r"\begin{table}[t]" "\n"
        r"\centering"       "\n"
        r"\caption{Comparison of Models (MSE, MAE, Time, and Parameter Count).}" "\n"
        + table_latex +
        "\n\\label{tab:manual_booktabs}\n"
        "\\end{table}"
    )

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(table_latex)

    return table_latex
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Suppose you have:
#   model_names     = ["CauchyNet", "NBeats", "FNN", "SIREN"]   # etc.
#   abs_errors_all  = [array_of_abs_errs_for_CauchyNet, array_of_abs_errs_for_NBeats, ...]

def plot_boxplot_abs_errors(model_names, abs_errors_all, title="Absolute Error Distribution (Test Set)"):
    """
    Plots a boxplot of absolute errors for multiple models.

    Args:
      model_names     : list of str, e.g. ["CauchyNet", "NBeats", ...]
      abs_errors_all  : list of 1D arrays/lists, each containing absolute errors for that model
      title           : str, the main title of the plot
    """

    # 1) Configure global style suitable for top-tier ML journals
    sns.set_style("whitegrid")
    sns.set_context("paper")  # 'paper' or 'talk' are good; 'poster' is larger
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

    # 2) Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # 3) Plot the boxplot
    # The data input should be a list-of-lists or list-of-np.arrays
    # Each element is all abs errors for that model
    # We can specify a nice color palette
    palette = sns.color_palette("Set2", n_colors=len(model_names))

    # `showfliers=True/False` controls outliers (fliers). You can enable them if you want to see outliers.
    # `width` controls how wide each box is. `linewidth` for the boxes' edges.
    # `medianprops` to highlight the median line, if desired.
    sns.boxplot(
        data=abs_errors_all, 
        palette=palette, 
        width=0.6,
        linewidth=1.0,
        showfliers=True,
        ax=ax
    )

    # 4) Customize ticks/labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_title(title, pad=6)
    ax.set_ylabel("Absolute Error", labelpad=4)

    # Optionally, you can add an overarching grid or light horizontal lines
    # Seaborn with `whitegrid` already does it, but you can adjust more if needed.

    # 5) Remove excessive top/bottom space, show
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# Example usage:

# model_names = ["CauchyNet", "NBeats", "FNN", "SIREN"]
# abs_errors_all = [
#     np.random.rand(50)*0.1,   # CauchyNet errors
#     np.random.rand(50)*0.2,   # NBeats errors
#     np.random.rand(50)*0.15,  # FNN errors
#     np.random.rand(50)*0.18   # SIREN errors
# ]
#
# plot_boxplot_abs_errors(model_names, abs_errors_all,
#                         title="Absolute Error Distribution (Test Set)")


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