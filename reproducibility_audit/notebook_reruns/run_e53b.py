import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import traceback
_n=[0]
_cell=[0]
def _show(*a,**k):
    plt.rcParams['text.usetex'] = False
    plt.gcf().savefig(f'e53b_cell{_cell[0]:02d}_fig{_n[0]:02d}.png', dpi=110, bbox_inches='tight'); _n[0]+=1; plt.close('all')
plt.show=_show

_cell[0]=0
print('=== CELL 0 ===', flush=True)
try:
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    ##############################################################################
    # 1) Global Settings & Style
    ##############################################################################
    matplotlib.rcParams["pdf.fonttype"]  = 42
    matplotlib.rcParams["ps.fonttype"]   = 42
    matplotlib.rcParams["font.family"]   = "sans-serif"
    matplotlib.rcParams["figure.dpi"]    = 120
    
    sns.set_style("whitegrid")            # for a clean, journal-like style
    sns.set_palette("colorblind")         # colorblind-friendly palette
    
    SEED = 10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #from models import CauchyNet, CauchyNet1,SIREN, RBFNetwork, FNN_ReLU, FNN_Sigmoid, MinimalTransformer,NBeats, MinimalInformer
    from modles_Cauchys import  CauchyNet0,  CauchyNet1, CauchyNet1_NoImagPenalty, CauchyNet, CauchyNet_RealActivation, CauchyNet_NoImagPenalty,CauchyNet_PurelyRealParams,CauchyNet_NonHolomorphic
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 0 FAILED, continuing', flush=True)

_cell[0]=1
print('=== CELL 1 ===', flush=True)
try:
    import pandas as pd
    # Load the CSV file into a DataFrame
    
    df = pd.read_csv('M4_trend_data.csv')
    df
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 1 FAILED, continuing', flush=True)

_cell[0]=2
print('=== CELL 2 ===', flush=True)
try:
    import numpy as np
    
    def transform_time_series_to_domain(Y):
        """
        Map a time series Y_0, Y_1, ..., Y_n to a set of points
        (x_k, Y_k) on [x_start, x_end].
        
        Parameters:
        -----------
        Y : array-like
            The time series values of shape (n+1,).
        x_start : float
            Left boundary of the interval for x.
        x_end : float
            Right boundary of the interval for x.
        
        Returns:
        --------
        X : numpy.ndarray
            1D array of shape (n+1,) containing the mapped x-coordinates.
        Y : numpy.ndarray
            Same as input Y but converted to numpy array for consistency.
        """
        x_start=-1.0, 
        x_end=1.0
        Y = np.asarray(Y)
        n = len(Y) - 1  # because we have Y_0,...,Y_n (total n+1 points)
        if n < 1:
            raise ValueError("Time series must contain at least two points.")
        
        # Compute step size
        length = 2
        h = length / n
    
        # Create the domain array
        X = np.array([-1 + k*h for k in range(n+1)])
        
        return X, Y
    
    
    
    Y_vals = df['Y']
        
    X_mapped, Y_mapped = transform_time_series_to_domain(Y_vals)
        
    
    # Plot the function
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(X_mapped, Y_mapped, color="blue", linewidth=2, label="f(x)")
    #plt.title("Graph of function f(x) on [-1, 1]", fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("M4f.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 2 FAILED, continuing', flush=True)

_cell[0]=3
print('=== CELL 3 ===', flush=True)
try:
    def challenging_function(x):
        """
        A highly non-symmetric function designed to highlight differences between CauchyNet
        and sine-activation-based models like SIREN.
        """
        return (
            1 / ((x + 0.6)**2 + 0.005)   # Sharp positive peak at x=0.1
                        - 40*torch.exp(-2 * (x + 0.4)**2)  # Localized Gaussian bump at x=0.4
                    # +x**2 / ((x - 0.8)**4 + 0.01)     # Sharp negative peak at x=0.8
                + 50 * torch.sign(x) * torch.abs(torch.sin(4*x)+0.8)**1.5 * torch.sin(10*x) # Asymmetric cubic-like feature
        )
    
    # Generate x-values and compute y-values
    x_vals = torch.linspace(-1, 1, steps=150)  # Input range [-1, 1]
    y_vals = challenging_function(x_vals)  # Compute function outputs
    
    # Plot the function
    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(x_vals.numpy(), y_vals.numpy(),color='blue', s=5, label="($x_i, y_i$)")
    #plt.title("Graph of function f(x) on [-1, 1]", fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("toyf.png", dpi=300, bbox_inches='tight')
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 3 FAILED, continuing', flush=True)

_cell[0]=4
print('=== CELL 4 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from sklearn.preprocessing import MinMaxScaler
    
    # =============== Your Utility Functions ===============
    # Make sure these exist in your environment
    from utils2 import (
        loadData,                    # or define below
        train_and_evaluate_model,    # must be defined or imported
        plot_training_curves_with_confidence,  # must be defined or imported
        generate_latex_table,        # must be defined or imported
        get_model_colors,            # must be defined or imported
        plot_box_errors,             # must be defined or imported
        scatter_predictions_vs_truth,  # optional usage
        subplot_predictions,         # must be defined or imported
        plot_error_curves            # must be defined or imported
    )
    # import torch
    # from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
    
    # Y_vals = df['Y']  # e.g., a pandas Series of shape (N,)
    
    # # Transform to domain
    # X_mapped, Y_mapped = transform_time_series_to_domain(Y_vals)
    
    # # =============== 3) Set up Torch Tensors ===============
    # X = torch.tensor(X_mapped, dtype=torch.float32)
    # Y = torch.tensor(Y_mapped, dtype=torch.float32)
    
    # # =============== 4) Scale Y ===============
    # scaler = MinMaxScaler()
    # Y_np   = Y.numpy().reshape(-1, 1)
    # Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    # Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
    # # =============== 5) Data Splits ===============
    # # If `loadData` is your own function, you can just call it:
    # train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    
    import torch
    import pickle
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    
    # 1) Configure the environment style
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    # 2) Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3) Generate or load your data (Replace with your real data loading)
    data_size = 300
    X = torch.linspace(-1, 1, data_size)
    Y = challenging_function(X)  # e.g. shape (300,)
    
    # 4) Scale Y
    scaler = MinMaxScaler()
    Y_np   = Y.numpy().reshape(-1, 1)         # shape (300,1)
    Y_norm = scaler.fit_transform(Y_np).flatten()  # shape (300,)
    Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
    # 5) Build dataset + splits (loadData is your own function)
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 4 FAILED, continuing', flush=True)

_cell[0]=5
print('=== CELL 5 ===', flush=True)
try:
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
    
    def get_model_colors(model_names, emphasize_model="Cauchy(01)"):
        """
        Create a color map for each model using a custom 'Tableau 10' palette
        that is generally pleasing and balanced.
        Then use a gold highlight for `emphasize_model`.
        """
        # Tableau 10 palette in RGB tuples (normalized to 0..1)
        tableau10 = [
            (0.12156862745098039, 0.4666666666666667,  0.7058823529411765),  # blue
            (0.17254901960784313,  0.6274509803921569, 0.17254901960784313), # green
            (0.8392156862745098,   0.15294117647058825,0.1568627450980392),  # red
            (0.5803921568627451,   0.403921568627451,  0.7411764705882353),  # purple
            (0.4980392156862745,   0.4980392156862745, 0.4980392156862745),  # gray
            (0.7372549019607844,   0.7411764705882353, 0.13333333333333333), # olive
        ]
    
        # A gold highlight color (goldenrod) for emphasize_model
        highlight_color = (1.0, 0.4980392156862745, 0.054901960784313725)
    
        color_map = {}
        i_pal = 0
        for mname in model_names:
            if mname == emphasize_model:
                color_map[mname] = highlight_color
            else:
                color_map[mname] = tableau10[i_pal % len(tableau10)]
                i_pal += 1
    
        return color_map
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 5 FAILED, continuing', flush=True)

_cell[0]=6
print('=== CELL 6 ===', flush=True)
try:
    
    
    # =============== 6) Model Dictionary ===============
    # Make sure these classes are defined or imported
    # (Replace 'CauchyNet1' with 'CauchyNet' if that’s your real class name)
    models_dict = { "Cauchy": CauchyNet,
       # "CauchyNet0": CauchyNet1,
       # "CauchyNet_NIP": CauchyNet_NoImagPenalty,
       # "CauchyNet1": CauchyNet1,
       # "CauchyNet1_NIP": CauchyNet1_NoImagPenalty,
       # "CauchyNet3": CauchyNet_RealActivation,
       # "CauchyNet4": CauchyNet_PurelyRealParams,
    }
    
    # =============== 7) Create Color Map for Models ===============
    # #model_names = list(models_dict.keys())
    # # Generate model names (for example)
    # model_names = ["CauchyNet_a0", "CauchyNet_a0.1", "CauchyNet_a0.5", "CauchyNet_a1", "CauchyNet_a2"]
    
    # # Get the color map for the models (with 'CauchyNet' emphasized)
    # color_map = get_model_colors(model_names, emphasize_model="CauchyNet_a0")
    
    # =============== 7) Create Color Map for Models ===============
    model_names = ["Cauchy(0.1)", "Cauchy(0.3)", "Cauchy(0.5)", "Cauchy(1)", "Cauchy(1.5)"]
    
    # Get the color map for the models (with 'CauchyNet' emphasized)
    color_map = get_model_colors(model_names)#, emphasize_model="CauchyNet_a01")  # Adjusted to match format
    
    #color_map   = get_model_colors(model_names, emphasize_model="CauchyNet")
    
    # =============== 8) Train Configuration ===============
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size   = 1
    hidden_size  = 128
    output_size  = 1
    lr           = 0.01
    epochs       = 200
    num_repeats  = 10
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 6 FAILED, continuing', flush=True)

_cell[0]=7
print('=== CELL 7 ===', flush=True)
try:
    # =============== 9) Train Multiple Runs & Collect Logs ===============
    interval_logs = {}
    results_dict = {}
    final_models = {}  # To store the final models for each a
    training_logs = {}  # To store the training logs for each model
    
    # Values for parameter 'a' to experiment with
    a_values = [0.1, 0.3, 0.5, 1, 1.5]
    
    for model_name, constructor in models_dict.items():
        print(f"=== {model_name} ===")
        
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        Num_Params = None
    
        for a in a_values:  # Loop over different values of 'a'
            print(f"Training with a = {a}")
            a_key = f"{model_name}({str(a)})"  # Generate a unique key for each 'a'
            
            # Initialize lists to store logs for this specific 'a'
            interval_logs[a_key] = {'train': [], 'val': []}
            results_dict[a_key] = {
                'MSE_mean': None,
                'MSE_std': None,
                'MAE_mean': None,
                'MAE_std': None,
                'Time_mean': None,
                'Time_std': None,
                'Num_Params': None
            }
    
            for _ in range(num_repeats):
                # Train and evaluate model with parameter 'a'
                model, train_losses, val_losses, test_mse, test_mae, preds_scaled, truths_scaled, training_time, Num_Params = train_and_evaluate_model(
                    constructor,
                    input_size, hidden_size, output_size,
                    train_loader, val_loader, test_loader,
                    lr=lr, epochs=epochs, device=device, a=a
                )
    
                # Store the model for later use (final model for each a)
                if model_name not in final_models:
                    final_models[model_name] = {}  # Initialize the dictionary for the model name if not already there
                final_models[model_name][a] = model  # Store the model corresponding to parameter 'a'
                
                # Store the logs for this value of 'a'
                interval_logs[a_key]['train'].append(train_losses)
                interval_logs[a_key]['val'].append(val_losses)
    
                runs_mse.append(test_mse)
                runs_mae.append(test_mae)
                runs_time.append(training_time)
    
            # Compute statistics for each value of 'a'
            mse_mean, mse_std = np.mean(runs_mse), np.std(runs_mse)
            mae_mean, mae_std = np.mean(runs_mae), np.std(runs_mae)
            time_mean, time_std = np.mean(runs_time), np.std(runs_time)
    
            # Store the results in the results_dict
            results_dict[a_key] = {
                'MSE_mean': mse_mean,
                'MSE_std': mse_std,
                'MAE_mean': mae_mean,
                'MAE_std': mae_std,
                'Time_mean': time_mean,
                'Time_std': time_std,
                'Num_Params': Num_Params
            }
    
            # Store training logs for each model in training_logs dictionary
            if model_name not in training_logs:
                training_logs[model_name] = {}  # Initialize the dictionary for the model name if not already there
            training_logs[model_name][a] = {
                'MSE_mean': mse_mean,
                'MSE_std': mse_std,
                'MAE_mean': mae_mean,
                'MAE_std': mae_std,
                'Time_mean': time_mean,
                'Time_std': time_std
            }
    
    # Ensure model names for plotting are based on the keys of interval_logs
    model_names = list(interval_logs.keys())
    
    # Plot training curves with confidence intervals for each 'a'
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves_all_a_values.pdf")
    
    # Now you can pass the color map to your plotting function
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves_all_a_values.pdf")
    
    # Optionally, save final models to disk
    for model_name, models in final_models.items():
        for a, model in models.items():
            torch.save(model.state_dict(), f"{model_name}({str(a)})_final_model.pth")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 7 FAILED, continuing', flush=True)

_cell[0]=8
print('=== CELL 8 ===', flush=True)
try:
    import json
    import numpy as np
    
    # Function to convert NumPy types to Python native types
    def convert_to_python_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Converts numpy float32 to native float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        else:
            return obj
    
    # Convert your logs to Python native types (to avoid JSON serialization issues)
    training_logs_python = convert_to_python_types(training_logs)
    
    # Save the logs in a JSON file
    with open("training_logs.json", "w") as log_file:
        json.dump(training_logs_python, log_file, indent=4)
    
    print("\nTraining logs saved successfully.")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 8 FAILED, continuing', flush=True)

_cell[0]=9
print('=== CELL 9 ===', flush=True)
try:
    len(list(final_models.keys()))
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 9 FAILED, continuing', flush=True)

_cell[0]=10
print('=== CELL 10 ===', flush=True)
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define a distinct color palette using Seaborn's color palette
    def generate_distinct_color_map(models):
        # Use a Seaborn palette with more distinct colors
        colors = sns.color_palette("husl", len(models))  # 'husl' is a distinct palette
        return {model_name: color for model_name, color in zip(models, colors)}
    
    color_map = generate_distinct_color_map(model_names)  # Update color map based on your models
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 10 FAILED, continuing', flush=True)

_cell[0]=11
print('=== CELL 11 ===', flush=True)
try:
    def plot_predictions_on_grid(X, Y_true, overlay_preds, color_map, filename):
        plt.figure(figsize=(10, 6))
    
        # Plot the true values
        plt.plot(X, Y_true, label='True', color='black', linewidth=2)
    
        # Plot predictions for each model
        for model_name, preds in overlay_preds.items():
            plt.plot(X, preds, label=model_name, linewidth=2)
    
        plt.xlabel('X values', fontsize=14)
        plt.ylabel('Predicted / True Y', fontsize=14)
        plt.title('Predictions on Uniform Grid', fontsize=16)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
    
        # Save and show plot
        plt.savefig(filename, dpi=300)
        plt.show()
    
    # Call the function
    plot_predictions_on_grid(X_plot_np, Y_plot_true_np, overlay_preds, color_map, "predictions_on_grid.png")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 11 FAILED, continuing', flush=True)

_cell[0]=12
print('=== CELL 12 ===', flush=True)
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define a distinct color palette using Seaborn's color palette
    def generate_distinct_color_map(models):
        # Use a Seaborn palette with more distinct colors
        colors = sns.color_palette("husl", len(models))  # 'husl' is a distinct palette
        return {model_name: color for model_name, color in zip(models, colors)}
    
    # Assuming `final_models` is a dictionary with model names as keys
    model_names = [f"{model_name}(str(a))" for model_name in final_models for a in final_models[model_name]]
    color_map = generate_distinct_color_map(model_names)  # Update color map based on your models
    
    color_map = get_model_colors(model_names)#, emphasize_model="CauchyNet_a01")  # Adjusted to match format
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 12 FAILED, continuing', flush=True)

_cell[0]=13
print('=== CELL 13 ===', flush=True)
try:
    # =============== 12) Evaluate Predictions on a Uniform Grid ===============
    # Example: last 100 points for overlay
    # 1) Torch version for model inference
    X_plot_torch = X[:-60].to(device)  # shape: (100,) or (100,1)
    Y_plot_true_torch = Y[:-60].to(device)  # unscaled ground truth
    
    # 2) Re-fit the scaler on the original Y_np for consistent inverse transform
    scaler.fit(Y_np)
    
    # 3) Inference with each final model, storing unscaled predictions in `overlay_preds`
    overlay_preds = {}
    for model_name, models in final_models.items():
        # Now loop over each model for different values of 'a' (models is a dictionary here)
        for a, model in models.items():
            model.eval()  # Switch model to evaluation mode for inference
            with torch.no_grad():
                # If your model expects shape (N,1), use unsqueeze(-1).float().
                # If it's (N,), adjust as needed.
                out = model(X_plot_torch.unsqueeze(-1).float())
    
                # Some models return a tuple (y_pred, ...); handle that:
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_scaled = out.cpu().numpy().flatten()
    
            # 4) Convert scaled predictions back to original scale
            preds_2d = preds_scaled.reshape(-1, 1)
            preds_unscaled = scaler.inverse_transform(preds_2d).flatten()
            overlay_preds[f"{model_name}({str(a)})"] = preds_unscaled
    
    # 5) For plotting, convert the last 100 points to NumPy arrays
    X_plot_np = X_plot_torch.cpu().numpy()      # shape (100,) or (100,1)
    Y_plot_true_np = Y_plot_true_torch.cpu().numpy() # shape (100,)
    
    # =============== 13) Subplot Predictions ===============
    # Now pass NumPy arrays to your plotting function.
    
    
    subplot_predictions(
        X_plot_np,
        Y_plot_true_np,
        overlay_preds,
        color_map,
        filename="subplot_predictions2.png"
    )
    
    # =============== 14) Box Plot of Errors ===============
    error_dict = {}
    for model_name, models in final_models.items():
        for a, model in models.items():
            model.eval()
            abs_errors = []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, _ = out
                        preds = y_real.flatten()
                    else:
                        preds = out.cpu().numpy().flatten()
                    truths = y_batch.cpu().numpy().flatten()
                    abs_errors.extend(np.abs(preds - truths))
            error_dict[f"{model_name}({str(a)})"] = np.array(abs_errors)
    
    plot_box_errors(error_dict, color_map, filename="box_errors2.png")
    
    # =============== 15) Print Summary ===============
    print("\n=== Summary of Results ===")
    for mname, stats in results_dict.items():
        print(f"{mname:12s} => "
              f"MSE={stats['MSE_mean']:.6f}±{stats['MSE_std']:.4f}, "
              f"MAE={stats['MAE_mean']:.4f}±{stats['MAE_std']:.4f}, "
              f"Time={stats['Time_mean']:.2f}±{stats['Time_std']:.2f}s, "
              f"#Params={stats['Num_Params']}")
    
    # =============== 16) Generate LaTeX Table ===============
    latex_str = generate_latex_table(results_dict, filename="results_summary.tex")
    print("\nLaTeX Table:\n")
    print(latex_str)
    print("\nSaved to 'results_summary2.tex'")
    
    # =============== 17) Plot Error Curves ===============
    # If your error-curve function also needs NumPy data, pass X_plot_np, Y_plot_true_np, etc.
    plot_error_curves(X_plot_np, Y_plot_true_np, overlay_preds, color_map, filename="error_curves2.png")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 13 FAILED, continuing', flush=True)

_cell[0]=14
print('=== CELL 14 ===', flush=True)
try:
    # 1) Generate a distinct color map
    color_map = generate_distinct_color_map(model_names)  # Update color map for all models
    
    # 2) Plot Subplot Predictions
    subplot_predictions(
        X_plot_np,
        Y_plot_true_np,
        overlay_preds,
        color_map,
        filename="subplot_predictions2.png"
    )
    
    # 3) Box Plot of Errors
    error_dict = {}
    for model_name, models in final_models.items():
        for a, model in models.items():
            model.eval()
            abs_errors = []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, _ = out
                        preds = y_real.flatten()
                    else:
                        preds = out.cpu().numpy().flatten()
                    truths = y_batch.cpu().numpy().flatten()
                    abs_errors.extend(np.abs(preds - truths))
            error_dict[f"{model_name}_a{str(a).replace('.', '')}"] = np.array(abs_errors)
    
    # Plot box errors
    plot_box_errors(error_dict, color_map, filename="box_errors2.png")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 14 FAILED, continuing', flush=True)

_cell[0]=15
print('=== CELL 15 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Example data (replace with your actual results from evaluation)
    T = ["a=0.1", "a=0.3", "a=0.5", "a=1", "a=1.5"]  # Values of 'a'
    sym_mse_test = [0.05, 0.07, 0.04, 0.03, 0.02]  # Example MSE for each model
    pinn_mse_test = [0.06, 0.08, 0.05, 0.04, 0.03]  # Example MSE for each model
    sym_time = [200, 250, 180, 220, 210]  # Example training time for each model (in seconds)
    pinn_time = [180, 230, 170, 210, 200]  # Example training time for each model (in seconds)
    
    T_indices = np.arange(len(T))  # [0, 1, 2, 3, 4]
    
    # Creating the plot
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    # Plotting MSE on the first axis
    ax1.plot(T_indices, sym_mse_test, marker='o', label='USD MSE', color='#581036', linestyle='-', linewidth=2.5, markersize=6)
    ax1.plot(T_indices, pinn_mse_test, marker='s', label='PINN MSE', color='#273977', linestyle='--', linewidth=2.5, markersize=6)
    ax1.set_ylabel('Testing MSE', fontsize=14, color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_xticks(T_indices)  
    ax1.set_xticklabels([f"T={t}" for t in T])
    ax1.legend(loc='upper left', fontsize=10, frameon=False)
    
    # Creating a second axis for the bar plots
    ax2 = ax1.twinx()
    
    bar_width = 0.2  
    
    # Plotting bar charts for training time
    bar1 = ax2.bar(T_indices - bar_width/2, sym_time, bar_width, color='#B98E95', alpha=0.6, label='USD Time', edgecolor='black')
    bar2 = ax2.bar(T_indices + bar_width/2, pinn_time, bar_width, color='#ACCBE8', alpha=0.6, label='PINN Time', edgecolor='black')
    
    ax2.set_ylim(0, max(max(sym_time), max(pinn_time)) * 1.5)  # Extend the time axis a bit
    ax2.set_ylabel('Training Time (s)', fontsize=14, color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
    
    ax2.legend(loc='upper right', fontsize=10, frameon=False)
    
    # Adding title and grid
    plt.title('Sampling method: CauchyNet Model Comparison', fontsize=16, pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 15 FAILED, continuing', flush=True)

_cell[0]=16
print('=== CELL 16 ===', flush=True)
try:
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
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 16 FAILED, continuing', flush=True)

_cell[0]=17
print('=== CELL 17 ===', flush=True)
try:
    # =============== 1) Save the Final Models ===============
    for model_name, model in final_models.items():
        save_path = f"{model_name}_final.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved {model_name} to {save_path}")
    
    # =============== 2) Load and Continue Training ===============
    continued_epochs = 100
    
    # Keep your same device and hyperparameters as before
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size   = 1
    hidden_size  = 128
    output_size  = 1
    lr           = 0.001
    epochs       = 50
    num_repeats  = 10
    
    interval_logs  = {}
    results_dict   = {}
    
    for model_name, constructor in models_dict.items():
        load_path = f"{model_name}_final.pth"
    
        # (A) Define a local constructor-like function that:
        #     - Creates a fresh instance of the same class,
        #     - Loads the saved state_dict,
        #     - Returns the loaded model (matching the original signature).
        def loaded_model_constructor(inp_size, hid_size, out_size):
            new_model = constructor(inp_size, hid_size, out_size).to(device)
            # Weights-only=True is recommended to avoid future pickle warnings:
            state_dict = torch.load(load_path, weights_only=True)
            new_model.load_state_dict(state_dict)
            new_model.train()
            return new_model
    
        print(f"Loaded {model_name} from {load_path} and continuing training for {continued_epochs} epochs...")
        print(f"=== {model_name} ===")
    
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        Num_Params = None
    
        # (B) Now call train_and_evaluate_model EXACTLY as in your original code,
        #     but replace the old 'constructor' with 'loaded_model_constructor'.
        for _ in range(num_repeats):
            (
                model,
                train_losses,
                val_losses,
                test_mse,
                test_mae,
                preds_scaled,
                truths_scaled,
                training_time,
                Num_Params
            ) = train_and_evaluate_model(
                loaded_model_constructor,
                input_size, hidden_size, output_size,
                train_loader, val_loader, test_loader,
                lr=lr, epochs=continued_epochs, device=device, a=0.1
            )
    
            runs_mse.append(test_mse)
            runs_mae.append(test_mae)
            runs_time.append(training_time)
            train_runs.append(train_losses)
            val_runs.append(val_losses)
    
        # (C) Aggregate results across runs
        mse_mean, mse_std   = np.mean(runs_mse),  np.std(runs_mse)
        mae_mean, mae_std   = np.mean(runs_mae),  np.std(runs_mae)
        time_mean, time_std = np.mean(runs_time), np.std(runs_time)
    
        interval_logs[model_name] = {
            'train': train_runs,
            'val':   val_runs
        }
        results_dict[model_name] = {
            'MSE_mean':   mse_mean,
            'MSE_std':    mse_std,
            'MAE_mean':   mae_mean,
            'MAE_std':    mae_std,
            'Time_mean':  time_mean,
            'Time_std':   time_std,
            'Num_Params': Num_Params
        }
    
    # =============== 3) Plot Training Curves ===============
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves3.pdf")
    
    # =============== 4) Final Run for Each Model ===============
    final_models = {}
    for model_name, constructor in models_dict.items():
        model, *_ = train_and_evaluate_model(
            constructor,
            input_size, hidden_size, output_size,
            train_loader, val_loader, test_loader,
            lr=lr, epochs=epochs, device=device
        )
        final_models[model_name] = model
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 17 FAILED, continuing', flush=True)

_cell[0]=18
print('=== CELL 18 ===', flush=True)
try:
    # =============== 1) Save the Final Models ===============
    for model_name, model in final_models.items():
        save_path = f"{model_name}_final.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved {model_name} to {save_path}")
    
    # =============== 2) Load and Continue Training ===============
    continued_epochs = 100
    
    # Keep your same device and hyperparameters as before
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size   = 1
    hidden_size  = 128
    output_size  = 1
    lr           = 0.01
    epochs       = 100
    num_repeats  = 10
    
    interval_logs  = {}
    results_dict   = {}
    
    for model_name, constructor in models_dict.items():
        load_path = f"{model_name}_final.pth"
    
        # (A) Define a local constructor-like function that:
        #     - Creates a fresh instance of the same class,
        #     - Loads the saved state_dict,
        #     - Returns the loaded model (matching the original signature).
        def loaded_model_constructor(inp_size, hid_size, out_size):
            new_model = constructor(inp_size, hid_size, out_size).to(device)
            # Weights-only=True is recommended to avoid future pickle warnings:
            state_dict = torch.load(load_path, weights_only=True)
            new_model.load_state_dict(state_dict)
            new_model.train()
            return new_model
    
        print(f"Loaded {model_name} from {load_path} and continuing training for {continued_epochs} epochs...")
        print(f"=== {model_name} ===")
    
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        Num_Params = None
    
        # (B) Now call train_and_evaluate_model EXACTLY as in your original code,
        #     but replace the old 'constructor' with 'loaded_model_constructor'.
        for _ in range(num_repeats):
            (
                model,
                train_losses,
                val_losses,
                test_mse,
                test_mae,
                preds_scaled,
                truths_scaled,
                training_time,
                Num_Params
            ) = train_and_evaluate_model(
                loaded_model_constructor,
                input_size, hidden_size, output_size,
                train_loader, val_loader, test_loader,
                lr=lr, epochs=continued_epochs, device=device
            )
    
            runs_mse.append(test_mse)
            runs_mae.append(test_mae)
            runs_time.append(training_time)
            train_runs.append(train_losses)
            val_runs.append(val_losses)
    
        # (C) Aggregate results across runs
        mse_mean, mse_std   = np.mean(runs_mse),  np.std(runs_mse)
        mae_mean, mae_std   = np.mean(runs_mae),  np.std(runs_mae)
        time_mean, time_std = np.mean(runs_time), np.std(runs_time)
    
        interval_logs[model_name] = {
            'train': train_runs,
            'val':   val_runs
        }
        results_dict[model_name] = {
            'MSE_mean':   mse_mean,
            'MSE_std':    mse_std,
            'MAE_mean':   mae_mean,
            'MAE_std':    mae_std,
            'Time_mean':  time_mean,
            'Time_std':   time_std,
            'Num_Params': Num_Params
        }
    
    # =============== 3) Plot Training Curves ===============
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves3.pdf")
    
    # =============== 4) Final Run for Each Model ===============
    final_models = {}
    for model_name, constructor in models_dict.items():
        model, *_ = train_and_evaluate_model(
            constructor,
            input_size, hidden_size, output_size,
            train_loader, val_loader, test_loader,
            lr=lr, epochs=epochs, device=device
        )
        final_models[model_name] = model
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 18 FAILED, continuing', flush=True)

_cell[0]=19
print('=== CELL 19 ===', flush=True)
try:
    for model_name, model in final_models.items():
        save_path = f"{model_name}_final.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved {model_name} to {save_path}")
    
    # ==========================================
    # 2) Loading the models and continuing train
    # ==========================================
    # You can now load the saved weights into a fresh instance
    # of the same model class, and then train again for more epochs.
    
    continued_epochs = 100
    
    # =============== 8) Train Configuration ===============
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size   = 1
    hidden_size  = 128
    output_size  = 1
    lr           = 0.01
    epochs       = 100
    num_repeats  = 10
    
    # =============== 9) Train Multiple Runs & Collect Logs ===============
    interval_logs  = {}
    results_dict   = {}
    
    for model_name, constructor in models_dict.items():
        # 2.1) Construct the same architecture
        loaded_model = constructor(input_size, hidden_size, output_size).to(device)
    
        # 2.2) Load the saved state_dict
        load_path = f"{model_name}_final.pth"
        loaded_model.load_state_dict(torch.load(load_path))
        loaded_model.train()
        print(f"Loaded {model_name} from {load_path} and continuing training for {continued_epochs} epochs...")
        print(f"=== {model_name} ===")
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        Num_Params = None
    
        for _ in range(num_repeats):
            (model, 
             train_losses, 
             val_losses,
             test_mse, 
             test_mae, 
             preds_scaled,
             truths_scaled,
             training_time,
             Num_Params) = train_and_evaluate_model(
                constructor,
                input_size, hidden_size, output_size,
                train_loader, val_loader, test_loader,
                lr=lr, epochs=epochs, device=device
            )
    
            runs_mse.append(test_mse)
            runs_mae.append(test_mae)
            runs_time.append(training_time)
            train_runs.append(train_losses)
            val_runs.append(val_losses)
    
        # Mean ± Std
        mse_mean, mse_std   = np.mean(runs_mse),  np.std(runs_mse)
        mae_mean, mae_std   = np.mean(runs_mae),  np.std(runs_mae)
        time_mean, time_std = np.mean(runs_time), np.std(runs_time)
    
        interval_logs[model_name] = {
            'train': train_runs,
            'val':   val_runs
        }
        results_dict[model_name] = {
            'MSE_mean':    mse_mean,
            'MSE_std':     mse_std,
            'MAE_mean':    mae_mean,
            'MAE_std':     mae_std,
            'Time_mean':   time_mean,
            'Time_std':    time_std,
            'Num_Params':  Num_Params
        }
    
    # =============== 10) Plot Training Curves (No LaTeX Error) ===============
    # Make sure your 'plot_training_curves_with_confidence' function does NOT use
    # double backslashes in the title. Use plain text or properly escaped LaTeX.
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves3.pdf")
    
    # =============== 11) Final Run for Each Model ===============
    final_models = {}
    for model_name, constructor in models_dict.items():
        model, *_ = train_and_evaluate_model(
            constructor,
            input_size, hidden_size, output_size,
            train_loader, val_loader, test_loader,
            lr=lr, epochs=epochs, device=device
        )
        final_models[model_name] = model
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 19 FAILED, continuing', flush=True)

_cell[0]=20
print('=== CELL 20 ===', flush=True)
try:
    
    
    # =============== 12) Evaluate Predictions on a Uniform Grid ===============
    # Example: last 100 points for overlay
    # 1) Torch version for model inference
    X_plot_torch      = X[:-60].to(device)  # shape: (100,) or (100,1)
    Y_plot_true_torch = Y[:-60].to(device)  # unscaled ground truth
    
    # 2) Re-fit the scaler on the original Y_np for consistent inverse transform
    scaler.fit(Y_np)
    
    # 3) Inference with each final model, storing unscaled predictions in `overlay_preds`
    overlay_preds = {}
    for model_name, model in final_models.items():
        model.eval()
        with torch.no_grad():
            # If your model expects shape (N,1), use unsqueeze(-1).float().
            # If it's (N,), adjust as needed.
            out = model(X_plot_torch.unsqueeze(-1).float())
            
            # Some models return a tuple (y_pred, ...); handle that:
            if isinstance(out, tuple):
                y_real, _ = out
                preds_scaled = y_real.cpu().numpy().flatten()
            else:
                preds_scaled = out.cpu().numpy().flatten()
    
        # 4) Convert scaled predictions back to original scale
        preds_2d        = preds_scaled.reshape(-1, 1)
        preds_unscaled  = scaler.inverse_transform(preds_2d).flatten()
        overlay_preds[model_name] = preds_unscaled
    
    # 5) For plotting, convert the last 100 points to NumPy arrays
    X_plot_np      = X_plot_torch.cpu().numpy()      # shape (100,) or (100,1)
    Y_plot_true_np = Y_plot_true_torch.cpu().numpy() # shape (100,)
    
    # =============== 13) Subplot Predictions ===============
    # Now pass NumPy arrays to your plotting function.
    subplot_predictions(
        X_plot_np,
        Y_plot_true_np,
        overlay_preds,
        color_map,
        filename="subplot_predictions3.png"
    )
    
    # =============== 14) Box Plot of Errors ===============
    error_dict = {}
    for mname, model in final_models.items():
        model.eval()
        abs_errors = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds = y_real.flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths = y_batch.cpu().numpy().flatten()
                abs_errors.extend(np.abs(preds - truths))
        error_dict[mname] = np.array(abs_errors)
    
    plot_box_errors(error_dict, color_map, filename="box_errors3.png")
    
    # =============== 15) Print Summary ===============
    print("\n=== Summary of Results ===")
    for mname, stats in results_dict.items():
        print(f"{mname:12s} => "
              f"MSE={stats['MSE_mean']:.6f}±{stats['MSE_std']:.6f}, "
              f"MAE={stats['MAE_mean']:.4f}±{stats['MAE_std']:.6f}, "
              f"Time={stats['Time_mean']:.2f}±{stats['Time_std']:.2f}s, "
              f"#Params={stats['Num_Params']}")
    
    # =============== 16) Generate LaTeX Table ===============
    latex_str = generate_latex_table(results_dict, filename="results_summary3.tex")
    print("\nLaTeX Table:\n")
    print(latex_str)
    print("\nSaved to 'results_summary2.tex'")
    
    # =============== 17) Plot Error Curves ===============
    # If your error-curve function also needs NumPy data, pass X_plot_np, Y_plot_true_np, etc.
    plot_error_curves(X_plot_np, Y_plot_true_np, overlay_preds, color_map, filename="error_curves3.png")
    
    print("\n=== Done! ===")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 20 FAILED, continuing', flush=True)

_cell[0]=21
print('=== CELL 21 ===', flush=True)
try:
    # ========================
    # 1) Saving the Final Model
    # ========================
    for model_name, model in final_models.items():
        save_path = f"{model_name}_final.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved {model_name} to {save_path}")
    
    # =============================================
    # 2) Loading Models & Continuing Training (100 epochs)
    # =============================================
    continued_epochs = 100
    
    for model_name, constructor in models_dict.items():
        load_path = f"{model_name}_final.pth"
        print(f"Loaded {model_name} from {load_path} and continuing training for {continued_epochs} epochs...")
        
        # -- A) Define a local function with the correct signature
        #     that re-creates the same model architecture, loads 
        #     the saved state dict, and returns it.
        def loaded_model_constructor(in_size, hid_size, out_size):
            # 1) Create a new instance of the same class
            reinit_model = constructor(in_size, hid_size, out_size)
            # 2) Load the state dict (set weights_only=True to suppress the FutureWarning)
            state_dict = torch.load(load_path, weights_only=True)
            reinit_model.load_state_dict(state_dict)
            # 3) Return the loaded (re-initialized) model
            return reinit_model
    
        # -- B) Now call train_and_evaluate_model
        #     exactly as in your original code, passing 
        #     'loaded_model_constructor' as the constructor argument:
        (
            _,
            train_losses_cont,
            val_losses_cont,
            test_mse_cont,
            test_mae_cont,
            preds_scaled_cont,
            truths_scaled_cont,
            training_time_cont,
            Num_Params_cont
        ) = train_and_evaluate_model(
            loaded_model_constructor, 
            input_size, hidden_size, output_size,
            train_loader, val_loader, test_loader,
            lr=lr, epochs=continued_epochs, device=device
        )
        
        print(f"Additional training complete for {model_name}. "
              f"Final test MSE: {test_mse_cont:.4f}, "
              f"Test MAE: {test_mae_cont:.4f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 21 FAILED, continuing', flush=True)

_cell[0]=22
print('=== CELL 22 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import time
    
    # =============== Your Utility Functions ===============
    # Ensure these exist in your environment or adapt accordingly.
    from utils2 import (
        loadData,
        plot_training_curves_with_confidence,
        generate_latex_table,
        get_model_colors,
        plot_box_errors,
        scatter_predictions_vs_truth,
        subplot_predictions,
        plot_error_curves
    )
    
    # ------------------------------------------------------------------------
    # EXAMPLE: Modify your train_and_evaluate_model to accept `scaler`
    # ------------------------------------------------------------------------
    def train_and_evaluate_model(
        model_constructor,
        input_size, hidden_size, output_size,
        train_loader, val_loader, test_loader,
        lr=0.005, epochs=200, device=None,
        scaler=None  # <-- ADDED SCALER HERE
    ):
        """
        Trains a given model (model_constructor) and computes test metrics
        on the UNscaled (original) data.
    
        Parameters
        ----------
        model_constructor : callable
            A class or function that returns a PyTorch model instance.
        input_size : int
        hidden_size : int
        output_size : int
        train_loader, val_loader, test_loader : torch.utils.data.DataLoader
            DataLoaders containing scaled y_batch (since we used scaler).
        lr : float
            Learning rate.
        epochs : int
            Number of epochs.
        device : torch.device
            CPU or GPU.
        scaler : fitted scaler (e.g., MinMaxScaler), optional
            Used to inverse-transform predictions and truths in the test loop.
    
        Returns
        -------
        model : trained torch.nn.Module
        train_losses : list of floats
        val_losses : list of floats
        test_mse : float
            MSE computed on UNscaled data.
        test_mae : float
            MAE computed on UNscaled data.
        preds_all_unscaled : np.ndarray
            Final unscaled predictions from the test set.
        truths_all_unscaled : np.ndarray
            Final unscaled ground truths from the test set.
        training_time : float
            How long training took in seconds.
        Num_Params : int
            Number of trainable parameters in the model.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # -------------------- Build Model --------------------
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        criterion = nn.MSELoss()
    
        train_losses = []
        val_losses   = []
        best_val_loss = float('inf')
        best_state    = None
        Num_Params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        start_time = time.time()
    
        # -------------------- Training Loop --------------------
        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0
            for x_batch, y_batch_scaled in train_loader:
                x_batch = x_batch.to(device)
                y_batch_scaled = y_batch_scaled.to(device)
    
                optimizer.zero_grad()
                out = model(x_batch)
    
                # Some models return (real, imag)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    loss = (criterion(y_real, y_batch_scaled.unsqueeze(-1))
                          + criterion(y_imag, torch.zeros_like(y_imag)))
                else:
                    loss = criterion(out, y_batch_scaled.unsqueeze(-1))
    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_accum += loss.item()
    
            train_loss = train_loss_accum / len(train_loader)
            train_losses.append(train_loss)
    
            # -------------------- Validation Loop --------------------
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch_scaled in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch_scaled = y_batch_scaled.to(device)
                    out = model(x_batch)
    
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        v_loss = (criterion(y_real, y_batch_scaled.unsqueeze(-1))
                                + criterion(y_imag, torch.zeros_like(y_imag)))
                    else:
                        v_loss = criterion(out, y_batch_scaled.unsqueeze(-1))
    
                    val_loss_accum += v_loss.item()
    
            val_loss = val_loss_accum / len(val_loader)
            val_losses.append(val_loss)
    
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
    
            scheduler.step(val_loss)
    
        end_time = time.time()
        training_time = end_time - start_time
    
        # -------------------- Load Best State --------------------
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # -------------------- Test Loop (Compute unscaled errors) --------------------
        model.eval()
        preds_list_scaled  = []
        truths_list_scaled = []
        with torch.no_grad():
            for x_batch, y_batch_scaled in test_loader:
                x_batch = x_batch.to(device)
                y_batch_scaled = y_batch_scaled.to(device)
    
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_scaled = out.cpu().numpy().flatten()
    
                truths_scaled = y_batch_scaled.cpu().numpy().flatten()
                preds_list_scaled.append(preds_scaled)
                truths_list_scaled.append(truths_scaled)
    
        # Convert from lists to arrays
        preds_all_scaled  = np.concatenate(preds_list_scaled)
        truths_all_scaled = np.concatenate(truths_list_scaled)
    
        # If a scaler is provided, invert them back to original domain
        if scaler is not None:
            preds_all_unscaled = scaler.inverse_transform(
                preds_all_scaled.reshape(-1, 1)
            ).flatten()
            truths_all_unscaled = scaler.inverse_transform(
                truths_all_scaled.reshape(-1, 1)
            ).flatten()
        else:
            preds_all_unscaled  = preds_all_scaled
            truths_all_unscaled = truths_all_scaled
    
        # Final test metrics on the ORIGINAL domain
        test_mse = mean_squared_error(truths_all_unscaled, preds_all_unscaled)
        test_mae = mean_absolute_error(truths_all_unscaled, preds_all_unscaled)
    
        return (
            model,
            train_losses,
            val_losses,
            test_mse,
            test_mae,
            preds_all_unscaled,   # unscaled predictions
            truths_all_unscaled,  # unscaled truths
            training_time,
            Num_Params
        )
    
    # =====================================================================
    # Everything below is your main workflow, with minimal modifications
    # =====================================================================
    
    # Let's assume `df`, `transform_time_series_to_domain`, `CauchyNet1`, `SIREN`, etc. are defined or imported
    # so that the snippet runs end-to-end.
    # For example:
    # from my_timeseries_stuff import transform_time_series_to_domain
    # from my_models import CauchyNet1, SIREN, NBeats, FNN_ReLU, MinimalTransformer, RBFNetwork
    
    
    # =============== Example DataFrame + transform (placeholder) ===============
    # df = ...
    # def transform_time_series_to_domain(yvals): return X_mapped, Y_mapped
    # (In your real code, these are presumably defined.)
    
    # =============== 1) Get Y from df, transform to domain ===============
    # Suppose df has a column 'Y'
    # Y_vals = df['Y']
    # X_mapped, Y_mapped = transform_time_series_to_domain(Y_vals)
    
    # Just ensuring these exist in the snippet:
    Y_vals = df['Y']  
    X_mapped, Y_mapped = transform_time_series_to_domain(Y_vals)
    
    # =============== 2) Create Torch Tensors ===============
    X = torch.tensor(X_mapped, dtype=torch.float32)
    Y = torch.tensor(Y_mapped, dtype=torch.float32)
    
    # =============== 3) Scale Y ===============
    scaler = MinMaxScaler()
    Y_np   = Y.numpy().reshape(-1, 1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
    # =============== 4) Data Splits ===============
    # If `loadData` is your own function, you can just call it:
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    # =============== 5) Model Dictionary ===============
    models_dict = {
        "CauchyNet":   CauchyNet1,
        "SIREN":       SIREN,
        "NBeats":      NBeats,
        "FNN":         FNN_ReLU,
        "Transformer": MinimalTransformer,
        "RBF":         RBFNetwork
        # "Informer":  MinimalInformer,  # if you have this model
    }
    
    # =============== 6) Create Color Map for Models ===============
    model_names = list(models_dict.keys())
    color_map   = get_model_colors(model_names, emphasize_model="CauchyNet")
    
    # =============== 7) Train Configuration ===============
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size   = 1
    hidden_size  = 128
    output_size  = 1
    lr           = 0.01
    epochs       = 200
    num_repeats  = 10
    
    # =============== 8) Train Multiple Runs & Collect Logs ===============
    interval_logs  = {}
    results_dict   = {}
    
    for model_name, constructor in models_dict.items():
        print(f"=== {model_name} ===")
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        Num_Params = None
    
        for _ in range(num_repeats):
            (
                model,
                train_losses,
                val_losses,
                test_mse,
                test_mae,
                preds_unscaled,
                truths_unscaled,
                training_time,
                Num_Params
            ) = train_and_evaluate_model(
                constructor,
                input_size, hidden_size, output_size,
                train_loader, val_loader, test_loader,
                lr=lr, epochs=epochs, device=device,
                scaler=scaler   # <-- PASS THE FITTED SCALER HERE
            )
    
            runs_mse.append(test_mse)
            runs_mae.append(test_mae)
            runs_time.append(training_time)
            train_runs.append(train_losses)
            val_runs.append(val_losses)
    
        # Mean ± Std
        mse_mean, mse_std   = np.mean(runs_mse),  np.std(runs_mse)
        mae_mean, mae_std   = np.mean(runs_mae),  np.std(runs_mae)
        time_mean, time_std = np.mean(runs_time), np.std(runs_time)
    
        interval_logs[model_name] = {
            'train': train_runs,
            'val':   val_runs
        }
        results_dict[model_name] = {
            'MSE_mean':    mse_mean,
            'MSE_std':     mse_std,
            'MAE_mean':    mae_mean,
            'MAE_std':     mae_std,
            'Time_mean':   time_mean,
            'Time_std':    time_std,
            'Num_Params':  Num_Params
        }
    
    # =============== 9) Plot Training Curves ===============
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    plot_training_curves_with_confidence(
        interval_logs,
        model_names,
        color_map,
        filename="training_curves2.pdf"
    )
    
    # =============== 10) Final Run for Each Model ===============
    final_models = {}
    for model_name, constructor in models_dict.items():
        model, *_ = train_and_evaluate_model(
            constructor,
            input_size, hidden_size, output_size,
            train_loader, val_loader, test_loader,
            lr=lr, epochs=epochs, device=device,
            scaler=scaler  # pass the scaler again
        )
        final_models[model_name] = model
    
    # ------------------------
    # Build a sample X_plot
    # ------------------------
    # Using the last 100 points for example, or anything you want:
    X_plot_torch      = X[:-60].to(device)         # e.g. shape (100,) or (100,1)
    Y_plot_true_torch = Y[:-60].to(device)         # unscaled ground truth
    # Re-fit the same scaler on original Y_np
    scaler.fit(Y_np)
    
    # ------------------------
    # Inference for overlay
    # ------------------------
    overlay_preds = {}
    for model_name, model in final_models.items():
        model.eval()
        with torch.no_grad():
            out = model(X_plot_torch.unsqueeze(-1).float())
            if isinstance(out, tuple):
                y_real, _ = out
                preds_scaled = y_real.cpu().numpy().flatten()
            else:
                preds_scaled = out.cpu().numpy().flatten()
    
        # Convert scaled predictions back to original
        preds_2d       = preds_scaled.reshape(-1, 1)
        preds_unscaled = scaler.inverse_transform(preds_2d).flatten()
        overlay_preds[model_name] = preds_unscaled
    
    # ------------------------
    # For plotting
    # ------------------------
    X_plot_np      = X_plot_torch.cpu().numpy()
    Y_plot_true_np = Y_plot_true_torch.cpu().numpy()
    
    # =============== 11) Subplot Predictions ===============
    subplot_predictions(
        X_plot_np,
        Y_plot_true_np,
        overlay_preds,
        color_map,
        filename="subplot_predictions2.pdf"
    )
    
    # =============== 12) Box Plot of Errors (Unscaled) ===============
    # If you want the box plot also in original scale, do inverse_transform below
    error_dict = {}
    for mname, model in final_models.items():
        model.eval()
        abs_errors = []
        with torch.no_grad():
            for x_batch, y_batch_scaled in test_loader:
                x_batch = x_batch.to(device)
                y_batch_scaled = y_batch_scaled.to(device)
                out = model(x_batch)
    
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_scaled = out.cpu().numpy().flatten()
    
                truths_scaled = y_batch_scaled.cpu().numpy().flatten()
    
                # Inverse-transform both to get unscaled domain
                preds_unscaled  = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                truths_unscaled = scaler.inverse_transform(truths_scaled.reshape(-1, 1)).flatten()
    
                abs_errors.extend(np.abs(preds_unscaled - truths_unscaled))
    
        error_dict[mname] = np.array(abs_errors)
    
    plot_box_errors(error_dict, color_map, filename="box_errors2.pdf")
    
    # =============== 13) Print Summary ===============
    print("\n=== Summary of Results ===")
    for mname, stats in results_dict.items():
        print(f"{mname:12s} => "
              f"MSE={stats['MSE_mean']:.6f}±{stats['MSE_std']:.4f}, "
              f"MAE={stats['MAE_mean']:.4f}±{stats['MAE_std']:.4f}, "
              f"Time={stats['Time_mean']:.2f}±{stats['Time_std']:.2f}s, "
              f"#Params={stats['Num_Params']}")
    
    # =============== 14) Generate LaTeX Table ===============
    latex_str = generate_latex_table(results_dict, filename="results_summary.tex")
    print("\nLaTeX Table:\n")
    print(latex_str)
    print("\nSaved to 'results_summary2.tex'")
    
    # =============== 15) Plot Error Curves ===============
    plot_error_curves(
        X_plot_np,
        Y_plot_true_np,
        overlay_preds,
        color_map,
        filename="error_curves2.pdf"
    )
    
    print("\n=== Done! ===")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 22 FAILED, continuing', flush=True)

_cell[0]=23
print('=== CELL 23 ===', flush=True)
try:
    # =============== 11) Final Run for Each Model ===============
    final_models = {}
    for model_name, constructor in models_dict.items():
        model, *_ = train_and_evaluate_model(
            constructor,
            input_size, hidden_size, output_size,
            train_loader, val_loader, test_loader,
            lr=lr, epochs=epochs, device=device
        )
        final_models[model_name] = model
    
    # =============== 12) Gather Full Test Set in Tensors ===============
    X_test_all, Y_test_all = [], []
    
    for x_batch, y_batch in test_loader:
        X_test_all.append(x_batch)
        Y_test_all.append(y_batch)
    
    X_test_all = torch.cat(X_test_all, dim=0).to(device)  # (N_test, input_dim)
    Y_test_all = torch.cat(Y_test_all, dim=0).to(device)  # (N_test, output_dim)
    
    import numpy as np
    
    predictions_dict = {}
    
    with torch.no_grad():
        for model_name, model in final_models.items():
            model.eval()
    
            preds_list = []
    
            # We'll loop through each sample (in order)
            for i in range(len(X_test_all)):
                # 1) Current input is shape (1, input_dim)
                x_curr = X_test_all[i].unsqueeze(0)  # add batch dim
    
                # 2) Forward pass
                out = model(x_curr)
    
                # If your model returns multiple items (e.g., (y_pred, ...)),
                # unpack the first element:
                if isinstance(out, tuple):
                    out = out[0]  # or y_pred, _ = out
    
                # 3) Convert to NumPy and store
                preds_list.append(out.cpu().numpy().squeeze())
    
            # Combine all predictions into one NumPy array of shape (N_test, ...)
            preds_array = np.array(preds_list)
            predictions_dict[model_name] = preds_array
    
    Y_test_np = Y_test_all.cpu().numpy()
    
    for model_name, preds in predictions_dict.items():
        plt.figure()
        plt.plot(Y_test_np[:10], label='Ground Truth', color='black')
        plt.plot(preds[:10],      label=model_name,    color='red')
        plt.title(f"Test Predictions (Step-by-Step): {model_name}")
        plt.legend()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 23 FAILED, continuing', flush=True)

_cell[0]=24
print('=== CELL 24 ===', flush=True)
try:
    Y_test_np = Y_test_all.cpu().numpy()
    
    for model_name, preds in predictions_dict.items():
        plt.figure()
        plt.plot(Y_test_np[:10], label='Ground Truth', color='black')
        plt.plot(preds[:10],      label=model_name,    color='red')
        plt.title(f"Test Predictions (Step-by-Step): {model_name}")
        plt.legend()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 24 FAILED, continuing', flush=True)

_cell[0]=25
print('=== CELL 25 ===', flush=True)
try:
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves1.pdf")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 25 FAILED, continuing', flush=True)

_cell[0]=26
print('=== CELL 26 ===', flush=True)
try:
    import numpy as np
    
    def transform_time_series_to_domain(Y, x_start=-1.0, x_end=0.9):
        """
        Map a time series Y_0, Y_1, ..., Y_n to a set of points
        (x_k, Y_k) on [x_start, x_end].
        
        Parameters:
        -----------
        Y : array-like
            The time series values of shape (n+1,).
        x_start : float
            Left boundary of the interval for x.
        x_end : float
            Right boundary of the interval for x.
        
        Returns:
        --------
        X : numpy.ndarray
            1D array of shape (n+1,) containing the mapped x-coordinates.
        Y : numpy.ndarray
            Same as input Y but converted to numpy array for consistency.
        """
        Y = np.asarray(Y)
        n = len(Y) - 1  # because we have Y_0,...,Y_n (total n+1 points)
        if n < 1:
            raise ValueError("Time series must contain at least two points.")
        
        # Compute step size
        length = x_end - x_start
        h = length / n
    
        # Create the domain array
        X = np.array([x_start + k*h for k in range(n+1)])
        
        return X, Y
    
    
    
    Y_vals = df['Y']
        
    X_mapped, Y_mapped = transform_time_series_to_domain(Y_vals)
        
    
    
    import torch
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from utils2 import (
        loadData,
        train_and_evaluate_model,
        plot_training_curves_with_confidence,
        generate_latex_table,
        get_model_colors,
        plot_box_errors,
        scatter_predictions_vs_truth,
        subplot_predictions,
        plot_error_curves
    )
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    # etc.
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) Generate Data
    #data_size = 3000
    #X = torch.linspace(0, 1, data_size)
    # # Replace with your actual function, e.g. "challenging_function"
    # # Make sure it's defined or in scope
    #Y = challenging_function(X) 
    
    X = torch.tensor(X_mapped, dtype=torch.float32)
    Y = torch.tensor(Y_mapped, dtype=torch.float32)
    
    # Scale Y
    scaler = MinMaxScaler()
    Y_np   = Y.numpy().reshape(-1, 1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
    from torch.utils.data import DataLoader, TensorDataset, Subset
    import torch
    
    def loadData(X, Y, batchSize=64):
        """
        Splits dataset into:
        - Train/Validation: [0, 0.8]
        - Test: [0.8, 1)
        Returns train_loader, val_loader, test_loader, and ds_test.
        """
        if X.dim() == 1:
            X = X.unsqueeze(-1)
    
        # Define training/validation and test split points
        split_point = 0.8
        split_idx   = int(len(X) * split_point)
    
        # Indices for splits
        train_val_indices = range(split_idx)  # Indices [0, 0.8)
        test_indices      = range(split_idx, len(X))  # Indices [0.8, 1)
    
        # Create training/validation and test datasets
        dataset = TensorDataset(X, Y)
        ds_train_val = Subset(dataset, train_val_indices)
        ds_test      = Subset(dataset, test_indices)
    
        # Further split train/val into 80/20 within [0, 0.8]
        train_size = int(0.8 * len(ds_train_val))  # 80% of train/val
        val_size   = len(ds_train_val) - train_size
    
        ds_train, ds_val = torch.utils.data.random_split(ds_train_val, [train_size, val_size])
    
        # Create DataLoaders
        train_loader = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        val_loader   = DataLoader(ds_val,   batch_size=batchSize, shuffle=False)
        test_loader  = DataLoader(ds_test,  batch_size=batchSize, shuffle=False)
    
        return train_loader, val_loader, test_loader, ds_test
    # Build dataset + splits
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    # Example model dictionary (assuming you have each class defined):
    models_dict = {
        #"CauchyNet":   CauchyNet,
        "CauchyNet":   CauchyNet1,
        "SIREN":       SIREN,
            "NBeats": NBeats,
            "FNN":    FNN_ReLU,
                "Transformer": MinimalTransformer,
        "RBF":     RBFNetwork
       # "Informer":    MinimalInformer,
    
    }
    
    # 2) Create a color map for consistent plotting across all figures
    model_names = list(models_dict.keys())
    color_map   = get_model_colors(model_names, emphasize_model="CauchyNet")
    
    # 3) Training configuration
    input_size  = 1
    hidden_size = 128
    output_size = 1
    lr          = 0.005
    epochs      = 200
    num_repeats = 1
    
    # 4) Train multiple runs & collect logs
    interval_logs  = {}
    results_dict   = {}
    
    for model_name, constructor in models_dict.items():
        print(f"=== {model_name} ===")
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs         = [], []
        Num_Params = None
    
        for _ in range(num_repeats):
            (model, 
             train_losses, 
             val_losses,
             test_mse, 
             test_mae, 
             preds_scaled,
             truths_scaled,
             training_time,
             Num_Params) = train_and_evaluate_model(
                constructor,
                input_size, hidden_size, output_size,
                train_loader, val_loader, test_loader,
                lr=lr, epochs=epochs, device=device
            )
    
            runs_mse.append(test_mse)
            runs_mae.append(test_mae)
            runs_time.append(training_time)
            train_runs.append(train_losses)
            val_runs.append(val_losses)
    
        # Mean ± Std
        mse_mean, mse_std  = np.mean(runs_mse),  np.std(runs_mse)
        mae_mean, mae_std  = np.mean(runs_mae),  np.std(runs_mae)
        time_mean, time_std= np.mean(runs_time), np.std(runs_time)
    
        interval_logs[model_name] = {
            'train': train_runs,
            'val':   val_runs
        }
        results_dict[model_name] = {
            'MSE_mean':    mse_mean,
            'MSE_std':     mse_std,
            'MAE_mean':    mae_mean,
            'MAE_std':     mae_std,
            'Time_mean':   time_mean,
            'Time_std':    time_std,
            'Num_Params':  Num_Params
        }
    
    # 5) Plot training curves (with confidence intervals)
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves1.pdf")
    
    # 6) Final run for each model (to produce final predictions)
    final_models = {}
    for model_name, constructor in models_dict.items():
        model, *_ = train_and_evaluate_model(
            constructor,
            input_size, hidden_size, output_size,
            train_loader, val_loader, test_loader,
            lr=lr, epochs=epochs, device=device
        )
        final_models[model_name] = model
    
    # 7) Evaluate predictions on a uniform grid for final overlay
    X_plot      = X[-100:].to(device)
    Y_plot_true = Y[-100:]   # unscaled
    overlay_preds = {}
    
    # Re-fit the scaler on the original Y for consistent inverses
    scaler.fit(Y_np)
    
    for model_name, model in final_models.items():
        model.eval()
        with torch.no_grad():
            out = model(X_plot.unsqueeze(-1).float())
            if isinstance(out, tuple):
                y_real, _ = out
                preds_scaled = y_real.cpu().numpy().flatten()
            else:
                preds_scaled = out.cpu().numpy().flatten()
        preds_2d = preds_scaled.reshape(-1,1)
        preds_unscaled = scaler.inverse_transform(preds_2d).flatten()
        overlay_preds[model_name] = preds_unscaled
    
    # 7a) Subplot predictions
    subplot_predictions(X_plot, Y_plot_true, overlay_preds, color_map, filename="subplot_predictions1.pdf")
    
    # 8) Box plot of errors on test set
    error_dict = {}
    for mname, model in final_models.items():
        model.eval()
        abs_errors = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds = y_real.flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths = y_batch.cpu().numpy().flatten()
                abs_errors.extend(np.abs(preds - truths))
        error_dict[mname] = np.array(abs_errors)
    
    plot_box_errors(error_dict, color_map, filename="box_errors1.pdf")
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def scatter_predictions_vs_truth_improved(
        y_true,
        preds_dict,
        color_map,
        marker_size=30,
        alpha_val=0.4,
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
        
    # 10) Print Summary
    print("\n=== Summary of 5-Run Results ===")
    for mname, stats in results_dict.items():
        print(f"{mname:12s} => "
              f"MSE={stats['MSE_mean']:.6f}±{stats['MSE_std']:.4f}, "
              f"MAE={stats['MAE_mean']:.4f}±{stats['MAE_std']:.4f}, "
              f"Time={stats['Time_mean']:.2f}±{stats['Time_std']:.2f}s, "
              f"#Params={stats['Num_Params']}")
    
    # 11) Save LaTeX table
    latex_str = generate_latex_table(results_dict, filename="results_summary.tex")
    print("\nLaTeX Table:\n")
    print(latex_str)
    print("\nSaved to 'results_summary.tex'")
    
    # 12) Plot error curves
    plot_error_curves(X_plot, Y_plot_true, overlay_preds, color_map, filename="error_curves1.pdf")
    
    print("\n=== Done! ===")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 26 FAILED, continuing', flush=True)

_cell[0]=27
print('=== CELL 27 ===', flush=True)
try:
    # 5) Plot training curves (with confidence intervals)
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves1.pdf")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 27 FAILED, continuing', flush=True)

_cell[0]=28
print('=== CELL 28 ===', flush=True)
try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from utils2 import (
        train_and_evaluate_model,
        plot_training_curves_with_confidence,
        generate_latex_table,
        plot_box_errors,
        scatter_predictions_vs_truth,
        subplot_predictions,
        plot_error_curves
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) Generate Data
    data_size = 1000
    X = torch.linspace(0, 1, data_size).to(device)
    Y = challenging_function(X.cpu())  # Replace with your actual challenging function
    scaler = MinMaxScaler()
    
    # Normalize Y
    Y_np = Y.numpy().reshape(-1, 1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32).to(device)
    
    # 2) Split Data into Train, Validation, Test
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    # 3) Define Model Dictionary
    models_dict = {
        "CauchyNet":   CauchyNet1,  # Replace with your CauchyNet class
        "SIREN":       SIREN,       # Replace with SIREN class
        "NBeats":      NBeats,      # Replace with NBeats class
        "FNN":         FNN_ReLU,    # Replace with FNN class
        "Transformer": MinimalTransformer,  # Replace with Transformer class
        "RBF":         RBFNetwork   # Replace with RBF class
    }
    
    # 4) Training Configuration
    input_size = 1
    hidden_size = 500
    output_size = 1
    lr = 0.005
    epochs = 200
    num_repeats = 1
    
    # 5) Train Multiple Runs
    interval_logs = {}
    results_dict = {}
    final_models = {}
    
    for model_name, constructor in models_dict.items():
        print(f"=== Training {model_name} ===")
        runs_mse, runs_mae, runs_time = [], [], []
    
        for _ in range(num_repeats):
            model, train_losses, val_losses, test_mse, test_mae, *_ = train_and_evaluate_model(
                constructor, input_size, hidden_size, output_size,
                train_loader, val_loader, test_loader, lr=lr, epochs=epochs, device=device
            )
    
            runs_mse.append(test_mse)
            runs_mae.append(test_mae)
            final_models[model_name] = model
    
        # Log results
        results_dict[model_name] = {
            'MSE_mean': np.mean(runs_mse),
            'MSE_std': np.std(runs_mse),
            'MAE_mean': np.mean(runs_mae),
            'MAE_std': np.std(runs_mae),
        }
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 28 FAILED, continuing', flush=True)

_cell[0]=29
print('=== CELL 29 ===', flush=True)
try:
    
    # 6) Evaluate Predictions on Test Set
    X_plot = torch.linspace(0, 1, 200).to(device)
    # Compute the true Y values on the plotting grid
    Y_plot_true = scaler.inverse_transform(
        challenging_function(X_plot.cpu()).numpy().reshape(-1, 1)
    ).flatten()
    overlay_preds = {}
    for model_name, model in final_models.items():
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_plot.unsqueeze(-1))
            if isinstance(preds_scaled, tuple):  # Handle tuple output (e.g., CauchyNet)
                preds_scaled = preds_scaled[0]
            preds_np = preds_scaled.cpu().numpy().flatten()
            preds_unscaled = scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
            overlay_preds[model_name] = preds_unscaled
    
    # Ensure all inputs to the plotting function are properly scaled
    subplot_predictions(
        X_plot.cpu().numpy(), 
        Y_plot_true, 
        overlay_preds, 
        color_map=color_map, 
        filename="subplot_predictions1.pdf"
    )
    # 7) Plot Predictions
    # 7) Plot Predictions
    subplot_predictions(
        X_plot.cpu().numpy(), 
        Y_plot_true, 
        overlay_preds, 
        color_map=color_map,  # Ensure color_map is passed
        filename="subplot_predictions1.pdf"
    )
    
    # Plot Error Curves
    plot_error_curves(
        X_plot.cpu().numpy(), 
        Y_plot_true, 
        overlay_preds, 
        color_map=color_map,  # Use the same color_map
        filename="error_curves1.pdf"
    )
    # 8) Plot Box Plot of Absolute Errors
    error_dict = {}
    for model_name, model in final_models.items():
        abs_errors = []
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds_scaled = model(x_batch)
                if isinstance(preds_scaled, tuple):  # Handle tuple output
                    preds_scaled = preds_scaled[0]
                preds_np = preds_scaled.cpu().numpy().flatten()
                y_np = scaler.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1)).flatten()
                abs_errors.extend(np.abs(preds_np - y_np))
        error_dict[model_name] = np.array(abs_errors)
    
    plot_box_errors(error_dict, color_map=color_map,filename="box_errors1.pdf")
    
    # 9) Scatter Plot Predictions vs. Truth
    scatter_predictions_vs_truth(Y_plot_true, overlay_preds, color_map=color_map,filename="scatter_predictions1.pdf")
    
    # 10) Print Summary
    print("\n=== Summary of Results ===")
    for model_name, metrics in results_dict.items():
        print(f"{model_name}: MSE={metrics['MSE_mean']:.6f}±{metrics['MSE_std']:.6f}, "
              f"MAE={metrics['MAE_mean']:.6f}±{metrics['MAE_std']:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 29 FAILED, continuing', flush=True)

_cell[0]=30
print('=== CELL 30 ===', flush=True)
try:
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    
    def evaluate_and_plot(model, test_loader, scaler=None, device="cpu"):
        """
        Evaluates the model on the test set and generates a plot of predictions vs. true values.
    
        Parameters:
        - model (torch.nn.Module): Trained model.
        - test_loader (DataLoader): DataLoader for the test set.
        - scaler (MinMaxScaler, optional): Scaler for inverse-transforming predictions (if normalization was used).
        - device (str): Device for computations ("cpu" or "cuda").
    
        Returns:
        - predictions (numpy array): Model predictions on the test set.
        - truths (numpy array): Ground truth values from the test set.
        """
        model.eval()
        predictions, truths = [], []
    
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                predictions.append(y_pred.cpu())
                truths.append(y_batch.cpu())
    
        # Concatenate predictions and truths
        predictions = torch.cat(predictions, dim=0).numpy()
        truths = torch.cat(truths, dim=0).numpy()
    
        # If a scaler is provided, inverse-transform the predictions and truths
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
            truths = scaler.inverse_transform(truths)
    
        # Plot predictions vs. truths
        plt.figure(figsize=(10, 6))
        plt.plot(truths, label="True Values", color="blue", linewidth=2)
        plt.plot(predictions, label="Predictions", color="orange", linestyle="--", linewidth=2)
        plt.xlabel("Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title("True Values vs. Predictions on Test Set", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        return predictions, truths
    
    # Example: Load a test set and model
    # Ensure you have trained your model and prepared test_loader
    
    # Example DataLoader
    test_loader = DataLoader(ds_test, batch_size=64, shuffle=False)
    
    
    
    from sklearn.preprocessing import MinMaxScaler
    
    # Example scaler used during data preprocessing
    scaler = MinMaxScaler()
    scaler.fit(Y_train.numpy().reshape(-1, 1))  # Fit on training data
    
    # Evaluate and plot
    predictions, truths = evaluate_and_plot(
        model=trained_model,
        test_loader=test_loader,
        scaler=scaler,   # Use the same scaler
        device=device
    )
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 30 FAILED, continuing', flush=True)

_cell[0]=31
print('=== CELL 31 ===', flush=True)
try:
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    ##############################################################################
    # 1) Global Settings & Style
    ##############################################################################
    matplotlib.rcParams["pdf.fonttype"]  = 42
    matplotlib.rcParams["ps.fonttype"]   = 42
    matplotlib.rcParams["font.family"]   = "sans-serif"
    matplotlib.rcParams["figure.dpi"]    = 120
    
    sns.set_style("whitegrid")            # for a clean, journal-like style
    sns.set_palette("colorblind")         # colorblind-friendly palette
    
    SEED = 10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ##############################################################################
    # 2) Minimal Transformer & Informer with batch_first=True
    ##############################################################################
    class MinimalTransformer(nn.Module):
        """
        A minimal single-step Transformer for function approximation (scalar input->output).
        """
        def __init__(self, input_size, hidden_size, output_size, nhead=4, num_layers=1):
            super().__init__()
            self.embedding = nn.Linear(input_size, hidden_size)
            # batch_first=True to avoid the nested_tensor warning
            encoder_layer  = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
            self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc_out    = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # x: (batch, input_dim=1)
            # We'll treat each sample as a "sequence of length=1"
            emb = self.embedding(x)                          # (batch, hidden_size)
            emb = emb.unsqueeze(1)                           # (batch, seq=1, hidden_size)
            enc = self.encoder(emb)                          # (batch, seq=1, hidden_size)
            enc = enc.squeeze(1)                             # (batch, hidden_size)
            out = self.fc_out(enc)                           # (batch, output_size)
            return out
    
    class MinimalInformer(nn.Module):
        """
        A toy 'Informer-like' model for demonstration with batch_first=True.
        """
        def __init__(self, input_size, hidden_size, output_size, nhead=4, num_layers=1):
            super().__init__()
            self.embedding = nn.Linear(input_size, hidden_size)
            self.gate      = nn.Linear(hidden_size, hidden_size)
            encoder_layer  = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
            self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc_out    = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            emb  = self.embedding(x)             # (batch, hidden_size)
            gate = torch.sigmoid(self.gate(emb)) * emb
            gate = gate.unsqueeze(1)             # (batch, seq=1, hidden_size)
            enc  = self.encoder(gate)            # (batch, seq=1, hidden_size)
            enc  = enc.squeeze(1)                # (batch, hidden_size)
            out  = self.fc_out(enc)              # (batch, output_size)
            return out
    
    ##############################################################################
    # 3) Other Networks (CauchyNet, SIREN, RBF, MLP, etc.)
    ##############################################################################
    class ReciprocalActivation(nn.Module):
        """Reciprocal activation: out = 1 / (x + epsilon)."""
        def __init__(self, epsilon=1e-10):
            super().__init__()
            self.epsilon = epsilon
        def forward(self, x):
            return 1.0 / (x + self.epsilon)
    
    class CauchyNet(nn.Module):
        """CauchyNet returning (y_real, y_imag)."""
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1.0, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=1.0, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            batch_size, _ = x.size()
            x_c = torch.complex(x, torch.zeros_like(x))
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)
            x_expanded  = x_c.expand(batch_size, self.hidden_size)
            activated   = self.activation(xi_expanded - x_expanded)
            y_complex   = torch.matmul(activated, self.lambda_) / self.hidden_size
            return y_complex.real, y_complex.imag
    
    class SIREN(nn.Module):
        """Minimal SIREN (single hidden layer)."""
        def __init__(self, input_size, hidden_size, output_size, w0_initial=30.0, w0_hidden=30.0):
            super().__init__()
            self.linear_in  = nn.Linear(input_size, hidden_size)
            self.linear_out = nn.Linear(hidden_size, output_size)
            self.w0_initial = w0_initial
            self.w0_hidden  = w0_hidden
            self._init_weights()
    
        def _init_weights(self):
            with torch.no_grad():
                nn.init.uniform_(
                    self.linear_in.weight, 
                    -1.0 / self.linear_in.in_features,
                     1.0 / self.linear_in.in_features
                )
                nn.init.zeros_(self.linear_in.bias)
                self.linear_in.weight *= self.w0_initial
    
                in_features = self.linear_out.in_features
                bound = (6.0 / in_features) ** 0.5
                nn.init.uniform_(self.linear_out.weight, -bound / self.w0_hidden, bound / self.w0_hidden)
                nn.init.zeros_(self.linear_out.bias)
    
        def sine(self, x, w0):
            return torch.sin(w0 * x)
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h   = self.sine(self.linear_in(x), w0=1.0)
            out = self.linear_out(h)
            return out
    
    class RBFNetwork(nn.Module):
        """Minimal RBF network with trainable centers/widths + linear readout."""
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.centers = nn.Parameter(torch.randn(hidden_size, input_size))
            self.log_sigmas = nn.Parameter(torch.zeros(hidden_size))
            self.linear = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            x_expanded       = x.unsqueeze(1)
            centers_expanded = self.centers.unsqueeze(0)
            dist_sq          = (x_expanded - centers_expanded).pow(2).sum(dim=2)
            sigma            = torch.exp(self.log_sigmas).unsqueeze(0)
            rbf_activations  = torch.exp(-dist_sq / (2 * sigma.pow(2)))
            out = self.linear(rbf_activations)
            return out
    
    class FNN_ReLU(nn.Module):
        """Basic MLP with ReLU."""
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.ReLU()
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    ##############################################################################
    # 4) Dataset & Splits
    ##############################################################################
    # def challenging_function(x):
    #     """A 'complicated' function: sin + cos + polynomial term."""
    #     return torch.sin(2*np.pi*x*3) + 0.3*torch.cos(2*np.pi*x*7) + 0.5*(x**2)
    
    def loadData(X, Y, batchSize=64):
        """Splits into 50%/25%/25% for train/val/test."""
        if X.dim() == 1:
            X = X.unsqueeze(-1)
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
    # 5) Training Function (Single Run)
    ##############################################################################
    def train_and_evaluate_model(
        model_constructor, input_size, hidden_size, output_size,
        train_loader, val_loader, test_loader,
        lr=0.001, epochs=200
    ):
        """
        Trains a single model and returns:
          - model
          - train_losses, val_losses
          - final test_mse, test_mae
          - predictions & truths on test
          - training_time
          - num_params
        """
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses = []
        val_losses   = []
    
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        # Time the training
        start_time = time.time()
    
        best_val_loss = float('inf')
        best_state    = None
    
        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)
    
                # For CauchyNet => (real, imag)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    y_b_exp = y_batch.unsqueeze(-1)
                    loss    = criterion(y_real, y_b_exp) + 10*criterion(y_imag, torch.zeros_like(y_imag))
                else:
                    y_b_exp = y_batch.unsqueeze(-1)
                    loss    = criterion(out, y_b_exp)
    
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
    
            train_loss = train_loss_accum / len(train_loader)
            train_losses.append(train_loss)
    
            # Validation
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        y_b_exp = y_batch.unsqueeze(-1)
                        vloss   = criterion(y_real, y_b_exp) + criterion(y_imag, torch.zeros_like(y_imag))
                    else:
                        y_b_exp = y_batch.unsqueeze(-1)
                        vloss   = criterion(out, y_b_exp)
                    val_loss_accum += vloss.item()
    
            val_loss = val_loss_accum / len(val_loader)
            val_losses.append(val_loss)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
    
            scheduler.step(val_loss)
    
        # End time
        end_time = time.time()
        training_time = end_time - start_time
    
        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test
        model.eval()
        preds_list, truths_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    preds = y_real.cpu().numpy().flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths = y_batch.cpu().numpy().flatten()
                preds_list.append(preds)
                truths_list.append(truths)
    
        preds_all  = np.concatenate(preds_list)
        truths_all = np.concatenate(truths_list)
        test_mse   = mean_squared_error(truths_all, preds_all)
        test_mae   = mean_absolute_error(truths_all, preds_all)
    
        return (
            model, 
            train_losses, 
            val_losses, 
            test_mse, 
            test_mae, 
            preds_all, 
            truths_all, 
            training_time, 
            num_params
        )
    
    ##############################################################################
    # 6) Plotting: Training Curves (with Confidence Interval)
    ##############################################################################
    def plot_training_curves_with_confidence(interval_logs, model_names):
        """
        interval_logs: { model_name : { 'train': [runs], 'val': [runs] } }
        where each 'runs' is a list of length num_repeats, and each item is
        a list of losses over epochs.
        """
        plt.figure(figsize=(8,6))
        colors      = sns.color_palette("husl", len(model_names))
        epochs_range= None
    
        for i, model_name in enumerate(model_names):
            train_runs = interval_logs[model_name]['train']
            val_runs   = interval_logs[model_name]['val']
    
            # Align them by min length
            min_len     = min(len(run) for run in train_runs)
            train_array = np.array([run[:min_len] for run in train_runs])
            val_array   = np.array([run[:min_len] for run in val_runs])
    
            mean_train = train_array.mean(axis=0)
            std_train  = train_array.std(axis=0)
            mean_val   = val_array.mean(axis=0)
            std_val    = val_array.std(axis=0)
    
            if epochs_range is None:
                epochs_range = np.arange(1, min_len+1)
    
            c = colors[i]
            # Plot train
            plt.plot(epochs_range, mean_train, label=f"{model_name} - Train", color=c, linewidth=2)
            plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train, color=c, alpha=0.2)
            # Plot val
            plt.plot(epochs_range, mean_val, '--', label=f"{model_name} - Val", color=c, linewidth=2)
            plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val, color=c, alpha=0.2)
    
        plt.yscale("log")
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("MSE Loss (Log Scale)", fontsize=14)
        plt.title("Training & Validation (Mean ± Std) over 5 Runs", fontsize=15, fontweight='bold')
        plt.legend(fontsize=10, loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    ##############################################################################
    # 7) Box Plot of Errors
    ##############################################################################
    def plot_box_errors(error_dict):
        """
        error_dict = { model_name: array_of_abs_errors, ... }
        We show a standard box plot (no violin).
        """
        model_names = list(error_dict.keys())
        errors_data = [error_dict[m] for m in model_names]
    
        plt.figure(figsize=(8,5))
        # Use a bright palette
        sns.set_palette("bright")
        box = plt.boxplot(errors_data, labels=model_names, patch_artist=True)
    
        # Color the boxes (optional)
        for patch, color in zip(box['boxes'], sns.color_palette("bright", len(model_names))):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
    
        plt.title("Distribution of Absolute Errors (Test Set)", fontsize=14, fontweight='bold')
        plt.ylabel("Absolute Error", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    ##############################################################################
    # 8) Scatter Plot: Predictions vs. Truth
    ##############################################################################
    def scatter_predictions_vs_truth(truths, preds_dict):
        """
        Plot each model's predictions vs. truth on one scatter plot.
        A perfect predictor => points on diagonal y=x.
        """
        plt.figure(figsize=(6,6))
        sns.set_palette("bright")
        for model_name, pred in preds_dict.items():
            plt.scatter(pred, truths, s=40, alpha=0.5, label=model_name)
    
        min_val = min(truths.min(), *(p.min() for p in preds_dict.values()))
        max_val = max(truths.max(), *(p.max() for p in preds_dict.values()))
        rng     = max_val - min_val
        plt.plot([min_val - 0.1*rng, max_val + 0.1*rng],
                 [min_val - 0.1*rng, max_val + 0.1*rng],
                 'k--', linewidth=2)
    
        plt.xlabel("Predicted", fontsize=13)
        plt.ylabel("True", fontsize=13)
        plt.title("Predictions vs. True (Test Set)", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    ##############################################################################
    # 9) Main: Putting It All Together
    ##############################################################################
    if __name__ == "__main__":
        # 1) Generate Data
        data_size = 300
        X = torch.linspace(0, 1, data_size)
        Y = challenging_function(X)
    
        # Scale Y
        scaler = MinMaxScaler()
        Y_np   = Y.numpy().reshape(-1, 1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
        # Split data
        train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
        # 2) Define model constructors
        def cauchy_constructor(i, h, o):   return CauchyNet(i, h, o)
        def siren_constructor(i, h, o):    return SIREN(i, h, o)
        def rbf_constructor(i, h, o):      return RBFNetwork(i, h, o)
        def fnn_relu(i, h, o):            return FNN_ReLU(i, h, o)
        def transf_constructor(i, h, o):  return MinimalTransformer(i, h, o, nhead=4, num_layers=1)
        def informer_constructor(i, h, o):return MinimalInformer(i, h, o, nhead=4, num_layers=1)
    
        models_dict = {
            "CauchyNet":   cauchy_constructor,
            "SIREN":       siren_constructor,
            "RBF_Net":     rbf_constructor,
            "FNN_ReLU":    fnn_relu,
            "Transformer": transf_constructor,
            "Informer":    informer_constructor
        }
    
        # 3) Training config
        input_size  = 1
        hidden_size = 128
        output_size = 1
        lr          = 0.001
        epochs      = 400
        num_repeats = 5
    
        # 4) Train multiple runs & collect logs
        interval_logs  = {}
        results_dict   = {}
        for model_name, constructor in models_dict.items():
            print(f"=== {model_name} ===")
            runs_mse, runs_mae, runs_time = [], [], []
            train_runs, val_runs = [], []
            num_params = None
    
            for _ in range(num_repeats):
                (model, 
                 train_losses, 
                 val_losses,
                 test_mse, 
                 test_mae, 
                 preds_scaled,
                 truths_scaled,
                 training_time,
                 n_params) = train_and_evaluate_model(
                    constructor,
                    input_size, hidden_size, output_size,
                    train_loader, val_loader, test_loader,
                    lr=lr, epochs=epochs
                )
                runs_mse.append(test_mse)
                runs_mae.append(test_mae)
                runs_time.append(training_time)
                train_runs.append(train_losses)
                val_runs.append(val_losses)
                if num_params is None:
                    num_params = n_params
    
            # Mean ± Std
            mse_mean, mse_std = np.mean(runs_mse), np.std(runs_mse)
            mae_mean, mae_std = np.mean(runs_mae), np.std(runs_mae)
            time_mean, time_std = np.mean(runs_time), np.std(runs_time)
    
            interval_logs[model_name] = {
                'train': train_runs,
                'val':   val_runs
            }
            results_dict[model_name] = {
                'mse_mean':    mse_mean, 'mse_std':    mse_std,
                'mae_mean':    mae_mean, 'mae_std':    mae_std,
                'time_mean':   time_mean,'time_std':   time_std,
                'num_params':  num_params
            }
    
        # 5) Plot training curves with confidence intervals
        plot_training_curves_with_confidence(interval_logs, list(models_dict.keys()))
    
        # 6) Train a final “best run” model for each to produce final predictions
        final_models = {}
        for model_name, constructor in models_dict.items():
            model, _, _, _, _, _, _, _, _ = train_and_evaluate_model(
                constructor,
                input_size, hidden_size, output_size,
                train_loader, val_loader, test_loader,
                lr=lr, epochs=epochs
            )
            final_models[model_name] = model
    
        # 7) Plot each model's predictions on a uniform grid X_plot
        X_plot = torch.linspace(-1, 1, 300)
        Y_plot_true = challenging_function(X_plot).numpy()  # unscaled
    
        # Prepare overlay
        overlay_preds = {}
        for model_name, model in final_models.items():
            model.eval()
            with torch.no_grad():
                out = model(X_plot.unsqueeze(-1).float().to(device))
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    preds_plot_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_plot_scaled = out.cpu().numpy().flatten()
            # Inverse transform
            preds_2d = preds_plot_scaled.reshape(-1,1)
            preds_unscaled = scaler.inverse_transform(preds_2d).flatten()
            overlay_preds[model_name] = preds_unscaled
    
        # 7a) Overlay Plot
        plt.figure(figsize=(8,5))
        x_np = X_plot.numpy()
        sort_idx = np.argsort(x_np)
        x_sorted = x_np[sort_idx]
        y_sorted = Y_plot_true[sort_idx]
        plt.plot(x_sorted, y_sorted, color='black', linewidth=3, label="True")
        color_cycle = sns.color_palette("husl", len(models_dict))
        for i, (mname, preds) in enumerate(overlay_preds.items()):
            p_sorted = preds[sort_idx]
            plt.plot(x_sorted, p_sorted, '--', color=color_cycle[i], linewidth=2, label=mname)
        plt.title("Model Predictions vs. True Function", fontsize=15, fontweight='bold')
        plt.xlabel("x", fontsize=13)
        plt.ylabel("y", fontsize=13)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
        # 8) Box Plot of absolute errors on test set
        #    We'll just reuse the last-run errors from the loop above if needed,
        #    but let's quickly re-run final predictions on test set.
        error_dict = {}
        for mname, model in final_models.items():
            model.eval()
            abs_errors = []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, _ = out
                        preds = y_real.cpu().numpy().flatten()
                    else:
                        preds = out.cpu().numpy().flatten()
                    truths= y_batch.cpu().numpy().flatten()
                    abs_errors.extend(np.abs(preds - truths))
            error_dict[mname] = np.array(abs_errors)
    
        plot_box_errors(error_dict)
    
        # 9) Scatter Plot (Test set)
        #    Need unscaled truths
        #    We'll re-inverse the test Y-values for a fair scatter
        test_data_np = ds_test[:]
        x_test_all   = test_data_np[0]
        y_test_all   = test_data_np[1].numpy()
        # ds_test is scaled. We'll inverse transform y
        y_unscaled   = scaler.inverse_transform(y_test_all.reshape(-1,1)).flatten()
    
        # For predictions, get unscaled predictions
        scatter_preds = {}
        for mname, model in final_models.items():
            preds_list = []
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    out = model(x_batch.to(device))
                    if isinstance(out, tuple):
                        y_real, _ = out
                        preds = y_real.cpu().numpy().flatten()
                    else:
                        preds = out.cpu().numpy().flatten()
                    preds_list.append(preds)
            preds_all = np.concatenate(preds_list)
            # Inverse-scale
            preds_all_2d = preds_all.reshape(-1,1)
            preds_all_un = scaler.inverse_transform(preds_all_2d).flatten()
            scatter_preds[mname] = preds_all_un
    
        # Now plot
        scatter_predictions_vs_truth(y_unscaled, scatter_preds)
    
        # 10) Print Summary
        print("\n=== Summary of 5-Run Results ===")
        for mname, stats in results_dict.items():
            print(f"{mname:12s} => MSE={stats['mse_mean']:.4f}±{stats['mse_std']:.4f}, "
                  f"MAE={stats['mae_mean']:.4f}±{stats['mae_std']:.4f}, "
                  f"Time={stats['time_mean']:.2f}±{stats['time_std']:.2f}s, "
                  f"#Params={stats['num_params']}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 31 FAILED, continuing', flush=True)

_cell[0]=32
print('=== CELL 32 ===', flush=True)
try:
    ########################################
    # utils.py
    ########################################
    
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    
    ##############################################################################
    # 1) Data Loading
    ##############################################################################
    def loadData(X, Y, batchSize=64):
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
        train_loader = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        val_loader   = DataLoader(ds_val,   batch_size=batchSize, shuffle=False)
        test_loader  = DataLoader(ds_test,  batch_size=batchSize, shuffle=False)
        return train_loader, val_loader, test_loader, ds_test
    
    ##############################################################################
    # 2) Train & Evaluate Model
    ##############################################################################
    def train_and_evaluate_model(
        model_constructor, input_size, hidden_size, output_size,
        train_loader, val_loader, test_loader,
        lr=0.001, epochs=200, device=None
    ):
        """
        Trains a model with a simple Adam + MSE routine, returning:
          (model,
           train_losses, val_losses,
           test_mse, test_mae,
           preds_all, truths_all,
           training_time, num_params)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Construct model & define loss
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
    
        # Optimizer & scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_state    = None
    
        # Count trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        # Start timer
        start_time = time.time()
    
        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)
    
                # If model returns tuple (e.g., real & imag):
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    y_b_exp = y_batch.unsqueeze(-1)  # shape => (batch, 1)
                    loss = criterion(y_real, y_b_exp) + criterion(y_imag, torch.zeros_like(y_imag))
                else:
                    y_b_exp = y_batch.unsqueeze(-1)
                    loss    = criterion(out, y_b_exp)
    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_accum += loss.item()
    
            # Record average training loss
            train_loss = train_loss_accum / len(train_loader)
            train_losses.append(train_loss)
    
            # Validation
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        y_b_exp  = y_batch.unsqueeze(-1)
                        vloss    = criterion(y_real, y_b_exp) + criterion(y_imag, torch.zeros_like(y_imag))
                    else:
                        y_b_exp  = y_batch.unsqueeze(-1)
                        vloss    = criterion(out, y_b_exp)
                    val_loss_accum += vloss.item()
    
            val_loss = val_loss_accum / len(val_loader)
            val_losses.append(val_loss)
    
            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
    
            # Step scheduler
            scheduler.step(val_loss)
    
        # End timer
        end_time = time.time()
        training_time = end_time - start_time
    
        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test set
        model.eval()
        preds_list, truths_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds = y_real.cpu().numpy().flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths = y_batch.cpu().numpy().flatten()
                preds_list.append(preds)
                truths_list.append(truths)
    
        preds_all  = np.concatenate(preds_list)
        truths_all = np.concatenate(truths_list)
        test_mse   = mean_squared_error(truths_all, preds_all)
        test_mae   = mean_absolute_error(truths_all, preds_all)
    
        return (model,
                train_losses,
                val_losses,
                test_mse,
                test_mae,
                preds_all,
                truths_all,
                training_time,
                num_params)
    
    ##############################################################################
    # 3) Plot Training Curves With Confidence Intervals
    ##############################################################################
    def plot_training_curves_with_confidence(interval_logs, model_names):
        """
        interval_logs: { model_name: { 'train': [list_of_runs], 'val': [list_of_runs] } }
        Each run is a list of losses across epochs.
        We'll compute mean/std for each epoch index and plot the result with shading.
        """
        plt.figure(figsize=(8, 6))
        colors = sns.color_palette("husl", len(model_names))
        epochs_range = None
    
        for i, model_name in enumerate(model_names):
            train_runs = interval_logs[model_name]['train']
            val_runs   = interval_logs[model_name]['val']
    
            min_len     = min(len(run) for run in train_runs)
            train_array = np.array([run[:min_len] for run in train_runs])
            val_array   = np.array([run[:min_len] for run in val_runs])
    
            mean_train = train_array.mean(axis=0)
            std_train  = train_array.std(axis=0)
            mean_val   = val_array.mean(axis=0)
            std_val    = val_array.std(axis=0)
    
            if epochs_range is None:
                epochs_range = np.arange(1, min_len+1)
    
            color = colors[i]
    
            # Plot training
            plt.plot(epochs_range, mean_train, label=f"{model_name} - Train",
                     color=color, linewidth=2)
            plt.fill_between(epochs_range,
                             mean_train - std_train,
                             mean_train + std_train,
                             color=color, alpha=0.2)
    
            # Plot validation
            plt.plot(epochs_range, mean_val, '--', label=f"{model_name} - Val",
                     color=color, linewidth=2)
            plt.fill_between(epochs_range,
                             mean_val - std_val,
                             mean_val + std_val,
                             color=color, alpha=0.2)
    
        plt.yscale('log')
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("MSE Loss (Log Scale)", fontsize=14)
        plt.title("Training & Validation (Mean ± Std) over multiple runs",
                  fontsize=15, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    ##############################################################################
    # 4) Generate and **Save** a LaTeX Table
    ##############################################################################
    def generate_latex_table(results_summary, filename=None):
        """
        results_summary: dict of the form
           {
             model_name: {
                'mse_mean': float, 'mse_std': float,
                'mae_mean': float, 'mae_std': float,
                'time_mean': float, 'time_std': float,
                'num_params': int
             },
             ...
           }
        filename (str or None): if provided, save table to this file.
    
        Returns LaTeX string.
        """
        header = r"\begin{table}[ht]\centering" + "\n"
        header += r"\begin{tabular}{lcccc}" + "\n"
        header += r"\toprule" + "\n"
        header += r"Model & MSE & MAE & Time (s) & \# Params \\" + "\n"
        header += r"\midrule" + "\n"
    
        rows = ""
        for model_name, stats in results_summary.items():
            mse_str  = f"{stats['mse_mean']:.4f}±{stats['mse_std']:.4f}"
            mae_str  = f"{stats['mae_mean']:.4f}±{stats['mae_std']:.4f}"
            time_str = f"{stats['time_mean']:.2f}±{stats['time_std']:.2f}"
            params   = f"{stats['num_params']}"
            row = f"{model_name} & {mse_str} & {mae_str} & {time_str} & {params} \\\\ \n"
            rows += row
    
        footer = (r"\bottomrule" + "\n" +
                  r"\end{tabular}" + "\n" +
                  r"\caption{Comparison of Models}" + "\n" +
                  r"\end{table}")
    
        latex_str = header + rows + footer
    
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(latex_str)
    
        return latex_str
    
    ##############################################################################
    # 5) Box Plot of Errors
    ##############################################################################
    def plot_box_errors(error_dict):
        """
        error_dict = { model_name: array_of_abs_errors, ... }
        Shows a standard box plot of errors for each model.
        """
        model_names = list(error_dict.keys())
        errors_data = [error_dict[m] for m in model_names]
    
        plt.figure(figsize=(8, 6))
        box = plt.boxplot(errors_data, labels=model_names, patch_artist=True)
    
        # Color the boxes (optional)
        palette = sns.color_palette("husl", len(model_names))
        for patch, color in zip(box['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
    
        plt.title("Distribution of Absolute Errors (Test Set)",
                  fontsize=15, fontweight='bold')
        plt.ylabel("Absolute Error", fontsize=13)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    ##############################################################################
    # 6) Scatter Plot: Predictions vs. Truth
    ##############################################################################
    def scatter_predictions_vs_truth(truths, preds_dict):
        """
        Plot predictions vs. truth for all models on a single scatter plot.
        A perfect predictor => diagonal line y=x.
        """
        plt.figure(figsize=(7, 7))
        palette = sns.color_palette("husl", len(preds_dict))
        for i, (model_name, pred) in enumerate(preds_dict.items()):
            plt.scatter(pred, truths, s=40, alpha=0.6, label=model_name,
                        color=palette[i])
    
        min_val = min(truths.min(), *(p.min() for p in preds_dict.values()))
        max_val = max(truths.max(), *(p.max() for p in preds_dict.values()))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    
        plt.xlabel("Predicted", fontsize=13)
        plt.ylabel("True", fontsize=13)
        plt.title("Predictions vs. True (Test Set)", fontsize=15, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    ##############################################################################
    # 7) Prediction Errors (Overlay)
    ##############################################################################
    def plot_prediction_errors(X_plot, Y_plot_true, overlay_preds, errors):
        """
        For function approximation:
          overlay_preds: { model_name: 1D array of predicted y-values }
          errors:        { model_name: 1D array of absolute errors }
        """
        x_np = X_plot.cpu().numpy() if torch.is_tensor(X_plot) else X_plot
        sort_idx = np.argsort(x_np)
        x_sorted = x_np[sort_idx]
        y_sorted = Y_plot_true[sort_idx]
    
        # 7a) Overlaid predictions
        plt.figure(figsize=(9, 5))
        plt.plot(x_sorted, y_sorted, 'k-', linewidth=3, label="True")
        palette = sns.color_palette("husl", len(overlay_preds))
        for i, (mname, preds) in enumerate(overlay_preds.items()):
            p_sorted = preds[sort_idx]
            plt.plot(x_sorted, p_sorted, '--', linewidth=2, color=palette[i],
                     label=mname)
        plt.title("Model Predictions vs. True Function", fontsize=15, fontweight='bold')
        plt.xlabel("x", fontsize=13)
        plt.ylabel("y", fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
        # 7b) Plot error curves
        plt.figure(figsize=(9, 5))
        for i, (mname, err) in enumerate(errors.items()):
            e_sorted = err[sort_idx]
            plt.plot(x_sorted, e_sorted, '-', linewidth=2, color=palette[i],
                     label=f"{mname} Error")
        plt.title("Prediction Errors (Absolute Difference)", fontsize=15, fontweight='bold')
        plt.xlabel("x", fontsize=13)
        plt.ylabel("Absolute Error", fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 32 FAILED, continuing', flush=True)

_cell[0]=33
print('=== CELL 33 ===', flush=True)
try:
    # Imports
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    from utils import (
        loadData,
        train_and_evaluate_model,
        plot_training_curves_with_confidence,
        generate_latex_table,
        get_model_colors,
        plot_box_errors,
        scatter_predictions_vs_truth,
        plot_prediction_errors
    )
    # 1) Generate Data
    data_size = 300
    X = torch.linspace(0, 1, data_size)
    Y = challenging_function(X)
    
    # Scale Y
    scaler = MinMaxScaler()
    Y_np   = Y.numpy().reshape(-1, 1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
    # Split data
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    # 2) Define model constructors
    def cauchy_constructor(i, h, o):   return CauchyNet(i, h, o)
    def siren_constructor(i, h, o):    return SIREN(i, h, o)
    def rbf_constructor(i, h, o):      return RBFNetwork(i, h, o)
    def fnn_relu(i, h, o):             return FNN_ReLU(i, h, o)
    def transf_constructor(i, h, o):   return MinimalTransformer(i, h, o, nhead=4, num_layers=1)
    def informer_constructor(i, h, o): return MinimalInformer(i, h, o, nhead=4, num_layers=1)
    
    models_dict = {
        "CauchyNet":   cauchy_constructor,
        "SIREN":       siren_constructor,
        "RBF_Net":     rbf_constructor,
        "FNN_ReLU":    fnn_relu,
        "Transformer": transf_constructor,
        "Informer":    informer_constructor
    }
    
    # 3) Training config
    input_size  = 1
    hidden_size = 128
    output_size = 1
    lr          = 0.005
    epochs      = 400
    num_repeats = 5
    
    # 4) Train multiple runs & collect logs
    interval_logs  = {}
    results_dict   = {}
    for model_name, constructor in models_dict.items():
        print(f"=== {model_name} ===")
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        num_params = None
    
        for _ in range(num_repeats):
            (model, 
             train_losses, 
             val_losses,
             test_mse, 
             test_mae, 
             preds_scaled,
             truths_scaled,
             training_time,
             n_params) = train_and_evaluate_model(
                constructor,
                input_size, hidden_size, output_size,
                train_loader, val_loader, test_loader,
                lr=lr, epochs=epochs
            )
            runs_mse.append(test_mse)
            runs_mae.append(test_mae)
            runs_time.append(training_time)
            train_runs.append(train_losses)
            val_runs.append(val_losses)
            if num_params is None:
                num_params = n_params
    
        # Mean ± Std
        mse_mean, mse_std = np.mean(runs_mse), np.std(runs_mse)
        mae_mean, mae_std = np.mean(runs_mae), np.std(runs_mae)
        time_mean, time_std = np.mean(runs_time), np.std(runs_time)
    
        interval_logs[model_name] = {
            'train': train_runs,
            'val':   val_runs
        }
        results_dict[model_name] = {
            'mse_mean':    mse_mean, 'mse_std':    mse_std,
            'mae_mean':    mae_mean, 'mae_std':    mae_std,
            'time_mean':   time_mean,'time_std':   time_std,
            'num_params':  num_params
        }
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 33 FAILED, continuing', flush=True)

_cell[0]=34
print('=== CELL 34 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    ##############################################################################
    # For “professional style” figures, as often done in top ML journals:
    ##############################################################################
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["font.family"]  = "sans-serif"
    matplotlib.rcParams["figure.dpi"]   = 120
    
    SEED = 10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ##############################################################################
    # 1) Activation & Model Definitions
    ##############################################################################
    
    class ReciprocalActivation(nn.Module):
        """
        Reciprocal activation: out = 1.0 / (x + epsilon).
        Helps avoid division by zero or very small denominators.
        """
        def __init__(self, epsilon=1e-8):
            super().__init__()
            self.epsilon = epsilon
    
        def forward(self, x):
            return 1.0 / (x + self.epsilon)
    
    class CauchyNet(nn.Module):
        """
        CauchyNet with one hidden dimension 'hidden_size' and final output size 'output_size'.
        Returns (y_real, y_imag).
        We'll train with MSE(y_real, y_true) + MSE(y_imag, 0).
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Complex parameters
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: shape [batch_size, 1]
            # If x arrives as [batch], fix it:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
    
            batch_size, in_size = x.size()
            x_c = torch.complex(x, torch.zeros_like(x))  # shape [batch,1]
    
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)  # [batch, hidden_size]
            x_expanded  = x_c.expand(batch_size, self.hidden_size)     # [batch, hidden_size]
    
            activated = self.activation(xi_expanded - x_expanded)      # [batch, hidden_size]
            y = torch.matmul(activated, self.lambda_) / self.hidden_size  # [batch, output_size], complex
            return y.real, y.imag
    
    
    ##############################################################################
    # Multi-Activation FNN definitions
    ##############################################################################
    
    class FNN_ReLU(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.ReLU()
    
        def forward(self, x):
            # Ensure shape [batch,1]
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_Sigmoid(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.Sigmoid()
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_SiLU(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.SiLU()
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    ##############################################################################
    # 2) Data Loading
    ##############################################################################
    
    def loadData(X, Y, batchSize=64):
        # Ensure X has shape [N,1]
        if X.dim() == 1:
            X = X.unsqueeze(-1)
    
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize   = int(total_size * 0.25)
        valSize    = testSize
        trainSize  = total_size - testSize - valSize
    
        ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
    
        train_loader= DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        val_loader  = DataLoader(ds_val,   batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(ds_test,  batch_size=batchSize, shuffle=False)
    
        return train_loader, val_loader, test_loader, ds_test
    
    ##############################################################################
    # 3) Training & Evaluation
    ##############################################################################
    
    def train_and_evaluate_model(model_constructor, name,
                                 train_loader, val_loader, test_loader,
                                 input_size=1, hidden_size=32, output_size=1,
                                 lr=0.001, epochs=100):
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_state    = None
    
        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)
                if isinstance(out, tuple):
                    # (y_real, y_imag)
                    y_real, y_imag = out
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                else:
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch_expanded)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
    
            train_loss = train_loss_accum / len(train_loader)
            train_losses.append(train_loss)
    
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                    else:
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(out, y_batch_expanded)
                    val_loss_accum += vloss.item()
    
            val_loss = val_loss_accum / len(val_loader)
            val_losses.append(val_loss)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
    
            scheduler.step(val_loss)
    
        # Plot training & validation curve
        plt.figure(figsize=(5,4))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses,   label='Val Loss', linewidth=2)
        plt.title(f"{name} Learning Curve", fontsize=13)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
    
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test set
        model.eval()
        preds_list, truths_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    preds = y_real.cpu().numpy().flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths = y_batch.cpu().numpy().flatten()
                preds_list.append(preds)
                truths_list.append(truths)
    
        preds_all  = np.concatenate(preds_list)
        truths_all = np.concatenate(truths_list)
    
        # Return scaled predictions, truths
        return model, mean_squared_error(truths_all, preds_all), mean_absolute_error(truths_all, preds_all), preds_all, truths_all
    
    ##############################################################################
    # 4) Additional Plotting Helpers
    ##############################################################################
    
    def plot_prediction_curves(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plots each model's predicted curve vs. the true data over x_test, in unscaled space.
        We'll sort x_test for a nicer line plot.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        plt.figure(figsize=(6,4))
        plt.plot(x_sorted, y_sorted, 'k-', label="True", linewidth=2)
        for model_name, pred_array in predictions_dict.items():
            pred_sorted = pred_array[sort_idx]
            plt.plot(x_sorted, pred_sorted, '--', label=model_name, linewidth=1.5)
    
        plt.title(f"Prediction Curves {title_suffix}", fontsize=13)
        plt.xlabel("Input X", fontsize=12)
        plt.ylabel("Output Y", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plot each model's residual = (pred - true) vs. x_test in subplots, unscaled domain.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            pred_sorted = pred_array[sort_idx]
            residuals   = pred_sorted - y_sorted
            ax.scatter(x_sorted, residuals, alpha=0.6, label=model_name)
            ax.axhline(0.0, color="r", linestyle="--", linewidth=1.2)
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Residual", fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
            ax.legend(fontsize=10)
    
        plt.tight_layout()
        plt.show()
    
    def plot_residual_histograms(y_test, predictions_dict, bins=20, title_suffix=""):
        """
        Plot histogram of residuals for each model, side by side in subplots, unscaled domain.
        """
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            residuals = pred_array - y_test
            ax.hist(residuals, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
            ax.set_xlabel("Residual", fontsize=12)
            ax.set_ylabel("Count",    fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
    
        plt.tight_layout()
        plt.show()
    
    ##############################################################################
    # 5) Main script: Compare CauchyNet and multiple FNN variants
    ##############################################################################
    import torch
    import numpy as np
    
    # 1) Define the function
    def compute_function(x):
        """
        For x (tensor or array), compute:
        y = x^2 + 7*cos(5*x^2) + 2*sin(3*x) + 1/(x^2 + 8)
        """
        return x**2 + 7*torch.cos(5*x**2) + 2*torch.sin(3*x) + 1/(x**2 + 8)
    
    # 2) Generate x-values from 1.0 to 1.5, step=0.01
    #    We'll use torch.arange for that range:
    x_values = torch.arange(1.0, 1.5 + 0.01, 0.01)  # 1.5 + 0.01 to include 1.5 exactly
    
    # 3) Compute y-values using the formula
    y_values = compute_function(x_values)
    
    # 4) Print or store results
    #print("x_values:", x_values)
    #print("y_values:", y_values)
    
    
    
    if __name__ == "__main__":
        # Synthetic data
        data_size = 100
        X = torch.linspace(-1, 1, data_size)  # shape [N]
        Y = compute_function(X)               # shape [N]
    
    
    
        # Scale Y
        scaler = MinMaxScaler()
        Y_np   = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
        # Build dataset + splits
        train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
        input_size  = 1
        hidden_size = 32
        output_size = 1
        lr          = 0.001
        epochs      = 600
    
        def cauchy_constructor(in_f, h, out_f):
            return CauchyNet(in_f, h, out_f)
    
        def fnn_relu(in_f, h, out_f):
            return FNN_ReLU(in_f, h, out_f)
    
        def fnn_sigmoid(in_f, h, out_f):
            return FNN_Sigmoid(in_f, h, out_f)
    
        def fnn_silu(in_f, h, out_f):
            return FNN_SiLU(in_f, h, out_f)
    
        models_dict = {
            "CauchyNet"   : cauchy_constructor,
            "FNN_ReLU"    : fnn_relu,
            "FNN_Sigmoid" : fnn_sigmoid,
            "FNN_SiLU"    : fnn_silu
        }
    
        results = {}
        print("Model\t\tMSE\t\tMAE")
        for name, constructor in models_dict.items():
            model, mse_val, mae_val, preds_scaled, truths_scaled = train_and_evaluate_model(
                constructor, name,
                train_loader, val_loader, test_loader,
                input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, lr=lr, epochs=epochs
            )
            print(f"{name:10s}\t{mse_val:.6f}\t{mae_val:.6f}")
            results[name] = (mse_val, mae_val, preds_scaled, truths_scaled)
    
        # 1) Summarize MSE in bar chart
        model_names = list(results.keys())
        mses  = [results[n][0] for n in model_names]
    
        plt.figure(figsize=(5,4))
        plt.bar(model_names, mses, color='skyblue')
        plt.title("Comparison of MSE on Test Set", fontsize=13)
        plt.ylabel("MSE", fontsize=12)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()
    
        # 2) Build final test set in unscaled space:
        # ds_test is a subset of the original data (X, Y unscaled).
        test_indices = ds_test.indices
        X_np_all = X.numpy()  # unscaled X
        Y_np_all = Y.numpy()  # unscaled Y
        X_test_np = X_np_all[test_indices]   # shape [testSize]
        Y_test_np = Y_np_all[test_indices]   # shape [testSize]
    
        # 3) Convert scaled preds -> unscaled preds
        predictions_dict = {}
        for name, (mse_val, mae_val, preds_scaled, truths_scaled) in results.items():
            # Each is shape [testSize], but in scaled space
            preds_2d = preds_scaled.reshape(-1,1)
            preds_unscaled = scaler.inverse_transform(preds_2d).flatten()
            # We'll store unscaled predictions for plotting
            predictions_dict[name] = preds_unscaled
    
        # 4) Plot the predicted curves
        plot_prediction_curves(X_test_np, Y_test_np, predictions_dict, title_suffix="(Unscaled)")
    
        # 5) Residual scatter
        plot_residuals(X_test_np, Y_test_np, predictions_dict, title_suffix="(Unscaled)")
    
        # 6) Residual histogram
        plot_residual_histograms(Y_test_np, predictions_dict, bins=15, title_suffix="(Unscaled)")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 34 FAILED, continuing', flush=True)

_cell[0]=35
print('=== CELL 35 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    ##############################################################################
    # For “professional style” figures, as often done in top ML journals:
    ##############################################################################
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["font.family"]  = "sans-serif"
    matplotlib.rcParams["figure.dpi"]   = 120
    
    SEED = 10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ##############################################################################
    # 1) Activation & Model Definitions
    ##############################################################################
    
    class ReciprocalActivation(nn.Module):
        """
        Reciprocal activation: out = 1.0 / (x + epsilon).
        Helps avoid division by zero or very small denominators.
        """
        def __init__(self, epsilon=1e-8):
            super().__init__()
            self.epsilon = epsilon
    
        def forward(self, x):
            return 1.0 / (x + self.epsilon)
    
    class CauchyNet(nn.Module):
        """
        CauchyNet with one hidden dimension 'hidden_size' and final output size 'output_size'.
        Returns (y_real, y_imag).
        We'll train with MSE(y_real, y_true) + MSE(y_imag, 0).
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Complex parameters
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: shape [batch_size, 1]
            # If x arrives as [batch], fix it:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
    
            batch_size, in_size = x.size()
            x_c = torch.complex(x, torch.zeros_like(x))  # shape [batch,1]
    
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)  # [batch, hidden_size]
            x_expanded  = x_c.expand(batch_size, self.hidden_size)     # [batch, hidden_size]
    
            activated = self.activation(xi_expanded - x_expanded)      # [batch, hidden_size]
            y = torch.matmul(activated, self.lambda_) / self.hidden_size  # [batch, output_size], complex
            return y.real, y.imag
    
    
    ##############################################################################
    # Multi-Activation FNN definitions
    ##############################################################################
    
    class FNN_ReLU(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.ReLU()
    
        def forward(self, x):
            # Ensure shape [batch,1]
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_Sigmoid(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.Sigmoid()
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_SiLU(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.SiLU()
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    ##############################################################################
    # 2) Data Loading
    ##############################################################################
    
    def loadData(X, Y, batchSize=64):
        # Ensure X has shape [N,1]
        if X.dim() == 1:
            X = X.unsqueeze(-1)
    
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize   = int(total_size * 0.25)
        valSize    = testSize
        trainSize  = total_size - testSize - valSize
    
        ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
    
        train_loader= DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        val_loader  = DataLoader(ds_val,   batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(ds_test,  batch_size=batchSize, shuffle=False)
    
        return train_loader, val_loader, test_loader, ds_test
    
    ##############################################################################
    # 3) Training & Evaluation
    ##############################################################################
    
    def train_and_evaluate_model(model_constructor, name,
                                 train_loader, val_loader, test_loader,
                                 input_size=1, hidden_size=32, output_size=1,
                                 lr=0.001, epochs=100):
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_state    = None
    
        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)
                if isinstance(out, tuple):
                    # (y_real, y_imag)
                    y_real, y_imag = out
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                else:
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch_expanded)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
    
            train_loss = train_loss_accum / len(train_loader)
            train_losses.append(train_loss)
    
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                    else:
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(out, y_batch_expanded)
                    val_loss_accum += vloss.item()
    
            val_loss = val_loss_accum / len(val_loader)
            val_losses.append(val_loss)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
    
            scheduler.step(val_loss)
    
        # Plot training & validation curve
        plt.figure(figsize=(5,4))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses,   label='Val Loss', linewidth=2)
        plt.title(f"{name} Learning Curve", fontsize=13)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
    
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test set
        model.eval()
        preds_list, truths_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    preds = y_real.cpu().numpy().flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths = y_batch.cpu().numpy().flatten()
                preds_list.append(preds)
                truths_list.append(truths)
    
        preds_all  = np.concatenate(preds_list)
        truths_all = np.concatenate(truths_list)
    
        # Return scaled predictions, truths
        return model, mean_squared_error(truths_all, preds_all), mean_absolute_error(truths_all, preds_all), preds_all, truths_all
    
    ##############################################################################
    # 4) Additional Plotting Helpers
    ##############################################################################
    
    def plot_prediction_curves(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plots each model's predicted curve vs. the true data over x_test, in unscaled space.
        We'll sort x_test for a nicer line plot.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        plt.figure(figsize=(6,4))
        plt.plot(x_sorted, y_sorted, 'k-', label="True", linewidth=2)
        for model_name, pred_array in predictions_dict.items():
            pred_sorted = pred_array[sort_idx]
            plt.plot(x_sorted, pred_sorted, '--', label=model_name, linewidth=1.5)
    
        plt.title(f"Prediction Curves {title_suffix}", fontsize=13)
        plt.xlabel("Input X", fontsize=12)
        plt.ylabel("Output Y", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plot each model's residual = (pred - true) vs. x_test in subplots, unscaled domain.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            pred_sorted = pred_array[sort_idx]
            residuals   = pred_sorted - y_sorted
            ax.scatter(x_sorted, residuals, alpha=0.6, label=model_name)
            ax.axhline(0.0, color="r", linestyle="--", linewidth=1.2)
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Residual", fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
            ax.legend(fontsize=10)
    
        plt.tight_layout()
        plt.show()
    
    def plot_residual_histograms(y_test, predictions_dict, bins=20, title_suffix=""):
        """
        Plot histogram of residuals for each model, side by side in subplots, unscaled domain.
        """
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            residuals = pred_array - y_test
            ax.hist(residuals, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
            ax.set_xlabel("Residual", fontsize=12)
            ax.set_ylabel("Count",    fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
    
        plt.tight_layout()
        plt.show()
    
    ##############################################################################
    # 5) Main script: Compare CauchyNet and multiple FNN variants
    ##############################################################################
    if __name__ == "__main__":
        # Synthetic data
        data_size = 100
        X = torch.linspace(-1, 1, data_size)  # shape [N]
        def compute_function(x):
            return x**2 + 7*torch.cos(5*x**2) + 2*torch.sin(3*x) + 1/(x**2 + 8)
        Y = compute_function(X)               # shape [N]
    
        # Scale Y
        scaler = MinMaxScaler()
        Y_np   = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
        # Build dataset + splits
        train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
        input_size  = 1
        hidden_size = 32
        output_size = 1
        lr          = 0.001
        epochs      = 600
    
        def cauchy_constructor(in_f, h, out_f):
            return CauchyNet(in_f, h, out_f)
    
        def fnn_relu(in_f, h, out_f):
            return FNN_ReLU(in_f, h, out_f)
    
        def fnn_sigmoid(in_f, h, out_f):
            return FNN_Sigmoid(in_f, h, out_f)
    
        def fnn_silu(in_f, h, out_f):
            return FNN_SiLU(in_f, h, out_f)
    
        models_dict = {
            "CauchyNet"   : cauchy_constructor,
            "FNN_ReLU"    : fnn_relu,
            "FNN_Sigmoid" : fnn_sigmoid,
            "FNN_SiLU"    : fnn_silu
        }
    
        results = {}
        print("Model\t\tMSE\t\tMAE")
        for name, constructor in models_dict.items():
            model, mse_val, mae_val, preds_scaled, truths_scaled = train_and_evaluate_model(
                constructor, name,
                train_loader, val_loader, test_loader,
                input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, lr=lr, epochs=epochs
            )
            print(f"{name:10s}\t{mse_val:.6f}\t{mae_val:.6f}")
            results[name] = (mse_val, mae_val, preds_scaled, truths_scaled)
    
        # 1) Summarize MSE in bar chart
        model_names = list(results.keys())
        mses  = [results[n][0] for n in model_names]
    
        plt.figure(figsize=(5,4))
        plt.bar(model_names, mses, color='skyblue')
        plt.title("Comparison of MSE on Test Set", fontsize=13)
        plt.ylabel("MSE", fontsize=12)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()
    
        # 2) Build final test set in unscaled space:
        # ds_test is a subset of the original data (X, Y unscaled).
        test_indices = ds_test.indices
        X_np_all = X.numpy()  # unscaled X
        Y_np_all = Y.numpy()  # unscaled Y
        X_test_np = X_np_all[test_indices]   # shape [testSize]
        Y_test_np = Y_np_all[test_indices]   # shape [testSize]
    
        # 3) Convert scaled preds -> unscaled preds
        predictions_dict = {}
        for name, (mse_val, mae_val, preds_scaled, truths_scaled) in results.items():
            # Each is shape [testSize], but in scaled space
            preds_2d = preds_scaled.reshape(-1,1)
            preds_unscaled = scaler.inverse_transform(preds_2d).flatten()
            # We'll store unscaled predictions for plotting
            predictions_dict[name] = preds_unscaled
    
        # 4) Plot the predicted curves
        plot_prediction_curves(X_test_np, Y_test_np, predictions_dict, title_suffix="(Unscaled)")
    
        # 5) Residual scatter
        plot_residuals(X_test_np, Y_test_np, predictions_dict, title_suffix="(Unscaled)")
    
        # 6) Residual histogram
        plot_residual_histograms(Y_test_np, predictions_dict, bins=15, title_suffix="(Unscaled)")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 35 FAILED, continuing', flush=True)

_cell[0]=36
print('=== CELL 36 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    ##############################################################################
    # For “professional style” figures, as often done in top ML journals:
    ##############################################################################
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["font.family"]  = "sans-serif"
    matplotlib.rcParams["figure.dpi"]   = 120
    
    SEED = 10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ##############################################################################
    # 1) Activation & Model Definitions
    ##############################################################################
    
    class ReciprocalActivation(nn.Module):
        """
        Reciprocal activation: out = 1.0 / (x + epsilon).
        Helps avoid division by zero or very small denominators.
        """
        def __init__(self, epsilon=1e-8):
            super().__init__()
            self.epsilon = epsilon
    
        def forward(self, x):
            return 1.0 / (x + self.epsilon)
    
    class CauchyNet(nn.Module):
        """
        CauchyNet with one hidden dimension 'hidden_size' and final output size 'output_size'.
        Returns (y_real, y_imag).
        We'll train with MSE(y_real, y_true) + MSE(y_imag, 0).
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Complex parameters
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: shape [batch_size, 1]
            batch_size, in_size = x.size()
            x_c = torch.complex(x, torch.zeros_like(x))  # shape [batch,1]
    
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)  # [batch, hidden_size]
            x_expanded  = x_c.expand(batch_size, self.hidden_size)      # [batch, hidden_size]
    
            activated = self.activation(xi_expanded - x_expanded)       # [batch, hidden_size]
            y = torch.matmul(activated, self.lambda_) / self.hidden_size   # [batch, output_size], complex
            return y.real, y.imag
    
    
    ##############################################################################
    # Multi-Activation FNN definitions
    ##############################################################################
    
    class FNN_ReLU(nn.Module):
        """
        2-layer feedforward neural net with ReLU activation.
        input_size -> hidden_size -> output_size
        """
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.ReLU()
    
        def forward(self, x):
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_Sigmoid(nn.Module):
        """
        2-layer feedforward neural net with Sigmoid activation.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.Sigmoid()
    
        def forward(self, x):
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_SiLU(nn.Module):
        """
        2-layer feedforward neural net with SiLU (Swish) activation.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.SiLU()   # or nn.Swish, if it existed
    
        def forward(self, x):
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    
    ##############################################################################
    # 2) Data Loading
    ##############################################################################
    
    def loadData(X, Y, batchSize=64):
        # X: [N], Y: [N], reshape X-> [N,1]
        X = X.unsqueeze(-1)  # [N,1]
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize = int(total_size * 0.25)
        valSize  = testSize
        trainSize= total_size - testSize - valSize
        ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
        test_indices = ds_test.indices  # array of shape [testSize]
        X_test_np = X[test_indices].numpy()
        Y_test_np = Y[test_indices].numpy()
    
    
        train_loader= DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        val_loader  = DataLoader(ds_val,   batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(ds_test,  batch_size=batchSize, shuffle=False)
        return train_loader, val_loader, test_loader
    
    ##############################################################################
    # 3) Training & Evaluation
    ##############################################################################
    
    def train_and_evaluate_model(model_constructor, name,
                                 train_loader, val_loader, test_loader,
                                 input_size=1, hidden_size=32, output_size=1,
                                 lr=0.001, epochs=100):
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        
        # for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_state    = None
    
        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)
                if isinstance(out, tuple):
                    # (y_real, y_imag) from CauchyNet
                    y_real, y_imag = out
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                else:
                    # normal feedforward
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch_expanded)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
    
            train_loss = train_loss_accum / len(train_loader)
            train_losses.append(train_loss)
    
            # Validation
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                    else:
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(out, y_batch_expanded)
                    val_loss_accum += vloss.item()
    
            val_loss = val_loss_accum / len(val_loader)
            val_losses.append(val_loss)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
    
            scheduler.step(val_loss)
    
        # Plot training & validation curve
        plt.figure(figsize=(5,4))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses,   label='Val Loss',   linewidth=2)
        plt.title(f"{name} Learning Curve", fontsize=13)
        plt.xlabel('Epoch',  fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
    
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test set
        model.eval()
        preds_list, truths_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    preds = y_real.cpu().numpy().flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths = y_batch.cpu().numpy().flatten()
                preds_list.append(preds)
                truths_list.append(truths)
    
        preds_all  = np.concatenate(preds_list)
        truths_all = np.concatenate(truths_list)
    
        mse_val = mean_squared_error(truths_all, preds_all)
        mae_val = mean_absolute_error(truths_all, preds_all)
        return model, mse_val, mae_val, preds_all, truths_all
    
    ##############################################################################
    # 4) Additional Plotting Helpers
    ##############################################################################
    
    def plot_prediction_curves(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plots each model's predicted curve vs. the true data over x_test.
        We'll sort x_test for a nicer line plot.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
    
        plt.figure(figsize=(6,4))
        plt.plot(x_sorted, y_sorted, 'k-', label="True", linewidth=2)
        for model_name, pred_array in predictions_dict.items():
            pred_sorted = pred_array[sort_idx]
            plt.plot(x_sorted, pred_sorted, '--', label=model_name, linewidth=1.5)
    
        plt.title(f"Prediction Curves {title_suffix}", fontsize=13)
        plt.xlabel("Input X", fontsize=12)
        plt.ylabel("Output Y", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
    
    
    def plot_residuals(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plot each model's residual = (pred - true) vs. x_test in subplots.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            pred_sorted = pred_array[sort_idx]
            residuals   = pred_sorted - y_sorted
            ax.scatter(x_sorted, residuals, alpha=0.6, label=model_name)
            ax.axhline(0.0, color="r", linestyle="--", linewidth=1.2)
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Residual", fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
            ax.legend(fontsize=10)
    
        plt.tight_layout()
        plt.show()
    
    
    def plot_residual_histograms(y_test, predictions_dict, bins=20, title_suffix=""):
        """
        Plot histogram of residuals for each model, side by side in subplots.
        """
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            residuals = pred_array - y_test
            ax.hist(residuals, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
            ax.set_xlabel("Residual", fontsize=12)
            ax.set_ylabel("Count",    fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
    
        plt.tight_layout()
        plt.show()
    
    
    ##############################################################################
    # 5) Main script: Compare CauchyNet and multiple FNN variants
    ##############################################################################
    
    if __name__ == "__main__":
        # Synthetic data
        data_size = 100
        X = torch.linspace(-1, 1, data_size)
        def compute_function(X):
            return X**2 + 7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
        Y = compute_function(X)
    
        # Scale Y
        scaler = MinMaxScaler()
        Y_np   = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
        # DataLoaders
        train_loader, val_loader, test_loader = loadData(X, Y_t, batchSize=32)
    
        input_size  = 1
        hidden_size = 32
        output_size = 1
        lr          = 0.001
        epochs      = 800
    
        # We define multiple feedforward nets with different activations + your CauchyNet
        def cauchy_constructor(in_f, h, out_f):
            return CauchyNet(in_f, h, out_f)
    
        def fnn_relu(in_f, h, out_f):
            return FNN_ReLU(in_f, h, out_f)
    
        def fnn_sigmoid(in_f, h, out_f):
            return FNN_Sigmoid(in_f, h, out_f)
    
        def fnn_silu(in_f, h, out_f):
            return FNN_SiLU(in_f, h, out_f)
    
        models_dict = {
            "CauchyNet"    : cauchy_constructor,
            "FNN_ReLU"     : fnn_relu,
            "FNN_Sigmoid"  : fnn_sigmoid,
            "FNN_SiLU"     : fnn_silu,
        }
    
        results = {}
        print("Model\tMSE\tMAE")
        for name, constructor in models_dict.items():
            model, mse_val, mae_val, preds_all, truths_all = train_and_evaluate_model(
                constructor, name,
                train_loader, val_loader, test_loader,
                input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, lr=lr, epochs=epochs
            )
            print(f"{name}\t{mse_val:.6f}\t{mae_val:.6f}")
            results[name] = (mse_val, mae_val, preds_all, truths_all)
    
        # 1) Bar chart of MSE
        names = list(results.keys())
        mses  = [results[n][0] for n in names]
    
        plt.figure(figsize=(5,4))
        plt.bar(names, mses, color='skyblue')
        plt.title("Comparison of MSE on Test Set", fontsize=13)
        plt.ylabel("MSE", fontsize=12)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()
    
        # 2) Gather predictions for additional plots
        # We'll pick any model's truths (they should be the same)
        any_model = next(iter(results.keys()))
        _, _, _, common_truths = results[any_model]
        # X_test_np = X.numpy()         # shape [Ntest], in [-1,1]
        # Y_test_np = common_truths      # shape [Ntest]
    
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize = int(total_size * 0.25)
        valSize  = testSize
        trainSize= total_size - testSize - valSize
    
        ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
    
    # We have a full X, Y of length N
        X_np = X.numpy()  # shape [N]
        Y_np = Y.numpy()  # shape [N]
    
    # test_indices is just a list of test-sample indices
        test_indices = ds_test.indices
    
    # Now gather
        X_test_np = X_np[test_indices]  # e.g. shape [25]
        Y_test_np = Y_np[test_indices]  # e.g. shape [25]
        print(len(X_test_np), len(Y_test_np)) 
    
    # (Now X_test_np, Y_test_np) are the same length.
    # No more IndexError from sorting one that is bigger than the other.
       # plot_prediction_curves(X_test_np, Y_test_np, predictions_dict)
    
        predictions_dict = {}
        for name, (_, _, preds_all, _) in results.items():
            predictions_dict[name] = preds_all
    
        # 3) Plot the predicted curves
        plot_prediction_curves(X_test_np, Y_test_np, predictions_dict, title_suffix="(Test)")
    
        # 4) Residual scatter plots
        plot_residuals(X_test_np, Y_test_np, predictions_dict, title_suffix="(Test)")
    
        # 5) Residual histograms
        plot_residual_histograms(Y_test_np, predictions_dict, bins=15, title_suffix="(Test)")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 36 FAILED, continuing', flush=True)

_cell[0]=37
print('=== CELL 37 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    ##############################################################################
    # For “professional style” figures, as often done in top ML journals:
    ##############################################################################
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["font.family"]  = "sans-serif"
    matplotlib.rcParams["figure.dpi"]   = 120
    
    SEED = 10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ##############################################################################
    # 1) Activation & Model Definitions
    ##############################################################################
    
    class ReciprocalActivation(nn.Module):
        """
        Reciprocal activation: out = 1.0 / (x + epsilon).
        Helps avoid division by zero or very small denominators.
        """
        def __init__(self, epsilon=1e-8):
            super().__init__()
            self.epsilon = epsilon
    
        def forward(self, x):
            return 1.0 / (x + self.epsilon)
    
    class CauchyNet(nn.Module):
        """
        CauchyNet with one hidden dimension 'hidden_size' and final output size 'output_size'.
        Returns (y_real, y_imag).
        We'll train with MSE(y_real, y_true) + MSE(y_imag, 0).
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Complex parameters
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: shape [batch_size, 1]
            # In case x arrives as [batch_size], ensure we unsqueeze:
            if x.dim() == 1:
                x = x.unsqueeze(-1)  # [batch,1]
    
            batch_size, in_size = x.size()  # in_size should be 1
            x_c = torch.complex(x, torch.zeros_like(x))  # shape [batch,1]
    
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)  # [batch, hidden_size]
            x_expanded  = x_c.expand(batch_size, self.hidden_size)     # [batch, hidden_size]
    
            activated = self.activation(xi_expanded - x_expanded)      # [batch, hidden_size]
            y = torch.matmul(activated, self.lambda_) / self.hidden_size  # [batch, output_size], complex
            return y.real, y.imag
    
    
    ##############################################################################
    # Multi-Activation FNN definitions
    ##############################################################################
    
    class FNN_ReLU(nn.Module):
        """
        2-layer feedforward neural net with ReLU activation.
        input_size -> hidden_size -> output_size
        """
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.ReLU()
    
        def forward(self, x):
            # Ensure shape [batch,1]
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_Sigmoid(nn.Module):
        """
        2-layer feedforward neural net with Sigmoid activation.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.Sigmoid()
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_SiLU(nn.Module):
        """
        2-layer feedforward neural net with SiLU (Swish) activation.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.SiLU()   # or nn.Swish, if it existed
    
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    ##############################################################################
    # 2) Data Loading
    ##############################################################################
    
    def loadData(X, Y, batchSize=64):
        """
        X: shape [N] or [N,1]
        Y: shape [N]
        We ensure final shape of X => [N,1] for the dataset.
        """
        if X.dim() == 1:
            X = X.unsqueeze(-1)  # shape => [N,1]
    
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize   = int(total_size * 0.25)
        valSize    = testSize
        trainSize  = total_size - testSize - valSize
    
        ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
    
        train_loader= DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        val_loader  = DataLoader(ds_val, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(ds_test,batch_size=batchSize, shuffle=False)
    
        return train_loader, val_loader, test_loader, ds_test
    
    ##############################################################################
    # 3) Training & Evaluation
    ##############################################################################
    
    def train_and_evaluate_model(model_constructor, name,
                                 train_loader, val_loader, test_loader,
                                 input_size=1, hidden_size=32, output_size=1,
                                 lr=0.001, epochs=100):
    
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    
        # For gradient stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_state    = None
    
        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
    
                out = model(x_batch)  # e.g. FNN => [batch,1], or Cauchy => (y_real,y_imag)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                else:
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch_expanded)
    
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
    
            train_loss = train_loss_accum / len(train_loader)
            train_losses.append(train_loss)
    
            # Validation
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                    else:
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(out, y_batch_expanded)
                    val_loss_accum += vloss.item()
    
            val_loss = val_loss_accum / len(val_loader)
            val_losses.append(val_loss)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
    
            scheduler.step(val_loss)
    
        # Plot training vs val curve
        plt.figure(figsize=(5,4))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses,   label='Val Loss', linewidth=2)
        plt.title(f"{name} Learning Curve", fontsize=13)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
    
        # Load best
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test set
        model.eval()
        preds_list, truths_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    preds = y_real.cpu().numpy().flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths = y_batch.cpu().numpy().flatten()
                preds_list.append(preds)
                truths_list.append(truths)
    
        preds_all  = np.concatenate(preds_list)
        truths_all = np.concatenate(truths_list)
    
        mse_val = mean_squared_error(truths_all, preds_all)
        mae_val = mean_absolute_error(truths_all, preds_all)
        return model, mse_val, mae_val, preds_all, truths_all
    
    ##############################################################################
    # 4) Additional Plotting Helpers
    ##############################################################################
    
    def plot_prediction_curves(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plots each model's predicted curve vs. the true data over x_test.
        We'll sort x_test for a nicer line plot.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        plt.figure(figsize=(6,4))
        plt.plot(x_sorted, y_sorted, 'k-', label="True", linewidth=2)
        for model_name, pred_array in predictions_dict.items():
            pred_sorted = pred_array[sort_idx]
            plt.plot(x_sorted, pred_sorted, '--', label=model_name, linewidth=1.5)
    
        plt.title(f"Prediction Curves {title_suffix}", fontsize=13)
        plt.xlabel("Input X", fontsize=12)
        plt.ylabel("Output Y", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plot each model's residual = (pred - true) vs. x_test in subplots.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            pred_sorted = pred_array[sort_idx]
            residuals   = pred_sorted - y_sorted
            ax.scatter(x_sorted, residuals, alpha=0.6, label=model_name)
            ax.axhline(0.0, color="r", linestyle="--", linewidth=1.2)
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Residual", fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
            ax.legend(fontsize=10)
    
        plt.tight_layout()
        plt.show()
    
    def plot_residual_histograms(y_test, predictions_dict, bins=20, title_suffix=""):
        """
        Plot histogram of residuals for each model, side by side in subplots.
        """
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            residuals = pred_array - y_test
            ax.hist(residuals, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
            ax.set_xlabel("Residual", fontsize=12)
            ax.set_ylabel("Count",    fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
    
        plt.tight_layout()
        plt.show()
    
    ##############################################################################
    # 5) Main script: Compare CauchyNet and multiple FNN variants
    ##############################################################################
    if __name__ == "__main__":
    
        # Synthetic data
        data_size = 100
        X = torch.linspace(-1, 1, data_size)  # shape [N]
        def compute_function(x):
            return x**2 + 7*torch.cos(5*x**2) + 2*torch.sin(3*x) + 1/(x**2 + 8)
        Y = compute_function(X)  # shape [N]
    
        # Scale Y
        scaler = MinMaxScaler()
        Y_np   = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
        # Build dataset + splits
        train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
        input_size  = 1
        hidden_size = 32
        output_size = 1
        lr          = 0.001
        epochs      = 600
    
        # We define multiple feedforward nets with different activations + your CauchyNet
        def cauchy_constructor(in_f, h, out_f):
            return CauchyNet(in_f, h, out_f)
    
        def fnn_relu(in_f, h, out_f):
            return FNN_ReLU(in_f, h, out_f)
    
        def fnn_sigmoid(in_f, h, out_f):
            return FNN_Sigmoid(in_f, h, out_f)
    
        def fnn_silu(in_f, h, out_f):
            return FNN_SiLU(in_f, h, out_f)
    
        models_dict = {
            "CauchyNet"   : cauchy_constructor,
            "FNN_ReLU"    : fnn_relu,
            "FNN_Sigmoid" : fnn_sigmoid,
            "FNN_SiLU"    : fnn_silu
        }
    
        results = {}
        print("Model\t\tMSE\t\tMAE")
        for name, constructor in models_dict.items():
            model, mse_val, mae_val, preds_all, truths_all = train_and_evaluate_model(
                constructor, name,
                train_loader, val_loader, test_loader,
                input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, lr=lr, epochs=epochs
            )
            print(f"{name:10s}\t{mse_val:.6f}\t{mae_val:.6f}")
            results[name] = (mse_val, mae_val, preds_all, truths_all)
    
        # 1) Summarize MSE in bar chart
        model_names = list(results.keys())
        mses  = [results[n][0] for n in model_names]
    
        plt.figure(figsize=(5,4))
        plt.bar(model_names, mses, color='skyblue')
        plt.title("Comparison of MSE on Test Set", fontsize=13)
        plt.ylabel("MSE", fontsize=12)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()
    
        # 2) For plotting, we need the actual test X, Y (unscaled)
        # ds_test is the subset of original data, X is not scaled, Y is not scaled => shape [testSize].
        # We can retrieve indices from ds_test and gather them from the original X, Y.
        test_indices = ds_test.indices  # e.g. shape [testSize]
        X_np_all = X.numpy()  # [N]
        Y_np_all = Y.numpy()  # [N]
        X_test_np = X_np_all[test_indices]  # shape [testSize]
        Y_test_np = Y_np_all[test_indices]  # shape [testSize]
    
        # 3) Build a dictionary for predictions. 
        #    However, 'preds_all' and 'truths_all' are in scaled space. 
        #    If we want to compare in scaled space, we use them as is. 
        #    If we want to revert them to original space, we'd do an inverse transform.
        #    Suppose we keep the scaled approach for the Y dimension, but X is unscaled:
        #    It's a bit of a mismatch. We can still see relative shape though.
        #    Alternatively, we do inverse transform on preds_all and truths_all to get unscaled.
        
        predictions_dict = {}
        for name, (_, _, preds_all, truths_all) in results.items():
            # Inverse transform if you want original Y scale:
            preds_2d  = preds_all.reshape(-1,1)
            truths_2d = truths_all.reshape(-1,1)
            preds_inv   = scaler.inverse_transform(preds_2d).flatten()
            truths_inv  = scaler.inverse_transform(truths_2d).flatten()
            # We'll store the unscaled predictions
            predictions_dict[name] = preds_inv
    
        # Also unscaled ground-truth
        # Now, Y_test_np is already unscaled => We can compare that with preds_inv
        # The shape of Y_test_np is same as test_indices => compare with predictions_dict
        # But note that 'predictions_dict[name]' is in the order of test dataset (accumulated), 
        # which matches the order in which test data was loaded. We want it sorted by the same ds_test order.
    
        # There's no default "accumulated" order guaranteed to match test_indices sorted. 
        # The simplest approach: reorder them to the same test_indices sorting. 
        # But we can just rely on X_test_np, Y_test_np direct. We'll not re-sort the predictions again, 
        # because we do not know the batch order. We can just do a new array.
        
        # Let's do the simpler approach: each model's 'preds_all' is in test dataset order => we can just match them 
        # with X_test_np, Y_test_np also in test dataset order if the loader is stable. 
        # If the loader is not stable, we have to do more advanced mapping. 
        # We'll assume it wasn't shuffled => or we can do a stable approach. 
        # In practice, random_split doesn't shuffle ds_test. We'll assume batch=1 or no randomization for the test loader, or stable sort. 
        # For clarity, let's do a direct approach: do not re-sort them. We'll just do a scatter plot. 
        # If we want line plotting, we can do sorting ourselves. 
        # For demonstration, let's show a simpler approach that won't fail. We'll create a direct scatter plot with the test data in the same order. 
        # But let's do your approach: we can do a line plot if we sort by X.
    
        # 4) Now let's produce the final plots (prediction curves, etc.) 
        #    We'll do it in a stable approach that doesn't rely on the order of test loader. 
        #    We'll just show a scatter plot for predictions vs. X.
    
        # (a) Scatter of predictions vs. X
        # plt.figure(figsize=(6,4))
        # plt.scatter(X_test_np, Y_test_np, 'ko', label='True', markersize=4)
        # for name in predictions_dict:
        #     plt.scatter(X_test_np, predictions_dict[name], 'o--', label=name, markersize=3)
        # plt.title("Predicted vs. X (Unsorted)", fontsize=12)
        # plt.xlabel("X test", fontsize=11)
        # plt.ylabel("Y test (Unscaled)", fontsize=11)
        # plt.legend(fontsize=10)
        # plt.tight_layout()
        # plt.show()
        plt.figure(figsize=(6,4))
    
    # 1) Plot the true test data
        plt.scatter(
        X_test_np, Y_test_np,
        marker='o', edgecolor='k', facecolor='none',  # open circle
        s=30, linewidth=1.2,
        label='True'
        )
    
    # 2) Plot each model's predictions
        colors = plt.cm.viridis(np.linspace(0, 1, len(predictions_dict)))
        for (name, preds), c in zip(predictions_dict.items(), colors):
            plt.scatter(
            X_test_np, preds,
            marker='x', color=c,
            s=40, linewidth=1.0,
            alpha=0.8,
            label=name
        )
    
        plt.title("Predictions vs. X (Test Set)", fontsize=13, pad=10)
        plt.xlabel("X test", fontsize=12)
        plt.ylabel("Y test (Unscaled)", fontsize=12)
        plt.legend(fontsize=11, frameon=True)
        plt.tight_layout()
        plt.show()
    
    
        # (b) For a "curve", we can do sorting by X and reindex predictions accordingly. 
        #     We'll just do your function plot_prediction_curves, but we must make sure 
        #     predictions_dict is sorted in the same order as X_test_np. 
        #     However, we only have 'predictions_dict[name]' in the test iteration order. 
        #     If the test set is small and not shuffled, it might be strictly ascending but not guaranteed. 
        #     Let's rely on your plot function anyway:
    
        plot_prediction_curves(X_test_np, Y_test_np, predictions_dict, title_suffix="(Unscaled)")
    
        # (c) Residual scatter
        plot_residuals(X_test_np, Y_test_np, predictions_dict, title_suffix="(Unscaled)")
    
        # (d) Residual hist
        plot_residual_histograms(Y_test_np, predictions_dict, bins=15, title_suffix="(Unscaled)")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 37 FAILED, continuing', flush=True)

_cell[0]=38
print('=== CELL 38 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    ##############################################################################
    # For “professional style” figures, as often done in top ML journals:
    ##############################################################################
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["font.family"]  = "sans-serif"
    matplotlib.rcParams["figure.dpi"]   = 120
    
    SEED = 10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ##############################################################################
    # 1) Activation & Model Definitions
    ##############################################################################
    
    class ReciprocalActivation(nn.Module):
        """
        Reciprocal activation: out = 1.0 / (x + epsilon).
        Helps avoid division by zero or very small denominators.
        """
        def __init__(self, epsilon=1e-8):
            super().__init__()
            self.epsilon = epsilon
    
        def forward(self, x):
            return 1.0 / (x + self.epsilon)
    
    class CauchyNet(nn.Module):
        """
        CauchyNet with one hidden dimension 'hidden_size' and final output size 'output_size'.
        Returns (y_real, y_imag).
        We'll train with MSE(y_real, y_true) + MSE(y_imag, 0).
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Complex parameters
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: shape [batch_size, 1]
            batch_size, in_size = x.size()
            x_c = torch.complex(x, torch.zeros_like(x))  # shape [batch,1]
    
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)  # [batch, hidden_size]
            x_expanded  = x_c.expand(batch_size, self.hidden_size)      # [batch, hidden_size]
    
            activated = self.activation(xi_expanded - x_expanded)       # [batch, hidden_size]
            y = torch.matmul(activated, self.lambda_) / self.hidden_size   # [batch, output_size], complex
            return y.real, y.imag
    
    
    ##############################################################################
    # Multi-Activation FNN definitions
    ##############################################################################
    
    class FNN_ReLU(nn.Module):
        """
        2-layer feedforward neural net with ReLU activation.
        input_size -> hidden_size -> output_size
        """
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.ReLU()
    
        def forward(self, x):
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_Sigmoid(nn.Module):
        """
        2-layer feedforward neural net with Sigmoid activation.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.Sigmoid()
    
        def forward(self, x):
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class FNN_SiLU(nn.Module):
        """
        2-layer feedforward neural net with SiLU (Swish) activation.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.act = nn.SiLU()   # or nn.Swish, if it existed
    
        def forward(self, x):
            h = self.act(self.fc1(x))
            out = self.fc2(h)
            return out
    
    
    ##########################################
    # Data Loading and Training
    ##########################################
    
    def loadData(X, Y, batchSize=64):
        X = X.unsqueeze(-1)  # [N,1]
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize= total_size - testSize - valSize
        ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
        train_loader = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        val_loader   = DataLoader(ds_val,   batch_size=batchSize, shuffle=False)
        test_loader  = DataLoader(ds_test,  batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader, ds_test
    
    def train_and_evaluate_model(model_constructor, name,
                                 train_loader, val_loader, test_loader,
                                 input_size=1, hidden_size=32, output_size=1,
                                 lr=0.001, epochs=100):
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_state    = None
    
        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                else:
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch_expanded)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
    
            train_loss = train_loss_accum / len(train_loader)
            train_losses.append(train_loss)
    
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                    else:
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(out, y_batch_expanded)
                    val_loss_accum += vloss.item()
    
            val_loss = val_loss_accum / len(val_loader)
            val_losses.append(val_loss)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
    
            scheduler.step(val_loss)
    
        # Plot train vs val
        plt.figure(figsize=(5,4))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses,   label='Val Loss', linewidth=2)
        plt.title(f"{name} Learning Curve", fontsize=13)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
    
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test set
        model.eval()
        preds_list, truths_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    preds = y_real.cpu().numpy().flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths = y_batch.cpu().numpy().flatten()
                preds_list.append(preds)
                truths_list.append(truths)
    
        preds_all  = np.concatenate(preds_list)
        truths_all = np.concatenate(truths_list)
    
        mse_val = mean_squared_error(truths_all, preds_all)
        mae_val = mean_absolute_error(truths_all, preds_all)
        return model, mse_val, mae_val, preds_all, truths_all
    
    
    ##############################################################################
    # 4) Additional Plotting Helpers
    ##############################################################################
    
    def plot_prediction_curves(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plots each model's predicted curve vs. the true data over x_test.
        We'll sort x_test for a nicer line plot.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        plt.figure(figsize=(6,4))
        plt.plot(x_sorted, y_sorted, 'k-', label="True", linewidth=2)
        for model_name, pred_array in predictions_dict.items():
            pred_sorted = pred_array[sort_idx]
            plt.plot(x_sorted, pred_sorted, '--', label=model_name, linewidth=1.5)
    
        plt.title(f"Prediction Curves {title_suffix}", fontsize=13)
        plt.xlabel("Input X", fontsize=12)
        plt.ylabel("Output Y", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plot each model's residual = (pred - true) vs. x_test in subplots.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            pred_sorted = pred_array[sort_idx]
            residuals   = pred_sorted - y_sorted
            ax.scatter(x_sorted, residuals, alpha=0.6, label=model_name)
            ax.axhline(0.0, color="r", linestyle="--", linewidth=1.2)
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Residual", fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
            ax.legend(fontsize=10)
    
        plt.tight_layout()
        plt.show()
    
    def plot_residual_histograms(y_test, predictions_dict, bins=20, title_suffix=""):
        """
        Plot histogram of residuals for each model, side by side in subplots.
        """
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(4*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            residuals = pred_array - y_test
            ax.hist(residuals, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
            ax.set_xlabel("Residual", fontsize=12)
            ax.set_ylabel("Count",    fontsize=12)
            ax.set_title(f"{model_name} {title_suffix}", fontsize=12)
    
        plt.tight_layout()
        plt.show()
    
    
    ##############################################################################
    # 5) Main script: Compare CauchyNet and multiple FNN variants
    ##############################################################################
    if __name__ == "__main__":
        # Synthetic data
        data_size = 100
        X = torch.linspace(-1, 1, data_size)
        def compute_function(X):
            return X**2 + 7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
        Y = compute_function(X)
    
        # Scale Y
        scaler = MinMaxScaler()
        Y_np   = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
        # Build dataset + splits
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize   = int(total_size * 0.25)
        valSize    = testSize
        trainSize  = total_size - testSize - valSize
        ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
    
        # Create data loaders for *training*. 
        # For neural net training, we only need X->Y_t after scaling. So:
        # We'll do a separate approach: For the test set, we also want the original X (for plotting).
        # We'll do a second dataset for scaled Y: 
        ds_full_scaled = TensorDataset(X, torch.tensor(Y_norm, dtype=torch.float32))
        ds_train_scl, ds_val_scl, ds_test_scl = random_split(ds_full_scaled, [trainSize, valSize, testSize])
        train_loader = DataLoader(ds_train_scl, batch_size=32, shuffle=True)
        val_loader   = DataLoader(ds_val_scl,   batch_size=32, shuffle=False)
        test_loader  = DataLoader(ds_test_scl,  batch_size=32, shuffle=False)
    
        input_size  = 1
        hidden_size = 32
        output_size = 1
        lr          = 0.001
        epochs      = 800
    
        # Define multiple feedforward nets with different activations + your CauchyNet
        def cauchy_constructor(in_f, h, out_f):
            return CauchyNet(in_f, h, out_f)
    
        def fnn_relu(in_f, h, out_f):
            return FNN_ReLU(in_f, h, out_f)
    
        def fnn_sigmoid(in_f, h, out_f):
            return FNN_Sigmoid(in_f, h, out_f)
    
        def fnn_silu(in_f, h, out_f):
            return FNN_SiLU(in_f, h, out_f)
    
        models_dict = {
            "CauchyNet"    : cauchy_constructor,
            "FNN_ReLU"     : fnn_relu,
            "FNN_Sigmoid"  : fnn_sigmoid,
            "FNN_SiLU"     : fnn_silu,
        }
    
        results = {}
        print("Model\t\tMSE\t\tMAE")
        for name, constructor in models_dict.items():
            model, mse_val, mae_val, preds_all, truths_all = train_and_evaluate_model(
                constructor, name, train_loader, val_loader, test_loader,
                input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, lr=lr, epochs=epochs
            )
            print(f"{name:12s}\t{mse_val:.6f}\t{mae_val:.6f}")
            results[name] = (mse_val, mae_val, preds_all, truths_all)
    
        # 1) Bar chart of MSE
        names = list(results.keys())
        mses  = [results[n][0] for n in names]
    
        plt.figure(figsize=(5,4))
        plt.bar(names, mses, color='skyblue')
        plt.title("Comparison of MSE on Test Set", fontsize=13)
        plt.ylabel("MSE", fontsize=12)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()
    
        # Reconstruct the test set's original X, Y (unscaled) from ds_test
        # ds_test is indices of the original data (X, Y), but Y is not scaled. So let's just retrieve them:
        test_indices = ds_test.indices  # a list of test sample indices
        X_np  = X.numpy()  # shape [N]
        Y_np2 = Y.numpy()  # shape [N]
        # Just gather the test portion
        X_test_np = X_np[test_indices]   # shape = [testSize]
        Y_test_np = Y_np2[test_indices]  # shape = [testSize]
    
        # Build predictions_dict for the test set
        predictions_dict = {}
        for name, (_, _, preds_all, truths_all) in results.items():
            # 'preds_all' is scaled predictions, we have not inverse-transformed them.
            # if you'd like to invert scale: 
            # preds_all_2d = preds_all.reshape(-1,1)
            # preds_inv    = scaler.inverse_transform(preds_all_2d).flatten()
            # But if your test Y was also scaled in 'truths_all', we can directly compare in scaled space
            # or we can re-scale everything. 
            # For demonstration, let's assume 'truths_all' is also scaled -> we do scaled plots.
            predictions_dict[name] = preds_all
    
        # 3) Plot the predicted curves
        plot_prediction_curves(X_test_np, truths_all, predictions_dict, title_suffix="(Scaled)")
    
        # 4) Residual scatter plots
        plot_residuals(X_test_np, truths_all, predictions_dict, title_suffix="(Scaled)")
    
        # 5) Residual histograms
        plot_residual_histograms(truths_all, predictions_dict, bins=15, title_suffix="(Scaled)")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 38 FAILED, continuing', flush=True)

_cell[0]=39
print('=== CELL 39 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    
    SEED=10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ##########################################
    # Models and Activation
    ##########################################
    
    class ReciprocalActivation(nn.Module):
        def forward(self, x):
            epsilon = 1e-8
            return 1.0 / (x + epsilon)
    
    class CauchyNet0(nn.Module):
    
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet0, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch,1]
            batch_size, in_size = x.size()
            x_c = torch.complex(x, torch.zeros_like(x)) # [batch,1]
    
            # xi: [hidden_size], expand to [batch,hidden_size]
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1) # [batch,H]
    
            # x_c: [batch,1], to broadcast: expand [batch,H]
            x_expanded = x_c.expand(batch_size, self.hidden_size) # [batch,H]
    
            activated = self.activation(xi_expanded - x_expanded) # [batch,H]
    
            # Multiply by lambda_ [H,O]
            y = torch.matmul(activated, self.lambda_)/self.hidden_size  # [batch,O]
    
            y_real = y.real
            y_imag = y.imag
            return y_real, y_imag
    
         
    class FNN(nn.Module):
        """
        FNN:
        Input: [batch,1]
        Output: [batch,1]
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(FNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            # x: [batch,1]
            h = self.relu(self.fc1(x))
            out = self.fc2(h)
            return out
    
    class CauchyNetD(nn.Module):
        """
        CauchyNetD:
        Input: [batch, input_size=1]
        Output: y.imag => [batch, output_size=1]
    
        Steps:
        - Expand x and xi to [batch, hidden_size, output_size]
        - Apply reciprocal activation
        - Multiply by lambda_ and sum over hidden dimension
        - Return imaginary part of y
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNetD, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            angles = 2 * np.pi * torch.rand(hidden_size)
            angles, _ = torch.sort(angles)
            self.angles = nn.Parameter(angles)
    
            angle_diffs = torch.diff(angles)
            angle_diffs = torch.cat([angle_diffs, angle_diffs[-1:]], dim=0)
    
            random_values = torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat)
            scaling_factors = torch.cos(self.angles) * angle_diffs
            scaling_factors = scaling_factors[:, None].repeat(1, output_size)
            self.lambda_ = nn.Parameter(random_values * scaling_factors)
    
            self.xi = nn.Parameter(torch.normal(mean=0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat))
    
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch,1]
            batch, in_size = x.size()  # in_size=1
            x_c = torch.complex(x, torch.zeros_like(x)) # [batch,1]
    
            # xi: [hidden_size, output_size]
            # Expand to [batch,hidden_size,output_size]
            xi_expanded = self.xi.unsqueeze(0).expand(batch, -1, -1) # [batch,H,O]
    
            # x_c: [batch,1], expand to [batch,H,O]
            # first expand to [batch,H]
            x_h = x_c.expand(batch, self.hidden_size) # [batch,H]
            # then expand to O dimension
            x_ho = x_h.unsqueeze(-1).expand(batch, self.hidden_size, self.output_size) # [batch,H,O]
    
            # Activate:
            activated = self.activation(xi_expanded - x_ho) # [batch,H,O]
    
            # Multiply by lambda_ and sum over H
            # lambda_: [H,O], expand to [batch,H,O]
            lambda_exp = self.lambda_.unsqueeze(0) # [1,H,O]
            lambda_exp = lambda_exp.expand(batch, -1, -1) # [batch,H,O]
    
            y = (activated * lambda_exp).sum(dim=1) / self.hidden_size # [batch,O]
    
            # Return imaginary part
            return y.real, y.imag        
    ##########################################
    # Data Loading and Training
    ##########################################
    
    def loadData(X, Y, batchSize=64):
        # X: [N], Y: [N]
        # reshape X to [N,1]
        X = X.unsqueeze(-1) # [N,1]
    
        full_dataset = TensorDataset(X, Y)
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader,
                                 input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=20):
        model = model_class(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        best_val_loss = float('inf')
        best_state = None
        train_losses = []
        val_losses = []
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch) 
                if isinstance(out, tuple):
                    # (y_real,y_imag)
                    y_real, y_imag = out
                    # y_real,y_imag: [batch,output_size=1], shape [batch,1]
                    y_batch_expanded = y_batch.unsqueeze(-1) # [batch,1]
                    loss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                else:
                    # out: [batch,1]
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch_expanded)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
    
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                    else:
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        vloss = criterion(out, y_batch_expanded)
                    val_loss += vloss.item()
            val_loss /= len(val_loader)
    
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
    
            scheduler.step(val_loss)
        
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Plot losses
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(model_name)
        plt.legend()
        plt.show()
    
        # Test evaluation
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    preds = y_real.cpu().numpy().flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths_ = y_batch.cpu().numpy().flatten()
                predictions.append(preds)
                truths.append(truths_)
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
    
        mse = mean_squared_error(truths, predictions)
        mae = mean_absolute_error(truths, predictions)
        total_time = 0.0
        return param_count, total_time, mse, mae
    
    if __name__ == "__main__":
        data_size = 100
        X = torch.linspace(-1,1,data_size)
        def compute_function(X):
            return X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
        Y = compute_function(X)
        
        scaler = MinMaxScaler()
        Y_np = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_scaled = torch.tensor(Y_norm, dtype=torch.float32)
    
        train_loader, val_loader, test_loader = loadData(X, Y_scaled, batchSize=64)
    
        input_size = 1
        hidden_size = 32
        output_size = 1
        lr = 0.001
        epochs = 2000
    
        models_to_compare = {
            "FNN": FNN,
            "CauchyNet0": CauchyNet0,
            "Cauchy_FFN": Cauchy_FFN,
            "CauchyNetD": CauchyNetD
        }
    
        print("Model\tParameters\tTime(s)\tMSE\tMAE")
        for name, cls in models_to_compare.items():
            param_count, total_time, mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                                        input_size=input_size, hidden_size=hidden_size,
                                                                        output_size=output_size, lr=lr, epochs=epochs)
            print(f"{name}\t{param_count}\t{total_time:.2f}\t{mse:.6f}\t{mae:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 39 FAILED, continuing', flush=True)

_cell[0]=40
print('=== CELL 40 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    
    SEED=10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ##########################################
    # Models and Activation
    ##########################################
    
    class ReciprocalActivation(nn.Module):
        def forward(self, x):
            epsilon = 1e-8
            return 1.0 / (x + epsilon)
    
    class CauchyNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch,1]
            batch_size, in_size = x.size()
            x_c = torch.complex(x, torch.zeros_like(x))  # [batch,1]
    
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)  # [batch,H]
            x_expanded  = x_c.expand(batch_size, self.hidden_size)      # [batch,H]
    
            activated = self.activation(xi_expanded - x_expanded)       # [batch,H]
    
            y = torch.matmul(activated, self.lambda_)/self.hidden_size   # [batch,O]
            y_real = y.real
            y_imag = y.imag
            return y_real, y_imag
    
    
    
    
    class FNN(nn.Module):
        """
        FNN:
        Input: [batch,1]
        Output: [batch,1]
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(FNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            h = self.relu(self.fc1(x))
            out = self.fc2(h)
            return out
    
    
    
    
    ##########################################
    # Data Loading and Training
    ##########################################
    
    from torch.utils.data import TensorDataset, DataLoader, random_split
    
    def loadData(X, Y, batchSize=64):
        # X: [N], Y: [N] => reshape X-> [N,1]
        X = X.unsqueeze(-1)  # [N,1]
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_ds, val_ds, test_ds = random_split(dataset, [trainSize, valSize, testSize])
        train_loader = DataLoader(train_ds, batch_size=batchSize, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batchSize, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batchSize, shuffle=False)
        return train_loader, val_loader, test_loader
    
    
    def train_and_evaluate_model(model_constructor, name,
                                 train_loader, val_loader, test_loader,
                                 input_size=1, hidden_size=32, output_size=1,
                                 lr=0.001, epochs=100):
        """
        model_constructor: a function/lambda that accepts (input_size, hidden_size, output_size)
                           and returns an instantiated model object.
        """
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-6)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses = []
        val_losses   = []
        best_val_loss= float('inf')
        best_state   = None
    
        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)
                if isinstance(out, tuple):
                    # (y_real,y_imag)
                    y_real, y_imag = out
                    # shape [batch,1]
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                else:
                    y_batch_expanded = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch_expanded)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
    
            epoch_train_loss = running_train_loss / len(train_loader)
            train_losses.append(epoch_train_loss)
    
            # Validation
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    if isinstance(out, tuple):
                        y_real, y_imag = out
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        val_loss = criterion(y_real, y_batch_expanded) + criterion(y_imag, torch.zeros_like(y_imag))
                    else:
                        y_batch_expanded = y_batch.unsqueeze(-1)
                        val_loss = criterion(out, y_batch_expanded)
                    running_val_loss += val_loss.item()
    
            epoch_val_loss = running_val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
    
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state    = model.state_dict()
    
            scheduler.step(epoch_val_loss)
            #scheduler.step(val_loss)
            # current_lr = optimizer.param_groups[0]['lr']
            # print(f"Epoch {epoch+1}, LR={current_lr:.6f}, Train Loss={running_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}")
    
    
        # Plot training & validation curve
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses,   label='Val Loss')
        plt.title(f"{name}")
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.show()
    
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test set
    
    
        # Evaluate on test set
        model.eval()
        predictions = []
        truths      = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    preds = y_real.cpu().numpy().flatten()
                else:
                    preds = out.cpu().numpy().flatten()
                truths_ = y_batch.cpu().numpy().flatten()
                predictions.append(preds)
                truths.append(truths_)
    
        preds_all  = np.concatenate(predictions)
        truths_all = np.concatenate(truths)
    
        mse_val = mean_squared_error(truths_all, preds_all)
        mae_val = mean_absolute_error(truths_all, preds_all)
        return model, mse_val, mae_val, preds_all, truths_all
    
    
    ########################################################################################
    # Example usage
    ########################################################################################
    if __name__ == "__main__":
        data_size = 100
        X = torch.linspace(-1, 1, data_size)
        def compute_function(X):
            return X**2 + 7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
        Y = compute_function(X)
    
        # Scale the output
        scaler = MinMaxScaler()
        Y_np   = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_t    = torch.tensor(Y_norm, dtype=torch.float32)
    
        train_loader, val_loader, test_loader = loadData(X, Y_t, batchSize=64)
    
        input_size  = 1
        hidden_size = 32
        output_size = 1
        lr          = 0.001
        epochs      = 1000
        results = {}
        # Build dictionary with lambdas that accept (inp, hid, out) and return model
        models_dict = {
            "FNN":        lambda in_f, h, out_f: FNN(in_f, h, out_f),
            "CauchyNet": lambda in_f, h, out_f: CauchyNet(in_f, h, out_f)
        }
    
        print("Name\tMSE\tMAE")
        for name, constructor in models_dict.items():
            model, mse_val, mae_val, preds_all, truths_all = train_and_evaluate_model(
                constructor, name, train_loader, val_loader, test_loader,
                input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, lr=lr, epochs=epochs
            )
            print(f"{name}\t{mse_val:.6f}\t{mae_val:.6f}")
    
    
            print("Name\tMSE\tMAE")
            results[name] = (mse_val, mae_val, preds_all, truths_all)
    
        # 1) Bar chart of MSE
        plt.figure()
        model_names = list(results.keys())
        mses = [results[mn][0] for mn in model_names]
        plt.bar(model_names, mses, color='skyblue')
        plt.title("Comparison of MSE on Test Set")
        plt.ylabel("MSE")
        plt.show()
    
        # 2) Predicted-vs-True scatter for each model
        #    We'll do subplots, each showing how predictions compare to ground-truth in test set
        #    Typically, you'd want them near the 45-degree line
        fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 4), sharey=True)
        if len(model_names) == 1:
            axes = [axes]  # just to keep indexing consistent
        for ax, mn in zip(axes, model_names):
            _, _, preds_all, truths_all = results[mn]
            ax.scatter(truths_all, preds_all, alpha=0.5, label=mn)
            ax.plot([0,1],[0,1], 'r--')  # diagonal line if data is in [0,1] range after scaling
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_title(mn)
            ax.legend()
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 40 FAILED, continuing', flush=True)

_cell[0]=41
print('=== CELL 41 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    
    def plot_prediction_curves(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plot each model's predicted curve vs. the true data over x_test.
    
        - x_test: 1D array of test inputs, shape [Ntest]
        - y_test: 1D array of ground-truth outputs, shape [Ntest]
        - predictions_dict: dict of {model_name: pred_array}, each pred_array shape [Ntest]
        - title_suffix: optional string to append to each plot title
        """
        # First, we want to ensure consistent ordering of the data points by x_test
        # so that the line plots make sense.
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        plt.figure(figsize=(7,5))
        plt.plot(x_sorted, y_sorted, "k-", label="True", linewidth=2)
    
        # Plot each model
        for model_name, pred_array in predictions_dict.items():
            pred_sorted = pred_array[sort_idx]
            plt.plot(x_sorted, pred_sorted, "--", label=model_name)
    
        plt.title(f"Prediction Curves {title_suffix}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
    
    def plot_residuals(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plot each model's residuals = (pred - y_test) vs. x_test in subplots.
        This visually shows how each model's error behaves across x.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        # We'll create subplots, one for each model
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            pred_sorted = pred_array[sort_idx]
            residuals = pred_sorted - y_sorted
            ax.scatter(x_sorted, residuals, alpha=0.5)
            ax.axhline(0.0, color="r", linestyle="--")
            ax.set_xlabel("x")
            ax.set_ylabel("Residual (pred - true)")
            ax.set_title(f"{model_name} Residuals {title_suffix}")
    
        plt.tight_layout()
        plt.show()
    
    def plot_residual_histograms(y_test, predictions_dict, bins=20, title_suffix=""):
        """
        Plot histogram of residuals for each model, side by side in subplots.
        This highlights the distribution (spread, bias) of errors.
        """
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            residuals = pred_array - y_test
            ax.hist(residuals, bins=bins, alpha=0.7, color="skyblue", edgecolor="k")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{model_name} Residual Hist {title_suffix}")
    
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 41 FAILED, continuing', flush=True)

_cell[0]=42
print('=== CELL 42 ===', flush=True)
try:
    
    def gather_predictions(results):
        """
        From the final 'results' dictionary that your train_and_evaluate returns,
        extract each model's predictions (preds_all) and the ground truth (truths_all).
        We assume all models have the same test set, so we can use any model's truths_all as reference.
        
        Returns:
          predictions_dict: {model_name: 1D np.array of shape [Ntest]}
          y_test_np:        The ground-truth test array of shape [Ntest]
        """
        # We'll pick one model to retrieve the ground truths from (they're the same across all models).
        # For convenience, pick the first item in results.
        any_model_name = next(iter(results.keys()))
        # truths_all is index [3]
        y_test_np = results[any_model_name][3]
    
        # Construct dictionary of predictions
        predictions_dict = {}
        for model_name, (mse_val, mae_val, preds_all, truths_all) in results.items():
            predictions_dict[model_name] = preds_all  # 1D numpy array
        
        return predictions_dict, y_test_np
    
    
    ###############################
    # Example usage with your code
    ###############################
    def main_plots_example():
    
    
        # 1) Gather all model predictions + the ground-truth from results
        predictions_dict, y_test_np = gather_predictions(results)
    
        # 2) Suppose we also have x_test_np for plotting
        #    If you have your test set sorted or not, we can handle it either way.
        #    For demonstration, let's mock an x_test_np array:
        x_test_np = np.array([-1.0, -0.8, -0.7, -0.6, -0.5, -0,4, -0,1,0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0])  # shape [Ntest=3]
    
        # 3) Now call the custom plotting functions we wrote previously
        #    (You might have them in a separate file or above in the same script.)
    
        # Example: plot_prediction_curves
        plot_prediction_curves(x_test_np, y_test_np, predictions_dict, title_suffix="(Test Set)")
    
        # Example: plot_residuals
        plot_residuals(x_test_np, y_test_np, predictions_dict, title_suffix="(Test Set)")
    
        # Example: plot_residual_histograms
        plot_residual_histograms(y_test_np, predictions_dict, bins=5, title_suffix="(Test Set)")
    
    ###############################
    # Plotting functions (as previously shown)
    ###############################
    import matplotlib.pyplot as plt
    
    def plot_prediction_curves(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plots each model's predicted curve vs. the true data over x_test.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        plt.figure(figsize=(7,5))
        plt.plot(x_sorted, y_sorted, "k-", label="True", linewidth=2)
    
        for model_name, pred_array in predictions_dict.items():
            pred_sorted = pred_array[sort_idx]
            plt.plot(x_sorted, pred_sorted, "--", label=model_name)
    
        plt.title(f"Prediction Curves {title_suffix}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
    
    def plot_residuals(x_test, y_test, predictions_dict, title_suffix=""):
        """
        Plot each model's residuals = (pred - true) vs. x_test in subplots.
        """
        sort_idx = np.argsort(x_test)
        x_sorted = x_test[sort_idx]
        y_sorted = y_test[sort_idx]
    
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            pred_sorted = pred_array[sort_idx]
            residuals = pred_sorted - y_sorted
            ax.scatter(x_sorted, residuals, alpha=0.6)
            ax.axhline(0.0, color="r", linestyle="--")
            ax.set_xlabel("x")
            ax.set_ylabel("Residual")
            ax.set_title(f"{model_name} Residuals {title_suffix}")
    
        plt.tight_layout()
        plt.show()
    
    def plot_residual_histograms(y_test, predictions_dict, bins=20, title_suffix=""):
        """
        Plot histogram of residuals for each model, side by side in subplots.
        """
        num_models = len(predictions_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models,4), sharey=True)
        if num_models == 1:
            axes = [axes]
    
        for ax, (model_name, pred_array) in zip(axes, predictions_dict.items()):
            residuals = pred_array - y_test
            ax.hist(residuals, bins=bins, alpha=0.7, color="skyblue", edgecolor="k")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{model_name} Residual Hist {title_suffix}")
    
        plt.tight_layout()
        plt.show()
    
    ###############################
    # Actually run the example
    ###############################
    if __name__ == "__main__":
        main_plots_example()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 42 FAILED, continuing', flush=True)

_cell[0]=43
print('=== CELL 43 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ReciprocalActivation(nn.Module):
        """
        out = 1 / (x + epsilon)
        """
        def __init__(self, epsilon=1e-8):
            super().__init__()
            self.epsilon = epsilon
        
        def forward(self, x):
            return 1.0 / (x + self.epsilon)
    
    
    class Cauchy(nn.Module):
        """
        A single Cauchy layer for real inputs/outputs.
        Input:  [batch, in_features]
        Output: [batch, out_features] (real)
        """
        def __init__(self, in_features, hidden_size, out_features):
            super().__init__()
            self.in_features  = in_features
            self.hidden_size  = hidden_size
            self.out_features = out_features
    
            # Complex weights
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size, out_features), dtype=torch.cfloat)
            )
            # Complex bias-like term
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch, in_features]
            # For demonstration, sum across in_features => shape [batch,1]
            x_sum = x.sum(dim=1, keepdim=True)  # [batch,1]
            batch_size = x_sum.shape[0]
    
            # Convert to complex
            x_c = torch.complex(x_sum, torch.zeros_like(x_sum)) # [batch,1]
            x_expanded = x_c.expand(batch_size, self.hidden_size)
    
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)
            activated = self.activation(xi_expanded - x_expanded)  # [batch, hidden_size]
    
            y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size
            y_real = y_complex.real  # shape [batch, out_features]
            return y_real
    
    
    class DeepCauchyNN(nn.Module):
        """
        DeepCauchyNN with exactly ONE Cauchy layer + TWO feed-forward layers.
        => 3 transformations total, analogous to a "3-layer" network:
            1) Cauchy(in_features -> cauchy_out)
            2) Linear(cauchy_out -> hidden_layer), with ReLU
            3) Linear(hidden_layer -> out_features)
        """
        def __init__(self, in_features, cauchy_out, hidden_size, out_features):
            super().__init__()
            # 1) Single Cauchy layer
            self.cauchy = Cauchy(in_features, hidden_size=cauchy_out, out_features=cauchy_out)
    
            # 2) Fully connected layer
            self.fc1 = nn.Linear(cauchy_out, hidden_size)
            # 3) Another fully connected layer to produce final output
            self.fc2 = nn.Linear(hidden_size, out_features)
    
            self.relu = nn.ReLU()
    
        def forward(self, x):
            # pass input through the single Cauchy layer
            x_c = self.cauchy(x)  # shape [batch, cauchy_out]
    
            # pass through linear + ReLU
            x1 = self.relu(self.fc1(x_c))  # shape [batch, hidden_size]
    
            # final linear
            out = self.fc2(x1)             # shape [batch, out_features]
            return out
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 43 FAILED, continuing', flush=True)

_cell[0]=44
print('=== CELL 44 ===', flush=True)
try:
    ###########################
    # 3) FNN_3Layer
    ###########################
    class FNN_3Layer(nn.Module):
        """
        A 3-layer feedforward net with ReLU:
          [in_features -> hidden1 -> hidden2 -> out_features]
          Each hidden layer uses ReLU.
        """
        def __init__(self, in_features, hidden1, hidden2, out_features):
            super().__init__()
            self.fc1  = nn.Linear(in_features, hidden1)
            self.fc2  = nn.Linear(hidden1,     hidden2)
            self.fc3  = nn.Linear(hidden2,     out_features)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            out= self.fc3(h2)
            return out
    
    ###########################
    # 4) Data & training function
    ###########################
    def loadData(X, Y, batchSize=64):
        X = X.unsqueeze(-1)  # [N,1]
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize = int(0.25 * total_size)
        valSize  = testSize
        trainSize= total_size - testSize - valSize
        ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
    
        train_loader= DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        val_loader  = DataLoader(ds_val,   batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(ds_test,  batch_size=batchSize, shuffle=False)
    
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate(model_class, name,
                           train_loader, val_loader, test_loader,
                           in_features=1, hidden1=16, hidden2=16, out_features=1,
                           lr=0.001, epochs=300):
        """
        model_class: e.g. DeepCauchyNN or FNN_3Layer constructor.
        We'll train and test, returning (mse, mae, preds, truths).
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(in_features, hidden1, hidden2, out_features).to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses = []
        val_losses   = []
        best_val_loss= float('inf')
        best_state   = None
    
        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)  # shape [batch,1]
                y_batch_expanded = y_batch.unsqueeze(-1)
                loss = criterion(out, y_batch_expanded)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
    
            train_loss = running_train_loss / len(train_loader)
            train_losses.append(train_loss)
    
            # Validation
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    val_loss = criterion(out, y_batch.unsqueeze(-1))
                    running_val_loss += val_loss.item()
    
            val_loss_avg = running_val_loss / len(val_loader)
            val_losses.append(val_loss_avg)
    
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_state    = model.state_dict()
    
            scheduler.step(val_loss_avg)
    
        # Optionally: plot training vs val
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses,   label="Val")
        plt.title(name)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.show()
    
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test set
        model.eval()
        preds_list, truths_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                preds_list.append(out.cpu().numpy().flatten())
                truths_list.append(y_batch.cpu().numpy().flatten())
    
        preds_all  = np.concatenate(preds_list)
        truths_all = np.concatenate(truths_list)
    
        mse_val = mean_squared_error(truths_all, preds_all)
        mae_val = mean_absolute_error(truths_all, preds_all)
    
        return mse_val, mae_val, preds_all, truths_all
    
    
    ###########################
    # 5) Main comparison
    ###########################
    if __name__ == "__main__":
        # Create a synthetic dataset
        data_size = 100
        X = torch.linspace(-1,1, data_size)
        def f(x):
            return x**2 + 7*torch.cos(5*x**2) + 2*torch.sin(10*x) + 1/(x**2 + 8)
        Y = f(X)
    
        # Scale outputs
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        Y_np = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_t = torch.tensor(Y_norm, dtype=torch.float32)
    
        # Dataloaders
        train_loader, val_loader, test_loader = loadData(X, Y_t, batchSize=64)
    
        # Compare DeepCauchyNN vs a 3-layer FNN
        # We'll define hidden sizes = 16,16 for demonstration
        in_features = 1
        hidden1     = 32
        hidden2     = 16
        out_features= 1
        lr          = 0.001
        epochs      = 2000
    
        # Model constructors
        # (They must accept the same signature: (in_features, hidden1, hidden2, out_features).)
        def cauchy_net(in_f, h1, h2, out_f):
            return DeepCauchyNN(in_f, h1, h2, out_f)
    
        def ffn_3layer(in_f, h1, h2, out_f):
            return FNN_3Layer(in_f, h1, h2, out_f)
    
        # Evaluate each
        model_dict = {
            "DeepCauchyNN": cauchy_net,
            "3LayerFNN":    ffn_3layer
        }
    
        results = {}
        for name, constructor in model_dict.items():
            mse_val, mae_val, preds_all, truths_all = train_and_evaluate(
                constructor, name,
                train_loader, val_loader, test_loader,
                in_features, hidden1, hidden2, out_features,
                lr=lr, epochs=epochs
            )
            results[name] = (mse_val, mae_val, preds_all, truths_all)
    
        # Print table
        print("\nModel Comparison:")
        print("Name\t\tMSE\t\tMAE")
        for name, (mse_val, mae_val, _, _) in results.items():
            print(f"{name}\t{mse_val:.6f}\t{mae_val:.6f}")
    
        # Extra plots
        # e.g. bar chart of MSE
        import matplotlib.pyplot as plt
        names = list(results.keys())
        mses  = [results[n][0] for n in names]
        plt.bar(names, mses, color="skyblue")
        plt.title("Comparison of MSE on Test Set")
        plt.ylabel("MSE")
        plt.show()
    
        # Suppose we have X_test_np, Y_test_np for final scatter
        # We'll gather predictions from each model in a dictionary
        # for some "plot_prediction_curves" or scatter plots
        X_test_np = X.numpy()     # shape [Ntest], in [-1,1]
        # We'll just reuse truths_all from any model (they're the same)
        any_model = next(iter(results.keys()))
        _, _, _, common_truths = results[any_model]
        Y_test_np = common_truths  # shape [Ntest]
    
        # Build a dictionary of predictions
        predictions_dict = {}
        for name, (mse_val, mae_val, preds_all, truths_all) in results.items():
            predictions_dict[name] = preds_all
    
        # Then call your own plotting code
        # e.g. plot_prediction_curves, etc.
        # (not re-included here for brevity)
        # plot_prediction_curves(X_test_np, Y_test_np, predictions_dict, title_suffix="Test Set")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 44 FAILED, continuing', flush=True)

_cell[0]=45
print('=== CELL 45 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    
    class ReciprocalActivation(nn.Module):
        """
        Reciprocal activation: out = 1 / (x + epsilon).
        Avoids dividing by zero or extremely small denominators.
        """
        def __init__(self, epsilon=1e-8):
            super().__init__()
            self.epsilon = epsilon
        
        def forward(self, x):
            return 1.0 / (x + self.epsilon)
    
    
    class Cauchy(nn.Module):
        """
        A single Cauchy layer:
          - Input shape:  [batch_size, in_features]
          - Output shape: [batch_size, out_features]
        Internally:
          - We store complex parameters (lambda_ and xi).
          - The forward pass expands and does (xi - x), applies reciprocal, then multiplies by lambda_,
            and returns real output only.
        """
        def __init__(self, in_features, hidden_size, out_features):
            super().__init__()
            self.in_features = in_features
            self.hidden_size = hidden_size
            self.out_features = out_features
    
            # Complex weights: shape [hidden_size, out_features]
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size, out_features), dtype=torch.cfloat)
            )
            # Complex "bias-like" term: shape [hidden_size]
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            """
            x: [batch_size, in_features].
            We'll sum across the in_features dimension for demonstration if in_features>1,
            so we effectively treat x as a single scalar. 
            Then broadcast across hidden_size -> reciprocal -> multiply -> real out.
            """
            # Sum across the in_features dimension if in_features>1
            x_sum = x.sum(dim=1, keepdim=True)  # [batch_size,1]
            batch_size = x_sum.size(0)
    
            # Convert to complex
            x_complex = torch.complex(x_sum, torch.zeros_like(x_sum)) # [batch,1]
    
            # Expand x_complex -> shape [batch, hidden_size]
            x_expanded = x_complex.expand(-1, self.hidden_size)  # [batch,hidden_size]
    
            # Expand xi: [hidden_size] -> [batch, hidden_size]
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)
    
            # Activation: reciprocal(xi - x)
            activated = self.activation(xi_expanded - x_expanded)  # [batch, hidden_size]
    
            # Multiply by lambda_ [hidden_size, out_features], average over hidden
            y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size  # [batch, out_features]
    
            y_real = y_complex.real  # [batch, out_features]
            return y_real
    
    
    class DeepCauchyNN(nn.Module):
        """
        A 3-layer network using only Cauchy layers.
        Each layer returns a real output, which is fed into the next Cauchy layer.
    
        Example shape:
          - Layer1: Cauchy(in_features -> hidden_size_1)
          - Layer2: Cauchy(hidden_size_1 -> hidden_size_2)
          - Layer3: Cauchy(hidden_size_2 -> out_features)
        """
        def __init__(self, in_features, hidden_size1, hidden_size2, out_features):
            super().__init__()
            # 1st Cauchy layer
            self.cauchy1 = Cauchy(in_features, hidden_size1, hidden_size2)
            # 2nd Cauchy layer
            self.cauchy2 = Cauchy(hidden_size2, hidden_size2, hidden_size2)
            # 3rd Cauchy layer
            self.cauchy3 = Cauchy(hidden_size2, hidden_size2, out_features)
    
        def forward(self, x):
            # Pass through first Cauchy layer
            x1 = self.cauchy1(x)  # shape [batch, hidden_size2]
            # Pass that real output into second layer
            x2 = self.cauchy2(x1) # shape [batch, hidden_size2]
            # And final layer to produce out_features
            x3 = self.cauchy3(x2) # shape [batch, out_features]
            return x3
    
    
    #####################
    # Example usage
    #####################
    if __name__ == "__main__":
        # Suppose we have batch_size=5, in_features=3
        # We'll pass a random input into the net.
        model = DeepCauchyNN(in_features=3, hidden_size1=8, hidden_size2=8, out_features=1)
        
        # random input: shape [batch=5, in_features=3]
        x_demo = torch.randn(5, 3)
        
        # forward pass
        y_demo = model(x_demo)  # shape [5,1]
        print("Output shape:", y_demo.shape)
        print("Output (first 2 samples):\n", y_demo[:2])
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 45 FAILED, continuing', flush=True)

_cell[0]=46
print('=== CELL 46 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    ###########################
    # 1) Activation & Cauchy module
    ###########################
    class ReciprocalActivation(nn.Module):
        def __init__(self, epsilon=1e-8):
            super().__init__()
            self.epsilon = epsilon
    
        def forward(self, x):
            return 1.0 / (x + self.epsilon)
    
    class Cauchy(nn.Module):
        """
        A single Cauchy layer for real inputs/outputs:
        Input:  [batch, in_features]
        Output: [batch, out_features] (real)
        """
        def __init__(self, in_features, hidden_size, out_features):
            super().__init__()
            self.in_features  = in_features
            self.hidden_size  = hidden_size
            self.out_features = out_features
    
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size, out_features), dtype=torch.cfloat)
            )
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x shape: [batch, in_features]
            # Summation across in_features if >1, else just keep x
            x_sum = x.sum(dim=1, keepdim=True)  # [batch,1]
            batch_size = x_sum.shape[0]
    
            # Convert to complex
            x_c = torch.complex(x_sum, torch.zeros_like(x_sum)) # [batch,1]
            x_expanded = x_c.expand(batch_size, self.hidden_size)
    
            xi_expanded = self.xi.unsqueeze(0).expand(batch_size, -1)
    
            activated = self.activation(xi_expanded - x_expanded)  # [batch, hidden_size]
    
            y_complex = torch.matmul(activated, self.lambda_) / self.hidden_size
            y_real = y_complex.real
            return y_real
    
    ###########################
    # 2) DeepCauchyNN (3-layer)
    ###########################
    class DeepCauchyNN(nn.Module):
        """
        Three-layer network of consecutive Cauchy modules:
          - Cauchy(in_features -> hidden1)
          - Cauchy(hidden1 -> hidden2)
          - Cauchy(hidden2 -> out_features)
        """
        def __init__(self, in_features, hidden1, hidden2, out_features):
            super().__init__()
            self.cauchy1 = Cauchy(in_features,  hidden1, hidden2)
            self.cauchy2 = Cauchy(hidden2,      hidden2, hidden2)
            self.cauchy3 = Cauchy(hidden2,      hidden2, out_features)
    
        def forward(self, x):
            x1 = self.cauchy1(x)
            x2 = self.cauchy2(x1)
            x3 = self.cauchy3(x2)
            return x3
    
    
    ###########################
    # 3) FNN_3Layer
    ###########################
    class FNN_3Layer(nn.Module):
        """
        A 3-layer feedforward net with ReLU:
          [in_features -> hidden1 -> hidden2 -> out_features]
          Each hidden layer uses ReLU.
        """
        def __init__(self, in_features, hidden1, hidden2, out_features):
            super().__init__()
            self.fc1  = nn.Linear(in_features, hidden1)
            self.fc2  = nn.Linear(hidden1,     hidden2)
            self.fc3  = nn.Linear(hidden2,     out_features)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            out= self.fc3(h2)
            return out
    
    ###########################
    # 4) Data & training function
    ###########################
    def loadData(X, Y, batchSize=64):
        X = X.unsqueeze(-1)  # [N,1]
        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        testSize = int(0.25 * total_size)
        valSize  = testSize
        trainSize= total_size - testSize - valSize
        ds_train, ds_val, ds_test = random_split(dataset, [trainSize, valSize, testSize])
    
        train_loader= DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        val_loader  = DataLoader(ds_val,   batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(ds_test,  batch_size=batchSize, shuffle=False)
    
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate(model_class, name,
                           train_loader, val_loader, test_loader,
                           in_features=1, hidden1=16, hidden2=16, out_features=1,
                           lr=0.001, epochs=300):
        """
        model_class: e.g. DeepCauchyNN or FNN_3Layer constructor.
        We'll train and test, returning (mse, mae, preds, truths).
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(in_features, hidden1, hidden2, out_features).to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
        train_losses = []
        val_losses   = []
        best_val_loss= float('inf')
        best_state   = None
    
        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(x_batch)  # shape [batch,1]
                y_batch_expanded = y_batch.unsqueeze(-1)
                loss = criterion(out, y_batch_expanded)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
    
            train_loss = running_train_loss / len(train_loader)
            train_losses.append(train_loss)
    
            # Validation
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    val_loss = criterion(out, y_batch.unsqueeze(-1))
                    running_val_loss += val_loss.item()
    
            val_loss_avg = running_val_loss / len(val_loader)
            val_losses.append(val_loss_avg)
    
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_state    = model.state_dict()
    
            scheduler.step(val_loss_avg)
    
        # Optionally: plot training vs val
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses,   label="Val")
        plt.title(name)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.show()
    
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Evaluate on test set
        model.eval()
        preds_list, truths_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                preds_list.append(out.cpu().numpy().flatten())
                truths_list.append(y_batch.cpu().numpy().flatten())
    
        preds_all  = np.concatenate(preds_list)
        truths_all = np.concatenate(truths_list)
    
        mse_val = mean_squared_error(truths_all, preds_all)
        mae_val = mean_absolute_error(truths_all, preds_all)
    
        return mse_val, mae_val, preds_all, truths_all
    
    
    ###########################
    # 5) Main comparison
    ###########################
    if __name__ == "__main__":
        # Create a synthetic dataset
        data_size = 100
        X = torch.linspace(-1,1, data_size)
        def f(x):
            return x**2 + 7*torch.cos(5*x**2) + 2*torch.sin(10*x) + 1/(x**2 + 8)
        Y = f(X)
    
        # Scale outputs
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        Y_np = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_t = torch.tensor(Y_norm, dtype=torch.float32)
    
        # Dataloaders
        train_loader, val_loader, test_loader = loadData(X, Y_t, batchSize=64)
    
        # Compare DeepCauchyNN vs a 3-layer FNN
        # We'll define hidden sizes = 16,16 for demonstration
        in_features = 1
        hidden1     = 32
        hidden2     = 16
        out_features= 1
        lr          = 0.001
        epochs      = 2000
    
        # Model constructors
        # (They must accept the same signature: (in_features, hidden1, hidden2, out_features).)
        def cauchy_net(in_f, h1, h2, out_f):
            return DeepCauchyNN(in_f, h1, h2, out_f)
    
        def ffn_3layer(in_f, h1, h2, out_f):
            return FNN_3Layer(in_f, h1, h2, out_f)
    
        # Evaluate each
        model_dict = {
            "DeepCauchyNN": cauchy_net,
            "3LayerFNN":    ffn_3layer
        }
    
        results = {}
        for name, constructor in model_dict.items():
            mse_val, mae_val, preds_all, truths_all = train_and_evaluate(
                constructor, name,
                train_loader, val_loader, test_loader,
                in_features, hidden1, hidden2, out_features,
                lr=lr, epochs=epochs
            )
            results[name] = (mse_val, mae_val, preds_all, truths_all)
    
        # Print table
        print("\nModel Comparison:")
        print("Name\t\tMSE\t\tMAE")
        for name, (mse_val, mae_val, _, _) in results.items():
            print(f"{name}\t{mse_val:.6f}\t{mae_val:.6f}")
    
        # Extra plots
        # e.g. bar chart of MSE
        import matplotlib.pyplot as plt
        names = list(results.keys())
        mses  = [results[n][0] for n in names]
        plt.bar(names, mses, color="skyblue")
        plt.title("Comparison of MSE on Test Set")
        plt.ylabel("MSE")
        plt.show()
    
        # Suppose we have X_test_np, Y_test_np for final scatter
        # We'll gather predictions from each model in a dictionary
        # for some "plot_prediction_curves" or scatter plots
        X_test_np = X.numpy()     # shape [Ntest], in [-1,1]
        # We'll just reuse truths_all from any model (they're the same)
        any_model = next(iter(results.keys()))
        _, _, _, common_truths = results[any_model]
        Y_test_np = common_truths  # shape [Ntest]
    
        # Build a dictionary of predictions
        predictions_dict = {}
        for name, (mse_val, mae_val, preds_all, truths_all) in results.items():
            predictions_dict[name] = preds_all
    
        # Then call your own plotting code
        # e.g. plot_prediction_curves, etc.
        # (not re-included here for brevity)
        # plot_prediction_curves(X_test_np, Y_test_np, predictions_dict, title_suffix="Test Set")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 46 FAILED, continuing', flush=True)

_cell[0]=47
print('=== CELL 47 ===', flush=True)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    import torch.nn.functional as F
    
    SEED=10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ############################################################
    # Activation
    ############################################################
    class ReciprocalActivation(nn.Module):
        def forward(self, input):
            epsilon = 1e-8
            return 1.0 / (input + epsilon)
    
    ############################################################
    # Models
    ############################################################
    
    class FNN(nn.Module):
        """
        A simple feed-forward network (MLP) for sequence data:
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
    
        - Processes each timestep independently.
        - Takes the last time step's output.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(FNN, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            # If input_size > 1 and we want a single dimension, we can still process each dimension independently
            # or just keep it as is. Here we handle all features equally per time step.
    
            batch, seq_len, in_size = x.size()
            # Flatten: [batch*seq_len, input_size]
            x_flat = x.view(batch*seq_len, in_size)
            h = torch.relu(self.fc1(x_flat))  # [batch*seq_len, hidden_size]
            out = self.fc3(h)                 # [batch*seq_len, output_size]
    
            # Reshape: [batch, seq_len, output_size]
            out = out.view(batch, seq_len, self.output_size)
            # Take last timestep
            out = out[:, -1, :] # [batch, output_size]
            return out
    
    class CauchyNet(nn.Module):
        """
        CauchyNet:
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        Uses complex parameters and reciprocal activation.
        If input_size>1, we average over features first.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat))
            self.xi = nn.Parameter(torch.normal(mean=0.0, std=0.5, size=(hidden_size, output_size), dtype=torch.cfloat))
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            if self.input_size > 1:
                x = x.mean(dim=2, keepdim=True) # [batch, seq_len, 1]
    
            batch, seq_len, _ = x.size()
    
            # Convert to complex:
            # We have xi and lambda_ of shape [hidden_size, output_size].
            # We'll treat each (time step, output dim) pair.
            # Expand x: from [batch, seq_len, 1] -> [batch, seq_len, hidden_size, output_size]
            # We need hidden_size dimension. Let's first replicate x along output_size:
            z = torch.complex(x, torch.zeros_like(x)) # [batch, seq_len, 1]
            z = z.unsqueeze(3) # [batch, seq_len, 1, 1]
            z = z.expand(-1, -1, self.hidden_size, self.output_size) # [b, seq_len, hidden_size, output_size]
    
            # xi: [hidden_size, output_size] -> expand to [batch, seq_len, hidden_size, output_size]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0) # [1,1,H,O]
            xi_expanded = xi_expanded.expand(batch, seq_len, -1, -1)
    
            H = self.activation(xi_expanded - z) # [b, seq_len, H, O]
    
            # Now we multiply by lambda_ and sum over H dimension:
            # lambda_: [H,O], expand: [1,1,H,O]
            lambda_exp = self.lambda_.unsqueeze(0).unsqueeze(0) # [1,1,H,O]
            # Element-wise multiply H and lambda, sum over H
            # H * lambda_exp: [b, seq_len, H, O]
            # sum over H -> [b, seq_len, O]
            y = (H * lambda_exp).sum(dim=2)/self.hidden_size
            # y: [b, seq_len, output_size]
            
            # Take last time step and real part
            out = y.real[:, -1, :] # [batch, output_size]
            return out
    
    class LSTM(nn.Module):
        """
        Basic LSTM
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        """
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out, _ = self.lstm(x) # [batch, seq_len, hidden_size]
            out = self.fc(out[:, -1, :]) # [batch, output_size]
            return out
    
    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super(Attention, self).__init__()
            self.attention = nn.Linear(hidden_dim, 1)
    
        def forward(self, hidden_states):
            # hidden_states: [batch, seq_len, hidden_dim]
            weights = self.attention(hidden_states) # [batch, seq_len, 1]
            weights = torch.softmax(weights, dim=1)
            context = torch.sum(weights * hidden_states, dim=1) # [batch, hidden_dim]
            return context, weights
    
    class AttentionLSTM(nn.Module):
        """
        LSTM + Attention
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        """
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
            super(AttentionLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.attention = Attention(hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            lstm_out, _ = self.lstm(x) # [batch, seq_len, hidden]
            context, weights = self.attention(lstm_out)
            output = self.fc(context) # [batch, output_size]
            return output
    
    class NBeatsBlock(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NBeatsBlock, self).__init__()
            flat_input_size = input_size # input_size = seq_len*feature_count if needed
            self.fc1 = nn.Linear(flat_input_size, hidden_size)
            self.fc_out = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            # Flatten
            b, s, f = x.size()
            x_flat = x.view(b, s*f)
            x = torch.relu(self.fc1(x_flat))
            return self.fc_out(x) # [batch, output_size]
    
    class NBeats(nn.Module):
        """
        NBeats model:
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        """
        def __init__(self, input_size, hidden_size, output_size, seq_len, num_blocks=3):
            super(NBeats, self).__init__()
            self.seq_len = seq_len
            self.input_size = input_size
            self.output_size = output_size
            flat_input_size = seq_len * input_size
            self.blocks = nn.ModuleList([NBeatsBlock(flat_input_size, hidden_size, output_size) for _ in range(num_blocks)])
        def forward(self, x):
            residual = x
            out = None
            for block in self.blocks:
                out = block(residual) # [batch, output_size]
                # Not performing residual decomposition for simplicity
            return out
    
    class Informer(nn.Module):
        """
        Informer:
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        """
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=2):
            super(Informer, self).__init__()
            if num_heads % 2 != 0:
                raise ValueError("num_heads must be even")
            
            self.input_expand = nn.Linear(input_size, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, src):
            # src: [batch, seq_len, input_size]
            src = self.input_expand(src) # [batch, seq_len, hidden_size]
            memory = self.transformer_encoder(src)
            out = self.fc(memory[:, -1, :]) # [batch, output_size]
            return out
    
    ############################################################
    # Data Loading and Training functions
    ############################################################
    
    def loadData(X, Y, lookBack=10, batchSize=64):
        data = []
        for i in range(len(Y)-lookBack):
            seq = Y[i:i+lookBack]
            label = Y[i+lookBack]
            data.append((seq, label))
        sequences, labels = zip(*data)
        sequences = torch.stack(sequences) # [N, lookBack]
        labels = torch.stack(labels)       # [N]
    
        # Add input_size=1 dimension: [N, lookBack, 1]
        sequences = sequences.unsqueeze(-1) # [N, lookBack, 1]
    
        full_dataset = TensorDataset(sequences, labels)
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader,
                                 seq_len=10, input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=20):
        model = model_class(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        best_val_loss = float('inf')
        best_state = None
    
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # x_batch: [batch, seq_len, 1]
                # y_batch: [batch]
                
                optimizer.zero_grad()
                out = model(x_batch) # [batch, output_size]
                if output_size == 1:
                    y_batch = y_batch.unsqueeze(-1) # [batch,1]
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
    
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch) # [batch, output_size]
                    if output_size == 1:
                        y_batch = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
    
        if best_state is not None:
            model.load_state_dict(best_state)
    
        # Test evaluation
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch) # [batch, output_size]
                preds = out.cpu().numpy()
                if output_size == 1:
                    preds = preds.squeeze(-1) # [batch]
                    truths_ = y_batch.cpu().numpy()
                else:
                    truths_ = y_batch.cpu().numpy()
                predictions.append(preds)
                truths.append(truths_)
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
    
        mse = mean_squared_error(truths, predictions)
        mae = mean_absolute_error(truths, predictions)
        total_time = 0.0  # If needed, track time.
    
        return param_count, total_time, mse, mae
    
    ############################################################
    # Example usage
    ############################################################
    if __name__ == "__main__":
        # Example Data
        data_size = 300
        lookBack = 10
        X = torch.linspace(-1, 1, data_size)
        def compute_function(X):
            return X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
        Y = compute_function(X)
        
        scaler = MinMaxScaler()
        Y_np = Y.numpy().reshape(-1,1)
        Y_norm = scaler.fit_transform(Y_np).reshape(-1)
        Y_scaled = torch.tensor(Y_norm, dtype=torch.float32)
    
        train_loader, val_loader, test_loader = loadData(X, Y_scaled, lookBack=lookBack, batchSize=64)
    
        input_size = 1
        hidden_size = 32
        output_size = 1
        lr = 0.001
        epochs = 20
    
        models_to_compare = {
            "FNN": FNN,
            "CauchyNet": CauchyNet,
            "LSTM": LSTM,
            "NBeats": lambda i,h,o: NBeats(i,h,o,seq_len=lookBack), 
            "Informer": Informer,
            "AttentionLSTM": AttentionLSTM
        }
    
        print("Model\tParameters\tTime(s)\tMSE\tMAE")
        for name, cls in models_to_compare.items():
            if name == "NBeats":
                # NBeats requires seq_len known at init
                param_count, total_time, mse, mae = train_and_evaluate_model(
                    lambda i,h,o: NBeats(i,h,o,seq_len=lookBack),
                    name, train_loader, val_loader, test_loader,
                    seq_len=lookBack, input_size=input_size, hidden_size=hidden_size,
                    output_size=output_size, lr=lr, epochs=epochs)
            else:
                param_count, total_time, mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                                            seq_len=lookBack, input_size=input_size,
                                                                            hidden_size=hidden_size,
                                                                            output_size=output_size, lr=lr, epochs=epochs)
            print(f"{name}\t{param_count}\t{total_time:.2f}\t{mse:.6f}\t{mae:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 47 FAILED, continuing', flush=True)

_cell[0]=48
print('=== CELL 48 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    
    class ReciprocalActivation(nn.Module):
        def forward(self, input):
            epsilon = 1e-12
            return 1.0 / (input + epsilon)
    import torch
    import torch.nn as nn
    
    class FFN(nn.Module):
        """
        Feed-Forward Network:
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
    
        Processes each timestep and input feature, then takes the last timestep's output.
        """
        def __init__(self,seq_len,  input_size, hidden_size, output_size):
            super(FFN, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            # If input_size > 1 and we want to treat them uniformly, we can process each time step:
            # We'll apply the MLP to each timestep independently.
            batch, seq_len, in_size = x.size()
    
            # Flatten to [batch*seq_len, input_size]
            x_flat = x.view(batch*seq_len, in_size)
            h = torch.relu(self.fc1(x_flat))   # [batch*seq_len, hidden_size]
            out = self.fc3(h)                  # [batch*seq_len, output_size]
    
            # Reshape back to [batch, seq_len, output_size]
            out = out.view(batch, seq_len, self.output_size)
            # Take last time step
            out = out[:, -1, :]  # [batch, output_size]
            return out
    
    
    
    class CauchyNet(nn.Module):
        """
        CauchyNet:
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        Takes the last timestep and returns real output only.
        """
        def __init__(self, seq_len,input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat))
            self.xi = nn.Parameter(torch.normal(mean=0.0, std=0.5, size=(hidden_size, output_size), dtype=torch.cfloat))
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            # If input_size > 1, average features
            if self.input_size > 1:
                x = x.mean(dim=2, keepdim=True) # [batch, seq_len, 1]
    
            batch, seq_len, _ = x.size()
            
            # Convert to complex and expand to [batch, seq_len, hidden_size, output_size]
            z = torch.complex(x, torch.zeros_like(x))  # [batch, seq_len, 1]
            z = z.expand(-1, -1, self.output_size)      # [batch, seq_len, output_size]
            # Now we have z: [batch, seq_len, output_size], but we need hidden_size dimension:
            # We must clarify original logic. The original code had xi and lambda_ shaped differently.
            # Let's simplify: Instead of expanding to [hidden_size, output_size], let's reshape logic:
            # Actually, the original code had xi shape [hidden_size], lambda_ shape [hidden_size, output_size].
            # We'll revert to similar logic as previous stable version: handle per time step.
    
            # Adjusting approach:
            # We'll treat each time step as a vector to transform. Original CauchyNet was complex:
            # Let's do the same flattening approach as before:
            
            # xi: [hidden_size, output_size]
            # WAIT, original xi was [hidden_size, output_size], now it's defined as (hidden_size, output_size).
            # The user's code sets xi and lambda_ differently:
            # The given code:
            # self.xi = nn.Parameter(torch.normal(mean=0.0, std=0.5, size=(hidden_size, output_size), dtype=torch.cfloat))
            # This means xi is [hidden_size, output_size]
            # lambda_ is also [hidden_size, output_size].
            # So H = self.xi - z: but z only has output_size dimension, we need hidden_size dimension too.
    
            # Let's assume z should be: [batch, seq_len, hidden_size, output_size]
            # to match xi and lambda_ dimensions. We'll expand z accordingly.
            z = z.unsqueeze(2) # [batch, seq_len, 1, output_size]
            z = z.expand(-1, -1, self.hidden_size, -1) # [batch, seq_len, hidden_size, output_size]
            # xi: [hidden_size, output_size] -> expand to match [batch, seq_len, hidden_size, output_size]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0) # [1,1,hidden_size,output_size]
            xi_expanded = xi_expanded.expand(batch, seq_len, -1, -1)
    
            H = self.activation(xi_expanded - z) # [batch, seq_len, hidden_size, output_size]
    
            # Flatten: [batch*seq_len, hidden_size, output_size]
            H_flat = H.view(batch*seq_len, self.hidden_size, self.output_size)
            # To apply lambda: [hidden_size, output_size] on H:
            # Actually, we want to do a contraction over hidden_size dimension. 
            # H_flat: [batch*seq_len, hidden_size, output_size]
            # lambda_: [hidden_size, output_size]
            # We want a matmul over hidden_size dimension:
            # We'll do: out = (H_flat.conjugate transpose over hidden?) The original code used y = H * lambda_/hidden_size
            # Actually, we can do an element-wise multiply and sum over hidden_size dimension:
            # y = sum_over_hidden(H_flat * lambda_/hidden_size)
            
            # Let's do:
            # H_flat: [B*T, H, O]
            # lambda_: [H, O]
            # Expand lambda_ to [1,H,O]
            lambda_exp = self.lambda_.unsqueeze(0) # [1,H,O]
            # element-wise multiply and sum over hidden_size=H
            y = (H_flat * lambda_exp).sum(dim=1)/self.hidden_size # [B*T, O]
    
            # Reshape back to [batch, seq_len, output_size]
            y = y.view(batch, seq_len, self.output_size)
    
            # Take last timestep and real part
            out = y.real[:, -1, :]
            return out
    
    
    class CauchyNet0(nn.Module):
        """
        A Cauchy-based network that:
        - Takes [batch, seq_len, input_size]
        - Averages over input features if input_size > 1
        - Applies Cauchy transform per time step, producing [batch, seq_len, output_size]
        - Returns the last time step's output [batch, output_size], real-only
        """
        def __init__(self,seq_len, input_size, hidden_size, output_size):
            super(CauchyNet0, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            if self.input_size > 1:
                # Average over input features
                x = x.mean(dim=2, keepdim=True)  # [batch, seq_len, 1]
    
            batch, seq_len, _ = x.size()
            
            # Convert to complex: [batch, seq_len, 1] -> complex
            x = torch.complex(x, torch.zeros_like(x))  # [batch, seq_len, 1]
            
            # xi: [hidden_size] -> [1,1,hidden_size]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0)  # [1,1,hidden]
            
            # Apply reciprocal activation
            # activated: [batch, seq_len, hidden_size]
            activated = self.activation(xi_expanded - x)
    
            # Flatten to [batch*seq_len, hidden_size]
            activated_flat = activated.view(batch * seq_len, self.hidden_size)
    
            # Linear transform with lambda_
            y = torch.matmul(activated_flat, self.lambda_) / self.hidden_size  # [batch*seq_len, output_size]
    
            # Reshape back to [batch, seq_len, output_size]
            y = y.view(batch, seq_len, self.output_size)
    
            # Take the last time step and real part only
            out = y.real[:, -1, :]  # [batch, output_size]
            return out
    
    class Cauchy(nn.Module):
        """
        Cauchy module for stacking.
        Input: [batch, seq_len, input_size]
        Output: [batch, seq_len, hidden_size] real-valued
        """
        def __init__(self, input_size, hidden_size):
            super(Cauchy, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            if self.input_size > 1:
                x = x.mean(dim=2, keepdim=True)
    
            batch, seq_len, _ = x.size()
            x = torch.complex(x, torch.zeros_like(x)) # [batch, seq_len, 1]
    
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0)
            activated = self.activation(xi_expanded - x) # [batch, seq_len, hidden]
            
            # Return real part
            y_real = activated.real
            return y_real  # [batch, seq_len, hidden_size]
    
    class CauchyLSTM(nn.Module):
        """
        Cauchy -> LSTM -> FC
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        """
        def __init__(self,seq_len,  input_size, hidden_size, output_size, lstm_layers=1):
            super(CauchyLSTM, self).__init__()
            self.cauchy = Cauchy(input_size, hidden_size)
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            y_real = self.cauchy(x) # [batch, seq_len, hidden_size]
            lstm_out, _ = self.lstm(y_real) # [batch, seq_len, hidden_size]
            out = self.fc(lstm_out[:, -1, :]) # [batch, output_size]
            out = out.real  # Ensure output is real
            return out
    
    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super(Attention, self).__init__()
            self.attention = nn.Linear(hidden_dim, 1)
    
        def forward(self, hidden_states):
            weights = self.attention(hidden_states) # [batch, seq_len, 1]
            weights = torch.softmax(weights, dim=1)
            context = torch.sum(weights * hidden_states, dim=1) # [batch, hidden_dim]
            return context, weights
    
    class AttentionLSTM(nn.Module):
        """
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        """
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
            super(AttentionLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.attention = Attention(hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            lstm_out, _ = self.lstm(x) # [batch, seq_len, hidden]
            context, weights = self.attention(lstm_out)
            out = self.fc(context)
            return out.real  # Ensure real
    
    class AttentionCauchy(nn.Module):
        """
        Cauchy -> Attention -> FC
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        """
        def __init__(self,seq_len,  input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
            super(AttentionCauchy, self).__init__()
            self.cauchy = Cauchy(input_size, hidden_size)
            self.attention = Attention(hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            cauchy_out = self.cauchy(x) # [batch, seq_len, hidden_size]
            context, weights = self.attention(cauchy_out)
            out = self.fc(context)
            return out.real
    
    class LSTM(nn.Module):
        """
        Basic LSTM
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size]
        """
        def __init__(self, seq_len,input_size, hidden_size, output_size, num_layers=1):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out.real
    
    
    import torch.nn.functional as F
    from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    class NBeatsBlock(nn.Module):
        def __init__(self, flat_input_size, hidden_size, output_size):
            super(NBeatsBlock, self).__init__()
            self.fc1 = nn.Linear(flat_input_size, hidden_size)
            self.fc_out = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # x: [batch, flat_input_size]
            x = torch.relu(self.fc1(x))
            return self.fc_out(x)  # [batch, output_size]
    
    class NBeats(nn.Module):
        """
        NBeats model for single-step forecasting:
        - Assumes seq_len and input_size are known at init.
        - Flattens input sequence into [batch, seq_len*input_size].
        - Passes through several fully-connected blocks.
        - Returns final output [batch, output_size].
        """
        def __init__(self, seq_len, input_size, hidden_size, output_size, num_blocks=3):
            super(NBeats, self).__init__()
            self.seq_len = seq_len
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.num_blocks = num_blocks
    
            flat_input_size = seq_len * input_size
            self.blocks = nn.ModuleList([NBeatsBlock(flat_input_size, hidden_size, output_size) for _ in range(num_blocks)])
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            batch, seq_len, in_size = x.size()
    
            # Flatten seq+features
            x_flat = x.view(batch, seq_len * in_size)  # [batch, seq_len*input_size]
    
            residual = x_flat
            out = None
            for block in self.blocks:
                out = block(residual)  # [batch, output_size]
                # If implementing full N-Beats logic, you'd do residual = residual - some_function(out)
                # For simplicity, we just keep overwriting out from the last block.
    
            # Ensure output is real
            out = out.real if torch.is_complex(out) else out
            return out
    
            
    # class NBeatsBlock(nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size):
    #         super(NBeatsBlock, self).__init__()
    #         # Flatten seq and input_size into one dimension: seq_len*input_size
    #         self.fc1 = nn.Linear(input_size, hidden_size)
    #         self.fc_out = nn.Linear(hidden_size, output_size)
    
    #     def forward(self, x):
    #         # x: [batch, seq_len, input_size]
    #         batch, seq_len, in_size = x.size()
    #         x_flat = x.view(batch, seq_len*in_size)
    #         x = torch.relu(self.fc1(x_flat))
    #         return self.fc_out(x) # [batch, output_size]
    
    # class NBeats(nn.Module):
    #     """
    #     NBeats
    #     Input: [batch, seq_len, input_size]
    #     Output: [batch, output_size]
    #     """
    #     def __init__(self, input_size, hidden_size, output_size, num_blocks=1):
    #         super(NBeats, self).__init__()
    #         # input_size here is seq_len*features combined
    #         # We'll handle flattening inside forward.
    #         self.input_size = input_size
    #         self.hidden_size = hidden_size
    #         self.output_size = output_size
    #         self.num_blocks = num_blocks
    #         self.blocks = nn.ModuleList([NBeatsBlock(input_size, hidden_size, output_size) for _ in range(num_blocks)])
    
    #     def forward(self, x):
    #         # x: [batch, seq_len, input_size]
    #         # Flatten input
    #         batch, seq_len, in_size = x.size()
    #         residual = x
    #         for block in self.blocks:
    #             out = block(residual) # [batch, output_size]
    #             # For simplicity, not performing residual decomposition:
    #             # Just rely on last block's output:
    #             residual = residual # no op
    #         return out.real
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SingleLayerFFTransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dropout=0.1, activation="relu", batch_first=True):
            super(SingleLayerFFTransformerEncoderLayer, self).__init__()
            self.batch_first = batch_first
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
    
            # Single-layer feedforward: from d_model -> d_model
            self.linear_ff = nn.Linear(d_model, d_model)
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.activation = F.relu if activation == "relu" else F.gelu
    
        def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
            # 'is_causal' is a new argument passed by TransformerEncoder in PyTorch 2.0+
            # We do not need to use it here, just accept it to avoid TypeError.
            
            # Self-attention sublayer
            src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)
            src = src + self.dropout(src2)
            src = self.norm1(src)
    
            # Single-layer feedforward sublayer
            src2 = self.linear_ff(src)
            src2 = self.activation(src2)
            src2 = self.dropout(src2)
            
            src = src + src2
            src = self.norm2(src)
            return src
    
    
    class Informer(nn.Module):
        def __init__(self, seq_len, input_size, hidden_size, output_size, num_layers=1, num_heads=2):
            super(Informer, self).__init__()
            if num_heads % 2 != 0:
                raise ValueError("num_heads must be even")
            
            self.input_expand = nn.Linear(input_size, hidden_size)
    
            # Use our custom single-layer FFN transformer encoder layer
            encoder_layer = SingleLayerFFTransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dropout=0.1,
                activation="relu",
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, src):
            # src: [batch, seq_len, input_size]
            src = self.input_expand(src) # [batch, seq_len, hidden_size]
            memory = self.transformer_encoder(src)
            out = self.fc(memory[:, -1, :]) # [batch, output_size]
            return out.real if torch.is_complex(out) else out
    # Example usage:
    # Replace the original TransformerEncoderLayer in Informer with this SingleLayerFFTransformerEncoderLayer.
    # For example:
    
    # class Informer(nn.Module):
    #     def __init__(self, seq_len, input_size, hidden_size, output_size, num_layers=1, num_heads=2):
    #         super(Informer, self).__init__()
    #         if num_heads % 2 != 0:
    #             raise ValueError("num_heads must be even")
            
    #         self.input_expand = nn.Linear(input_size, hidden_size)
    
    #         # Use our custom single-layer FFN transformer encoder layer
    #         encoder_layer = SingleLayerFFTransformerEncoderLayer(
    #             d_model=hidden_size,
    #             nhead=num_heads,
    #             dropout=0.1,
    #             activation="relu",
    #             batch_first=True
    #         )
    #         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    #         self.fc = nn.Linear(hidden_size, output_size)
    
    #     def forward(self, src):
    #         # src: [batch, seq_len, input_size]
    #         src = self.input_expand(src) # [batch, seq_len, hidden_size]
    #         memory = self.transformer_encoder(src)
    #         out = self.fc(memory[:, -1, :]) # [batch, output_size]
    #         return out.real if torch.is_complex(out) else out
    
    
    # class Informer(nn.Module):
    #     def __init__(self, seq_len, input_size, hidden_size, output_size, num_layers=1, num_heads=2):
    #         super(Informer, self).__init__()
    #         if num_heads % 2 != 0:
    #             raise ValueError("num_heads must be even")
            
    #         self.input_expand = nn.Linear(input_size, hidden_size)
    #         self.encoder_layer = nn.TransformerEncoderLayer(
    #             d_model=hidden_size,
    #             nhead=num_heads,
    #             dim_feedforward=hidden_size,
    #             batch_first=True
    #         )
    #         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    #         self.fc = nn.Linear(hidden_size, output_size)
    
    #     def forward(self, src):
    #         # src: [batch, seq_len, input_size]
    #         src = self.input_expand(src) # [batch, seq_len, hidden_size]
    #         memory = self.transformer_encoder(src)
    #         out = self.fc(memory[:, -1, :]) # [batch, output_size]
    #         return out
    
    class CauchyInformer(nn.Module):
    
        def __init__(self,seq_len, input_size, hidden_size, output_size,
                     inform_num_layers=1, inform_num_heads=2):
            super(CauchyInformer, self).__init__()
            
            # Cauchy transforms input to [batch, seq_len, cauchy_hidden_size]
            self.cauchy = Cauchy(input_size, hidden_size)
            
            # Informer takes [batch, seq_len, cauchy_hidden_size] as input
            # so we set Informer's input_size = cauchy_hidden_size
            self.informer = Informer(seq_len,input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     output_size=output_size,
                                     num_layers=inform_num_layers,
                                     num_heads=inform_num_heads)
        
        def forward(self, x):
            # x: [batch, seq_len, cauchy_input_size]
            y = self.cauchy(x) # [batch, seq_len, cauchy_hidden_size]
            out = self.informer(y) # [batch, output_size]
            return out
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 48 FAILED, continuing', flush=True)

_cell[0]=49
print('=== CELL 49 ===', flush=True)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    
    SEED=10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def loadData(X, Y, lookBack=3, batchSize=64):
        # Create sequences of length lookBack
        data = []
        for i in range(len(Y)-lookBack):
            seq = Y[i:i+lookBack]   # shape: [lookBack]
            label = Y[i+lookBack]   # scalar
            data.append((seq, label))
        sequences, labels = zip(*data)  # lists
        sequences = torch.stack(sequences)  # [N, lookBack]
        labels = torch.stack(labels)        # [N]
    
        # Add input_size=1 dimension: [N, lookBack, 1]
        sequences = sequences.unsqueeze(-1) # [N, lookBack, 1]
    
        full_dataset = TensorDataset(sequences, labels)
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader,seq_len =3,
                                 input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=200):
        model = model_class(seq_len,input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        start_time = time.time()
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # x_batch: [batch, seq_len, 1]
                # y_batch: [batch]
                optimizer.zero_grad()
                out = model(x_batch)  # out: [batch, output_size]
                # If output_size=1 and y_batch is [batch], reshape y_batch for criterion
                if output_size == 1:
                    y_batch = y_batch.unsqueeze(-1) # [batch,1]
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch) # [batch, output_size]
                    if output_size == 1:
                        y_batch = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        total_time = time.time() - start_time
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch) # [batch, output_size]
                preds = out.cpu().numpy()
                truths_ = y_batch.cpu().numpy()
                predictions.append(preds)
                truths.append(truths_)
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
    
        # If output_size=1, predictions is [batch,1], squeeze it for metrics
        if output_size == 1:
            predictions = predictions.squeeze(-1)
    
        mse = mean_squared_error(truths, predictions)
        mae = mean_absolute_error(truths, predictions)
        
        return param_count, total_time, mse, mae
    
    models_to_compare = {"FFN": FFN,
        "CauchyNet0": CauchyNet0,
        "CauchyNet": CauchyNet,
        "CauchyLSTM": CauchyLSTM,
        "AttentionCauchy": AttentionCauchy,
        "CauchyInformer":CauchyInformer,
        "LSTM": LSTM,
        "Informer": Informer,
        "NBeats": NBeats
        
    }
    
    data_size = 200
    lookBack = 3
    X = torch.linspace(-1,1,data_size)
    def compute_function(X):
        return X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
    Y = compute_function(X)
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1,1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_scaled = torch.tensor(Y_norm, dtype=torch.float32)
    
    train_loader, val_loader, test_loader = loadData(X, Y_scaled, lookBack=lookBack, batchSize=64)
    
    input_size = 1
    hidden_size = 32
    output_size = 1
    lr = 0.001
    epochs = 100
    
    print("Model\tParameters\tTime(s)\tMSE\tMAE")
    for name, cls in models_to_compare.items():
        param_count, total_time, mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader, seq_len=3,
                                                                    input_size=input_size, hidden_size=hidden_size,
                                                                    output_size=output_size, lr=lr, epochs=epochs)
        print(f"{name}\t{param_count}\t{total_time:.2f}\t{mse:.6f}\t{mae:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 49 FAILED, continuing', flush=True)

_cell[0]=50
print('=== CELL 50 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    
    class FFN(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, output_size=1):
            super(FFN, self).__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.layer2 = nn.Linear(hidden_size, output_size)
            self.activation = nn.ReLU()
    
        def forward(self, x):
            # x: [batch, seq_len] similar to CauchyNet0 input
            # Unsqueeze to match CauchyNet0 handling: [batch, seq_len, 1]
            x = x.unsqueeze(-1)
            # Apply first linear + activation
            out = self.activation(self.layer1(x))  # [batch, seq_len, hidden_size]
            # Apply second linear
            out = self.layer2(out)  # [batch, seq_len, output_size]
            # If you want final output per timestep, you can return out directly.
            # If you want last timestep only:
            # out = out[:, -1, :]
            return out
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 50 FAILED, continuing', flush=True)

_cell[0]=51
print('=== CELL 51 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    
    class ReciprocalActivation(nn.Module):
        def forward(self, input):
            epsilon = 1e-8
            return 1.0 / (input + epsilon)
    
    class CauchyNet(nn.Module):
        """
        CauchyNet is an independent network that:
        - Takes input assumed to be [batch, seq_len].
        - Applies complex transformations and a final linear layer.
        - Returns (y_real, y_imag) with shape [batch, seq_len, output_size].
        
        This network stands alone and is NOT designed for stacking with other modules.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1.0, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch, seq_len]
            # Convert x to complex: [batch, seq_len, 1]
            x = torch.complex(x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1)))
            
            batch_size, seq_len, _ = x.size()
            
            # xi: [hidden_size] -> [1,1,hidden_size]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0)  # [1,1,hidden]
            
            activated = self.activation(xi_expanded - x) # [batch, seq, hidden]
            
            activated_flat = activated.view(batch_size*seq_len, self.hidden_size)
            y = torch.matmul(activated_flat, self.lambda_)/self.hidden_size
            
            y = y.view(batch_size, seq_len, self.output_size)
            return y.real, y.imag
    
    class Cauchy(nn.Module):
        """
        Cauchy is a simpler module intended for stacking with other modules (e.g., LSTM).
        Input: [batch, seq_len, input_size]
        Output: [batch, seq_len, hidden_size]
        
        No final linear layer. Just applies the Cauchy transformation and returns the real part.
        This is compatible with standard PyTorch RNN interfaces that use batch_first=True.
        """
        def __init__(self, input_size, hidden_size):
            super(Cauchy, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            if self.input_size > 1:
                x = x.mean(dim=2, keepdim=True)  # reduce input features if input_size>1
            
            batch, seq_len, _ = x.size()
            
            # x: [batch, seq_len, 1], complex
            x = torch.complex(x, torch.zeros_like(x))
            
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0)  # [1,1,hidden]
            
            # activated: [batch, seq_len, hidden]
            activated = self.activation(xi_expanded - x)
            
            y_real = activated.real
            return y_real  # [batch, seq_len, hidden_size]
    
    # Example usage:
    # CauchyNet is standalone:
    batch, seq_len = 5, 10
    input_size = 1
    hidden_size = 32
    output_size = 16
    x_example = torch.randn(batch, seq_len)
    cauchy_net = CauchyNet(input_size, hidden_size, output_size)
    y_real, y_imag = cauchy_net(x_example)  # y_real, y_imag: [batch, seq_len, output_size]
    
    # Cauchy for stacking:
    x_example2 = torch.randn(batch, seq_len, input_size)
    cauchy_module = Cauchy(input_size, hidden_size)
    y_cauchy = cauchy_module(x_example2)  # [batch, seq_len, hidden_size], can be fed into an LSTM
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 51 FAILED, continuing', flush=True)

_cell[0]=52
print('=== CELL 52 ===', flush=True)
try:
           
    
    def loadData(X, Y, model_type='CauchyNet', lookBack=1, batchSize=64):
        full_dataset = TensorDataset(X, Y)  # X: [data_size], Y: [data_size]
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader, input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=200):
        model = model_class(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        start_time = time.time()
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # x_batch: [batch_size], we want [batch_size, seq_len], seq_len=1
                x_batch = x_batch.unsqueeze(-1) # [batch_size,1]
                y_batch = y_batch.unsqueeze(-1) # [batch_size,1]
                
                optimizer.zero_grad()
                outputs_real, outputs_imag = model(x_batch) # [batch, seq=1, out=1]
                # Take last step if needed:
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                    outputs_imag = outputs_imag[:, -1, :]
                loss = criterion(outputs_real, y_batch) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    x_batch = x_batch.unsqueeze(-1)
                    y_batch = y_batch.unsqueeze(-1)
                    outputs_real, outputs_imag = model(x_batch)
                    if outputs_real.ndim == 3:
                        outputs_real = outputs_real[:, -1, :]
                        outputs_imag = outputs_imag[:, -1, :]
                    loss = criterion(outputs_real, y_batch) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        total_time = time.time() - start_time
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch = x_batch.unsqueeze(-1)
                y_batch = y_batch.unsqueeze(-1)
                outputs_real, outputs_imag = model(x_batch)
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                predictions.append(outputs_real.cpu().numpy())
                truths.append(y_batch.cpu().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        mse = mean_squared_error(truths, predictions)
        mae = mean_absolute_error(truths, predictions)
        
        return param_count, total_time, mse, mae
    
    data_sizes = [100, 200, 300]
    models_to_compare = {
        "CauchyNet": CauchyNet,
        "CauchyLSTM": CauchyLSTM,
        "AttentionCauchy":AttentionCauchy,
        "LSTM": LSTM,
        "Informer": Informer,
        "NBeats": NBeats
    }
    
    input_size = 1
    hidden_size = 32
    output_size = 1
    lr = 0.001
    epochs = 300
    
    def compute_function(X):
        return X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
    
    for data_size in data_sizes:
        # Generate data
        X = torch.linspace(-1, 1, data_size)
        Y = compute_function(X)
        # Normalize Y
        scaler = MinMaxScaler()
        Y_np = Y.numpy().reshape(-1, 1)
        Y_normalized = scaler.fit_transform(Y_np).reshape(-1)
        Y_scaled = torch.tensor(Y_normalized, dtype=torch.float32)
        
        train_loader, val_loader, test_loader = loadData(X, Y_scaled, 'CauchyNet', lookBack=1, batchSize=64)
        
        print(f"\nData Size: {data_size}")
        print("Model\tParameters\tTime(s)\tMSE\tMAE")
        for name, cls in models_to_compare.items():
            param_count, total_time, mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                                        input_size=input_size, hidden_size=hidden_size,
                                                                        output_size=output_size, lr=lr, epochs=epochs)
            print(f"{name}\t{param_count}\t{total_time:.2f}\t{mse:.6f}\t{mae:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 52 FAILED, continuing', flush=True)

_cell[0]=53
print('=== CELL 53 ===', flush=True)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    
    SEED=10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class ReciprocalActivation(nn.Module):
        def forward(self, input):
            epsilon = 1e-8
            return 1.0 / (input + epsilon)
    
    
    
    class CauchyNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            batch_size, seq_len = x.size()
            x = torch.complex(x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1)))
            
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)
            activated = self.activation(xi_expanded - x) # [batch, seq, hidden]
            
            activated_flat = activated.view(batch_size*seq_len, self.hidden_size)
            y = torch.matmul(activated_flat, self.lambda_)/self.hidden_size
            
            y = y.view(batch_size, seq_len, self.output_size)
            return y.real, y.imag
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from math import pi
    
    
    
    
    
    class Cauchy(nn.Module):
        """
        A stateless Cauchy module that processes a sequence input and produces a transformed sequence output.
        Input: [seq_len, batch, input_size]
        Output: [seq_len, batch, hidden_size]
    
        This can be stacked with other modules like LSTM since it uses the same input/output format.
        """
        def __init__(self, input_size, hidden_size):
            super(Cauchy, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [seq_len, batch, input_size]
            
            # If input_size > 1 and original logic intended for input_size=1, reduce input dimension
            if self.input_size > 1:
                x = x.mean(dim=2, keepdim=True)
            
            seq_len, batch, _ = x.size()
            
            # Transpose to [batch, seq_len]
            x = x.transpose(0,1).squeeze(-1)  # now x: [batch, seq_len]
            
            # Convert x to complex: [batch, seq_len, 1]
            x = torch.complex(x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1)))
            
            # Expand xi: [hidden_size] -> [1,1,hidden_size]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0) # [1,1,hidden]
            
            # Apply reciprocal activation: [batch, seq_len, hidden_size]
            activated = self.activation(xi_expanded - x)
            
            # Transpose back to [seq_len, batch, hidden_size]
            activated = activated.transpose(0,1)
            
            # Return only the real part (or both real and imag if desired)
            # Here we can return real part as per original pattern
            y_real = activated.real
            return y_real
    
    class CauchyLSTM(nn.Module):
        """
        Stacks a Cauchy module followed by an LSTM.
        Input: [seq_len, batch, input_size]
        Output: [seq_len, batch, lstm_hidden_size]
        """
        def __init__(self, input_size, cauchy_hidden_size, lstm_hidden_size, lstm_layers=1):
            super(CauchyLSTM, self).__init__()
            # Initialize Cauchy module
            self.cauchy = Cauchy(input_size, cauchy_hidden_size)
            
            # Initialize LSTM
            self.lstm = nn.LSTM(input_size=cauchy_hidden_size,
                                hidden_size=lstm_hidden_size,
                                num_layers=lstm_layers,
                                batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            # x: [seq_len, batch, input_size]
            
            # Pass through Cauchy
            # Cauchy returns y_real: [seq_len, batch, hidden_size]
            y_real = self.cauchy(x)
            
            # Now feed y_real into the LSTM
            # LSTM input: [seq_len, batch, cauchy_hidden_size]
            lstm_out, (h_n, c_n) = self.lstm(y_real)
    
            lstm_out = self.fc(lstm_out[:, -1, :])
            return lstm_out
    
    
    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super(Attention, self).__init__()
            self.attention = nn.Linear(hidden_dim, 1)
    
        def forward(self, hidden_states):
            weights = self.attention(hidden_states)
            weights = torch.softmax(weights, dim=1)
            context = torch.sum(weights * hidden_states, dim=1)
            return context, weights
    
    class AttentionLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
            super(AttentionLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.attention = Attention(hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            context, weights = self.attention(lstm_out)
            output = self.fc(context)
            return output
    
    class AttentionCauchy(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
            super(AttentionCauchy, self).__init__()
            self.Cauchy = Cauchy(input_size, hidden_size,batch_first=True)
            self.attention = Attention(hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            lstm_out, (h_n, c_n) = self. Cauchy(x)
            context, weights = self.attention(lstm_out)
            output = self.fc(context)
            return output
    
    
    
    
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
    
    import torch.nn.functional as F
    from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Define the N-BEATS model
    class NBeatsBlock(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NBeatsBlock, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            # self.fc2 = nn.Linear(hidden_size, hidden_size)
            # self.fc3 = nn.Linear(hidden_size, hidden_size)
            # self.fc4 = nn.Linear(hidden_size, hidden_size)
            self.fc_out = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            # x = torch.relu(self.fc2(x))
            # x = torch.relu(self.fc3(x))
            # x = torch.relu(self.fc4(x))
            return self.fc_out(x)
    
    class NBeats(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_blocks=3):
            super(NBeats, self).__init__()
            self.blocks = nn.ModuleList([NBeatsBlock(input_size, hidden_size, output_size) for _ in range(num_blocks)])
    
        def forward(self, x):
            residual = x
            for block in self.blocks:
                out = block(residual)
                residual = residual - out
            return out
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import seaborn as sns
    from tqdm import tqdm
    
    
    
    
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import time
    
    class Informer(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=2):
            super(Informer, self).__init__()
            if num_heads % 2 != 0:
                raise ValueError("num_heads must be an even number for nested tensor optimization")
            
            self.input_expand = nn.Linear(input_size, hidden_size)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                batch_first=True  # Ensure batch_first is set to True
            )
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, src):
            src = self.input_expand(src)
            memory = self.transformer_encoder(src)
            output = self.fc(memory[:, -1, :])  # Use the output of the last time step
            return output
            
    
    
            
    
    def loadData(X, Y, model_type='CauchyNet', lookBack=1, batchSize=64):
        full_dataset = TensorDataset(X, Y)  # X: [data_size], Y: [data_size]
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader, input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=200):
        model = model_class(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        start_time = time.time()
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # x_batch: [batch_size], we want [batch_size, seq_len], seq_len=1
                x_batch = x_batch.unsqueeze(-1) # [batch_size,1]
                y_batch = y_batch.unsqueeze(-1) # [batch_size,1]
                
                optimizer.zero_grad()
                outputs_real, outputs_imag = model(x_batch) # [batch, seq=1, out=1]
                # Take last step if needed:
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                    outputs_imag = outputs_imag[:, -1, :]
                loss = criterion(outputs_real, y_batch) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    x_batch = x_batch.unsqueeze(-1)
                    y_batch = y_batch.unsqueeze(-1)
                    outputs_real, outputs_imag = model(x_batch)
                    if outputs_real.ndim == 3:
                        outputs_real = outputs_real[:, -1, :]
                        outputs_imag = outputs_imag[:, -1, :]
                    loss = criterion(outputs_real, y_batch) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        total_time = time.time() - start_time
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch = x_batch.unsqueeze(-1)
                y_batch = y_batch.unsqueeze(-1)
                outputs_real, outputs_imag = model(x_batch)
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                predictions.append(outputs_real.cpu().numpy())
                truths.append(y_batch.cpu().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        mse = mean_squared_error(truths, predictions)
        mae = mean_absolute_error(truths, predictions)
        
        return param_count, total_time, mse, mae
    
    data_sizes = [100, 200, 300]
    models_to_compare = {
        "CauchyNet": CauchyNet,
        "CauchyLSTM": CauchyLSTM,
        "AttentionCauchy":AttentionCauchy,
        "LSTM": LSTM,
        "Informer": Informer,
        "NBeats": NBeats
    }
    
    input_size = 1
    hidden_size = 32
    output_size = 1
    lr = 0.001
    epochs = 300
    
    def compute_function(X):
        return X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
    
    for data_size in data_sizes:
        # Generate data
        X = torch.linspace(-1, 1, data_size)
        Y = compute_function(X)
        # Normalize Y
        scaler = MinMaxScaler()
        Y_np = Y.numpy().reshape(-1, 1)
        Y_normalized = scaler.fit_transform(Y_np).reshape(-1)
        Y_scaled = torch.tensor(Y_normalized, dtype=torch.float32)
        
        train_loader, val_loader, test_loader = loadData(X, Y_scaled, 'CauchyNet', lookBack=1, batchSize=64)
        
        print(f"\nData Size: {data_size}")
        print("Model\tParameters\tTime(s)\tMSE\tMAE")
        for name, cls in models_to_compare.items():
            param_count, total_time, mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                                        input_size=input_size, hidden_size=hidden_size,
                                                                        output_size=output_size, lr=lr, epochs=epochs)
            print(f"{name}\t{param_count}\t{total_time:.2f}\t{mse:.6f}\t{mae:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 53 FAILED, continuing', flush=True)

_cell[0]=54
print('=== CELL 54 ===', flush=True)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    
    SEED=10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Assume these classes are defined and compatible with the shapes:
    # CauchyNet, CauchyLSTM, AttentionCauchy, LSTM, Informer, NBeats
    # Each should accept (input_size, hidden_size, output_size) in __init__ and produce output [batch, output_size].
    # If they produce a sequence, we take the last step. If they use attention, they produce final output directly.
    
    def loadData(X, Y, lookBack=10, batchSize=64):
        # Create sequences of length lookBack
        data = []
        for i in range(len(Y)-lookBack):
            seq = Y[i:i+lookBack]        # shape: [lookBack]
            label = Y[i+lookBack]        # scalar
            data.append((seq, label))
        sequences, labels = zip(*data)  # lists of arrays
        sequences = torch.stack(sequences)  # [N, lookBack]
        labels = torch.stack(labels)        # [N]
    
        # Add input_size=1 dimension: [N, lookBack, 1]
        sequences = sequences.unsqueeze(-1)
    
        full_dataset = TensorDataset(sequences, labels)
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader,
                                 input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=200):
        model = model_class(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        start_time = time.time()
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # x_batch: [batch, seq_len, 1]
                # y_batch: [batch]
                optimizer.zero_grad()
                out = model(x_batch)  # out: [batch, output_size]
                y_batch = y_batch.unsqueeze(-1)  # [batch,1]
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    y_batch = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        total_time = time.time() - start_time
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch) # [batch, output_size]
                predictions.append(out.cpu().numpy())
                truths.append(y_batch.cpu().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        mse = mean_squared_error(truths, predictions)
        mae = mean_absolute_error(truths, predictions)
        
        return param_count, total_time, mse, mae
    
    def compute_function(X):
        return X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
    
    # Example dictionary of models
    models_to_compare = {
        "CauchyNet": CauchyNet,
        "CauchyLSTM": CauchyLSTM,
        "AttentionCauchy": AttentionCauchy,
        "LSTM": LSTM,
        "Informer": Informer,
        "NBeats": NBeats
    }
    
    data_size = 300
    lookBack = 10
    X = torch.linspace(-1,1,data_size)
    Y = compute_function(X)
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1,1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_scaled = torch.tensor(Y_norm, dtype=torch.float32)
    
    train_loader, val_loader, test_loader = loadData(X, Y_scaled, lookBack=lookBack, batchSize=64)
    
    input_size = 1
    hidden_size = 32
    output_size = 1
    lr = 0.001
    epochs = 50
    
    print("Model\tParameters\tTime(s)\tMSE\tMAE")
    for name, cls in models_to_compare.items():
        param_count, total_time, mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                                    input_size=input_size, hidden_size=hidden_size,
                                                                    output_size=output_size, lr=lr, epochs=epochs)
        print(f"{name}\t{param_count}\t{total_time:.2f}\t{mse:.6f}\t{mae:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 54 FAILED, continuing', flush=True)

_cell[0]=55
print('=== CELL 55 ===', flush=True)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    
    SEED=10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class ReciprocalActivation(nn.Module):
        def forward(self, input):
            epsilon = 1e-8
            return 1.0 / (input + epsilon)
    class CauchyNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            batch_size, seq_len = x.size()
            x = torch.complex(x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1)))
            
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)
            activated = self.activation(xi_expanded - x) # [batch, seq, hidden]
            
            activated_flat = activated.view(batch_size*seq_len, self.hidden_size)
            y = torch.matmul(activated_flat, self.lambda_)/self.hidden_size
            
            y = y.view(batch_size, seq_len, self.output_size)
            return y.real#, y.imag
    
    
    class Cauchy(nn.Module):
        """
        A stateless Cauchy module that processes a sequence input and produces a transformed sequence output.
        Input: [batch, seq_len, input_size]
        Output: [batch, seq_len, hidden_size]
    
        No linear layer at the end, output dimension = hidden_size.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(Cauchy, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size  # Not used, but accepted for API consistency
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            if self.input_size > 1:
                x = x.mean(dim=2, keepdim=True)
            
            batch, seq_len, _ = x.size()
            # Convert x to complex: [batch, seq_len, 1]
            x = torch.complex(x, torch.zeros_like(x))
            
            # xi: [hidden_size] -> [1,1,hidden_size]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0) 
            # Apply reciprocal activation
            activated = self.activation(xi_expanded - x)  # [batch, seq_len, hidden_size]
            
            # Return real part
            y_real = activated.real
            return y_real
    
    class CauchyLSTM(nn.Module):
        """
        Cauchy -> LSTM -> FC
        Input: [batch, seq_len, input_size]
        Output: [batch, output_size] (final time step)
        """
        def __init__(self, input_size, cauchy_hidden_size, output_size, lstm_hidden_size=32, lstm_layers=1):
            super(CauchyLSTM, self).__init__()
            self.cauchy = Cauchy(input_size, cauchy_hidden_size, output_size)
            self.lstm = nn.LSTM(input_size=cauchy_hidden_size,
                                hidden_size=lstm_hidden_size,
                                num_layers=lstm_layers,
                                batch_first=True)
            self.fc = nn.Linear(lstm_hidden_size, output_size)
        
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            y_real = self.cauchy(x)  # [batch, seq_len, cauchy_hidden_size]
            lstm_out, (h_n, c_n) = self.lstm(y_real) # [batch, seq_len, lstm_hidden_size]
            
            # Take last time step
            out = self.fc(lstm_out[:, -1, :])  # [batch, output_size]
            return out
    
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # x: [batch, seq_len, input_size]
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :]) # [batch, output_size]
            return out
    
    def loadData(X, Y, model_type='CauchyNet', lookBack=10, batchSize=64):
        # Create sequences of length lookBack
        data = []
        for i in range(len(Y)-lookBack):
            seq = Y[i:i+lookBack]        # shape: [lookBack]
            label = Y[i+lookBack]        # scalar
            data.append((seq, label))
        sequences, labels = zip(*data)  # lists of arrays
        sequences = torch.stack(sequences)  # [N, lookBack]
        labels = torch.stack(labels)        # [N]
    
        # Add input_size=1 dimension: [N,lookBack,1]
        sequences = sequences.unsqueeze(-1)
    
        full_dataset = TensorDataset(sequences, labels)
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader,
                                 input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=200):
        model = model_class(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        start_time = time.time()
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # x_batch: [batch, seq_len, 1], y_batch: [batch]
                optimizer.zero_grad()
                out = model(x_batch)  # out: [batch, output_size]
                y_batch = y_batch.unsqueeze(-1)  # [batch,1]
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    out = model(x_batch)
                    y_batch = y_batch.unsqueeze(-1)
                    loss = criterion(out, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        total_time = time.time() - start_time
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch) # [batch, output_size]
                predictions.append(out.cpu().numpy())
                truths.append(y_batch.cpu().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        mse = mean_squared_error(truths, predictions)
        mae = mean_absolute_error(truths, predictions)
        
        return param_count, total_time, mse, mae
    
    # Example usage:
    data_size = 300
    lookBack = 10
    X = torch.linspace(-1,1,data_size)
    def compute_function(X):
        return X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
    Y = compute_function(X)
    
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1,1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_scaled = torch.tensor(Y_norm, dtype=torch.float32)
    
    train_loader, val_loader, test_loader = loadData(X, Y_scaled, 'CauchyNet', lookBack=lookBack, batchSize=64)
    
    models_to_compare = {
        "CauchyNet": CauchyNet,
        "CauchyLSTM": CauchyLSTM,
        "AttentionCauchy":AttentionCauchy,
        "LSTM": LSTM,
        "Informer": Informer,
        "NBeats": NBeats}
    
    input_size = 1
    hidden_size = 32
    output_size = 1
    lr = 0.001
    epochs = 50
    
    print("Model\tParameters\tTime(s)\tMSE\tMAE")
    for name, cls in models_to_compare.items():
        param_count, total_time, mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                                    input_size=input_size, hidden_size=hidden_size,
                                                                    output_size=output_size, lr=lr, epochs=epochs)
        print(f"{name}\t{param_count}\t{total_time:.2f}\t{mse:.6f}\t{mae:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 55 FAILED, continuing', flush=True)

_cell[0]=56
print('=== CELL 56 ===', flush=True)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import random
    
    SEED=10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class ReciprocalActivation(nn.Module):
        def forward(self, input):
            epsilon = 1e-8
            return 1.0 / (input + epsilon)
    
    # Assume we have the following models defined from previous code:
    # CauchyNet0, CauchyNet1, CauchyNet_t, CauchyNet_l, Cauchy_t1
    # with consistent input-output shapes.
    
    def loadData(X, Y, model_type='CauchyNet', lookBack=1, batchSize=64):
        full_dataset = TensorDataset(X, Y)  
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader, 
                                 input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=200):
        model = model_class(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch = x_batch.unsqueeze(-1) # [batch, 1]
                y_batch = y_batch.unsqueeze(-1) # [batch, 1]
                
                optimizer.zero_grad()
                outputs_real, outputs_imag = model(x_batch)
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                    outputs_imag = outputs_imag[:, -1, :]
                loss = criterion(outputs_real, y_batch) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    x_batch = x_batch.unsqueeze(-1)
                    y_batch = y_batch.unsqueeze(-1)
                    outputs_real, outputs_imag = model(x_batch)
                    if outputs_real.ndim == 3:
                        outputs_real = outputs_real[:, -1, :]
                        outputs_imag = outputs_imag[:, -1, :]
                    vloss = criterion(outputs_real, y_batch) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                    val_loss += vloss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch = x_batch.unsqueeze(-1)
                y_batch = y_batch.unsqueeze(-1)
                outputs_real, outputs_imag = model(x_batch)
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                predictions.append(outputs_real.cpu().numpy())
                truths.append(y_batch.cpu().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        mse = mean_squared_error(truths, predictions)
        mae = mean_absolute_error(truths, predictions)
        return mse, mae
    
    def compute_function(X):
        return X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
    
    # Define models
    from typing import Dict
    
    models_to_compare = {
        "CauchyNet0": CauchyNet0,
        "CauchyNet1": CauchyNet1,
        "CauchyNet_l": CauchyNet_l
    }
    
    data_sizes = [100, 200, 300, 400, 500]
    runs = 5
    input_size = 1
    hidden_size = 32
    output_size = 1
    lr = 0.001
    epochs = 200
    
    # Store results as: results[(model, data_size)] = {'mse_mean':..., 'mse_std':..., 'mse_runs':[...] }
    results = {}
    
    for data_size in data_sizes:
        # Generate data
        X = torch.linspace(-1, 1, data_size)
        Y = compute_function(X)
        # Normalize Y
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        Y_np = Y.numpy().reshape(-1, 1)
        Y_normalized = scaler.fit_transform(Y_np).reshape(-1)
        Y_scaled = torch.tensor(Y_normalized, dtype=torch.float32)
        
        train_loader, val_loader, test_loader = loadData(X, Y_scaled, 'CauchyNet', lookBack=1, batchSize=64)
        
        # For each model, run multiple times
        for name, cls in models_to_compare.items():
            mse_runs = []
            mae_runs = []
            # Fixed seeds for reproducibility:
            # We'll just use seeds [0,1,2,3,4] for runs.
            for seed_i in range(runs):
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                random.seed(seed_i)
                
                start_time = time.time()
                mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                    input_size=input_size, hidden_size=hidden_size,
                                                    output_size=output_size, lr=lr, epochs=epochs)
                elapsed_time = time.time() - start_time
                mse_runs.append(mse)
                mae_runs.append(mae)
            
            mse_mean = np.mean(mse_runs)
            mse_std = np.std(mse_runs)
            mae_mean = np.mean(mae_runs)
            mae_std = np.std(mae_runs)
            
            results[(name, data_size)] = {
                'mse_mean': mse_mean, 'mse_std': mse_std,
                'mae_mean': mae_mean, 'mae_std': mae_std,
                'mse_runs': mse_runs, 'mae_runs': mae_runs
            }
    
    # Plot for each dataset: bar plot with error bars (MSE ± std)
    for data_size in data_sizes:
        model_names = []
        mse_means = []
        mse_stds = []
        for name in models_to_compare.keys():
            model_names.append(name)
            mse_means.append(results[(name, data_size)]['mse_mean'])
            mse_stds.append(results[(name, data_size)]['mse_std'])
        
        x = np.arange(len(model_names))
        plt.figure(figsize=(8,4))
        plt.bar(x, mse_means, yerr=mse_stds, capsize=5, alpha=0.7)
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.ylabel("MSE")
        plt.title(f"MSE (Mean ± Std) for data_size={data_size}")
        plt.tight_layout()
        plt.show()
    
        print(f"\nData Size: {data_size}")
        print("Model\tParameters\tTime(s)\tMSE\tMAE")
        for name, cls in models_to_compare.items():
            mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                                        input_size=input_size, hidden_size=hidden_size,
                                                                        output_size=output_size, lr=lr, epochs=epochs)
            print(f"{name}\t{param_count}\t{total_time:.2f}\t{mse:.6f}\t{mae:.6f}")
    
    
    # Create a circular heat map:
    # Each data_size is a layer (ring), each model is a segment.
    # We'll map MSE_mean to color.
    
    model_list = list(models_to_compare.keys())
    num_models = len(model_list)
    num_data = len(data_sizes)
    
    # Create a matrix of MSE means
    mse_matrix = np.zeros((num_data, num_models))
    for i, ds in enumerate(data_sizes):
        for j, m in enumerate(model_list):
            mse_matrix[i, j] = results[(m, ds)]['mse_mean']
    
    # Circular heat map:
    # We'll use a polar plot. 
    # Radius steps for each data scenario
    # Angle steps for each model
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    
    angles = np.linspace(0, 2*np.pi, num_models, endpoint=False)
    # We will create rings: each ring corresponds to a data scenario.
    # radius steps: from radius=1 to radius=num_data
    # Each ring is divided into num_models segments.
    
    # Normalize MSE for color mapping
    mse_min, mse_max = mse_matrix.min(), mse_matrix.max()
    norm = plt.Normalize(mse_min, mse_max)
    cmap = plt.cm.viridis
    
    for i in range(num_data):
        radius = i + 1  # layer index starts at 1
        for j in range(num_models):
            # Each segment is a wedge from angles[j] to angles[j+1]
            ang_start = angles[j]
            ang_end = angles[j] + (2*np.pi/num_models)
            value = mse_matrix[i,j]
            color = cmap(norm(value))
            
            # Draw a wedge (ring segment)
            # We'll use ax.bar with width=(2*pi/num_models), height=small radial thickness
            # radius corresponds to the mid of the ring, we can keep ring thickness small
            width = 2*np.pi/num_models
            bar_height = 0.8  # ring thickness
            ax.bar(ang_start, bar_height, width=width, bottom=radius, color=color, edgecolor='black', alpha=0.9)
    
    # Set radius limits
    ax.set_ylim(0, num_data+1.5)
    
    # Remove tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label('MSE Mean')
    
    # Add annotations:
    # The number of layers = number of datasets: label each ring (layer)
    for i, ds in enumerate(data_sizes):
        ax.text(0, i+1.4, f"DS={ds}", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Label each model along outer radius
    for j, m in enumerate(model_list):
        ang = angles[j] + (angles[1]-angles[0])/2
        # place text outside at radius = num_data+1.3
        ax.text(ang, num_data+1.3, m, ha='center', va='center', fontsize=10, rotation=np.degrees(-ang+np.pi/2))
    
    plt.title("Circular Heat Map of MSE Means\n(Rings = DataSets, Segments = Models)", y=1.1, fontsize=14)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 56 FAILED, continuing', flush=True)

_cell[0]=57
print('=== CELL 57 ===', flush=True)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import random
    
    # Assume models, train_and_evaluate_model, and compute_function are defined
    
    models_to_compare = {
        "CauchyNet0": CauchyNet0,
        "CauchyNet1": CauchyNet1,
        "CauchyNet_l": CauchyNet_l
    }
    
    data_sizes = [100, 200, 300, 400, 500]
    runs = 5
    input_size = 1
    hidden_size = 32
    output_size = 1
    lr = 0.001
    epochs = 200
    
    results = {}
    
    for data_size in data_sizes:
        # Generate data
        X = torch.linspace(-1, 1, data_size)
        Y = compute_function(X)
        # Normalize Y
        Y_np = Y.numpy().reshape(-1, 1)
        scaler = MinMaxScaler()
        Y_normalized = scaler.fit_transform(Y_np).reshape(-1)
        Y_scaled = torch.tensor(Y_normalized, dtype=torch.float32)
        
        # Load data
        train_loader, val_loader, test_loader = loadData(X, Y_scaled, 'CauchyNet', lookBack=1, batchSize=64)
        
        # Run multiple times for each model
        for name, cls in models_to_compare.items():
            mse_runs = []
            mae_runs = []
            for seed_i in range(runs):
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                random.seed(seed_i)
                
                mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                    input_size=input_size, hidden_size=hidden_size,
                                                    output_size=output_size, lr=lr, epochs=epochs)
                mse_runs.append(mse)
                mae_runs.append(mae)
            
            mse_mean = np.mean(mse_runs)
            mse_std = np.std(mse_runs)
            mae_mean = np.mean(mae_runs)
            mae_std = np.std(mae_runs)
            
            results[(name, data_size)] = {
                'mse_mean': mse_mean, 'mse_std': mse_std,
                'mae_mean': mae_mean, 'mae_std': mae_std,
                'mse_runs': mse_runs, 'mae_runs': mae_runs
            }
    
    # Plot for each dataset: log scale bar plot with error bars (MSE ± std)
    for data_size in data_sizes:
        model_names = []
        mse_means = []
        mse_stds = []
        for name in models_to_compare.keys():
            model_names.append(name)
            mse_means.append(results[(name, data_size)]['mse_mean'])
            mse_stds.append(results[(name, data_size)]['mse_std'])
        
        # Log transform for plotting
        # Add a small offset to avoid log(0)
        offset = 1e-8
        log_mse_means = np.log10(np.array(mse_means) + offset)
        # For std in log space, approximate by std of log(x + offset)
        # A simple approach: We won't do a full log error propagation.
        # Just show variation in log scale by also taking log10(mse ± std) ~ It's tricky.
        # We'll just plot error bars in the linear domain or omit error bars in log domain.
        # If we must show error bars, it's best to do them in linear space.
        # Let's do error bars in linear space but plot log scale on y-axis.
    
        x = np.arange(len(model_names))
        plt.figure(figsize=(8,4))
        # Plot linear error bars not straightforward in log scale bar,
        # We'll just plot bars and then set yscale='log' to show log scale of MSE.
        plt.bar(x, mse_means, yerr=mse_stds, capsize=5, alpha=0.7)
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.ylabel("MSE (log scale)")
        plt.yscale('log')  # log scale on the y-axis
        plt.title(f"MSE (Mean ± Std, log scale) for data_size={data_size}")
        plt.tight_layout()
        plt.show()
    
    # Create a circular heat map:
    # Each ring = data_size (scenario)
    # Each segment = model
    # We'll use log scale for MSE again.
    
    model_list = list(models_to_compare.keys())
    num_models = len(model_list)
    num_data = len(data_sizes)
    
    mse_matrix = np.zeros((num_data, num_models))
    for i, ds in enumerate(data_sizes):
        for j, m in enumerate(model_list):
            mse_matrix[i, j] = results[(m, ds)]['mse_mean']
    
    # Log transform MSE for the circular plot
    offset = 1e-8
    mse_log = np.log10(mse_matrix + offset)
    mse_min, mse_max = mse_log.min(), mse_log.max()
    
    norm = plt.Normalize(mse_min, mse_max)
    cmap = plt.cm.viridis
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    angles = np.linspace(0, 2*np.pi, num_models, endpoint=False)
    width = 2*np.pi/num_models
    bar_height = 0.8
    
    for i in range(num_data):
        radius = i + 1
        ring_values = mse_log[i,:]
        for j in range(num_models):
            val_log = ring_values[j]
            color = cmap(norm(val_log))
            ax.bar(angles[j], bar_height, width=width, bottom=radius, color=color, edgecolor='black', alpha=0.9)
    
    ax.set_ylim(0, num_data+1.5)
    
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label('log10(MSE)', fontsize=10)
    
    # Label each ring (data_size)
    for i, ds in enumerate(data_sizes):
        r_pos = i+1 + bar_height/2
        ax.text(np.pi, r_pos, f"DS={ds}", ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    # Label each model around outer radius
    outer_radius = num_data + 1.3
    for j, m in enumerate(model_list):
        ang = angles[j] + width/2
        ax.text(ang, outer_radius, m, ha='center', va='center', fontsize=9, rotation=np.degrees(-ang+np.pi/2))
    
    plt.title("Circular Heat Map of log10(MSE) Means\n(Rings = DataSets, Segments = Models)", y=1.08, fontsize=14)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 57 FAILED, continuing', flush=True)

_cell[0]=58
print('=== CELL 58 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Example assumptions:
    # model_list = ["CauchyNet0","CauchyNet1","CauchyNet_l"]
    # data_sizes = [100, 200, 300, 400, 500]
    # mse_matrix shape: [num_data, num_models]
    
    # Using log10(MSE)
    offset = 1e-8
    mse_log = np.log10(mse_matrix + offset)
    
    num_data = len(data_sizes)
    num_models = len(model_list)
    
    mse_min, mse_max = mse_log.min(), mse_log.max()
    norm = Normalize(vmin=mse_min, vmax=mse_max)
    cmap = cm.get_cmap("cividis")
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    
    # Remove all axes lines and ticks for a clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    angles = np.linspace(0, 2*np.pi, num_models, endpoint=False)
    width = 2*np.pi / num_models
    bar_height = 0.8
    
    # Draw each ring (data scenario)
    for i in range(num_data):
        radius = i + 1
        ring_values = mse_log[i,:]
        for j in range(num_models):
            val_log = ring_values[j]
            color = cmap(norm(val_log))
            ax.bar(angles[j], bar_height, width=width, bottom=radius, color=color, edgecolor='white', alpha=1.0, linewidth=1)
    
    # Set radius limit
    ax.set_ylim(0, num_data + 1.5)
    
    # Add a colorbar on the right
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.08)
    cbar.set_label('log$_{10}$(MSE)', fontsize=12)
    
    # Increase font sizes and set professional style
    plt.rcParams.update({'font.size': 12, 'font.family':'sans-serif'})
    
    # Label each ring (dataset scenario) on the left (angle = pi)
    # Place them slightly to the left outside the ring
    for i, ds in enumerate(data_sizes):
        r_pos = i+1 + bar_height/2
        ax.text(np.pi, r_pos, f"DS={ds}", ha='center', va='center', fontsize=12, fontweight='bold', color='black')
    
    # Label each model around the outer radius
    outer_radius = num_data + 1.2
    for j, m in enumerate(model_list):
        ang = angles[j] + width/2
        # We'll rotate text so it's horizontal. 
        # To do this, convert angle to degrees and rotate text to be vertical along radius line, then shift.
        # However, to minimize rotation, let's just place them outside at an angle and rotate minimally.
        # We want the text to be vertical and readable: We'll rotate so text is oriented upright:
        rotation_deg = np.degrees(-ang + np.pi/2)
        # If rotation is large, we can ensure text stays upright by adjusting rotation.
        # Let's just ensure the text label is horizontal and doesn't rotate:
        # We'll accept a slight angle. Let's just keep rotation=0 for clean horizontal text:
        rotation_deg = 0
        ax.text(ang, outer_radius, m, ha='center', va='center', fontsize=11, rotation=rotation_deg, color='black')
    
    # Add a title
    plt.title("Circular Heat Map of log10(MSE)\nRings = DataSets, Segments = Models", fontsize=14, y=1.1)
    
    # Adjust layout for a professional look
    plt.tight_layout()
    
    # Optional: Add a subtle gray circle for visual reference
    for r in range(1, num_data+1):
        circle = plt.Circle((0,0), r+bar_height, transform=ax.transData._b, color='black', fill=False, linewidth=0.5, alpha=0.5)
        ax.add_artist(circle)
    
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 58 FAILED, continuing', flush=True)

_cell[0]=59
print('=== CELL 59 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Assume model_list, data_sizes, mse_matrix are defined as previously.
    # mse_matrix: [num_data, num_models]
    # Rings = dataSets (radius), Segments = Models (angle)
    # mse_matrix[i,j] = mean MSE of dataset i and model j
    
    # Log transform MSE
    offset = 1e-8
    mse_log = np.log10(mse_matrix + offset)
    
    num_data = len(data_sizes)
    num_models = len(model_list)
    
    mse_min, mse_max = mse_log.min(), mse_log.max()
    norm = Normalize(vmin=mse_min, vmax=mse_max)
    cmap = cm.get_cmap("viridis")
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    plt.rcParams.update({'font.size': 12, 'font.family':'sans-serif'})
    
    angles = np.linspace(0, 2*np.pi, num_models, endpoint=False)
    width = 2*np.pi / num_models
    bar_height = 0.8
    
    # Function to determine text color based on background color brightness
    def get_text_color(r,g,b):
        # Compute luminance
        L = 0.299*r + 0.587*g + 0.114*b
        return 'white' if L < 0.5 else 'black'
    
    # Draw rings: each ring = a data scenario
    for i in range(num_data):
        radius = i + 1
        ring_values = mse_log[i,:]
        for j, model_name in enumerate(model_list):
            val_log = ring_values[j]
            color = cmap(norm(val_log))
            ax.bar(angles[j], bar_height, width=width, bottom=radius, color=color,
                   edgecolor='white', linewidth=1, alpha=1.0)
    
            # Compute wedge center for text placement
            ang_center = angles[j] + width/2
            radius_center = radius + bar_height/2
    
            # Determine text color for contrast
            r,g,b,a = cmap(norm(val_log))
            text_color = get_text_color(r,g,b)
    
            # Place model name at wedge center
            ax.text(ang_center, radius_center, model_name, ha='center', va='center', 
                    fontsize=10, color=text_color, rotation=0)
    
    # Set radius limit
    ax.set_ylim(0, num_data+1.5)
    
    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.08)
    cbar.set_label('log$_{10}$(MSE)', fontsize=12)
    
    # Label each data scenario on the left side
    for i, ds in enumerate(data_sizes):
        r_pos = i+1 + bar_height/2
        ax.text(np.pi, r_pos, f"DS={ds}", ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    
    plt.title("Circular Heat Map of log$_{10}$(MSE)\nRings = DataSets, Segments = Models",
              fontsize=14, y=1.08)
    
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 59 FAILED, continuing', flush=True)

_cell[0]=60
print('=== CELL 60 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Assume the following variables are defined:
    # model_list = ["CauchyNet0","CauchyNet1","CauchyNet_l"]
    # data_sizes = [100, 200, 300, 400, 500]
    # mse_matrix: shape [num_data, num_models], mse_matrix[i,j] = mean MSE
    
    # Log transform MSE
    offset = 1e-8
    mse_log = np.log10(mse_matrix + offset)
    
    num_data = len(data_sizes)
    num_models = len(model_list)
    
    mse_min, mse_max = mse_log.min(), mse_log.max()
    norm = Normalize(vmin=mse_min, vmax=mse_max)
    cmap = cm.get_cmap("plasma")
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    plt.rcParams.update({'font.size': 12, 'font.family':'sans-serif'})
    
    angles = np.linspace(0, 2*np.pi, num_models, endpoint=False)
    width = 2*np.pi / num_models
    bar_height = 0.8
    
    # We'll add more spacing between rings:
    # Instead of radius = i+1, use radius = 1 + i*(bar_height+0.5) for better separation
    def ring_radius(i):
        return 1 + i*(bar_height+0.5)
    
    # Determine text color for contrast
    def get_text_color(r, g, b):
        # Compute luminance
        L = 0.299*r + 0.587*g + 0.114*b
        return 'white' if L < 0.5 else 'black'
    
    # Draw rings: each ring = a data scenario
    for i in range(num_data):
        radius = ring_radius(i)
        ring_values = mse_log[i,:]
        for j, model_name in enumerate(model_list):
            val_log = ring_values[j]
            color = cmap(norm(val_log))
            ax.bar(angles[j], bar_height, width=width, bottom=radius, color=color,
                   edgecolor='white', linewidth=1, alpha=1.0)
    
            # Compute wedge center for text placement
            ang_center = angles[j] + width/2
            radius_center = radius + bar_height/2
    
            r,g,b,a = cmap(norm(val_log))
            text_color = get_text_color(r,g,b)
    
            # Place model name at wedge center
            # Smaller font to reduce overlap risk
            ax.text(ang_center, radius_center, model_name, ha='center', va='center', 
                    fontsize=9, color=text_color, rotation=0)
    
    # Set radius limit
    ax.set_ylim(0, ring_radius(num_data-1) + bar_height + 0.5)
    
    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.08)
    cbar.set_label('log$_{10}$(MSE)', fontsize=12)
    
    # Label each data scenario on the left side (angle=pi)
    # Place DS labels slightly inside radius to avoid overlap with wedge text
    for i, ds in enumerate(data_sizes):
        r_pos = ring_radius(i) + bar_height/2
        # DS label slightly smaller, bold
        ax.text(np.pi, r_pos, f"DS={ds}", ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    plt.title("Circular Heat Map of log$_{10}$(MSE)\nRings = DataSets, Segments = Models", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 60 FAILED, continuing', flush=True)

_cell[0]=61
print('=== CELL 61 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Assumptions:
    # model_list = ["CauchyNet0","CauchyNet1","CauchyNet_l"]
    # data_sizes = [100, 200, 300, 400, 500]
    # mse_matrix shape: [num_data, num_models], mse_matrix[i,j] = mean MSE
    
    # Log transform MSE
    offset = 1e-8
    mse_log = np.log10(mse_matrix + offset)
    
    num_data = len(data_sizes)
    num_models = len(model_list)
    
    mse_min, mse_max = mse_log.min(), mse_log.max()
    norm = Normalize(vmin=mse_min, vmax=mse_max)
    cmap = cm.get_cmap("magma")
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    plt.rcParams.update({'font.size': 12, 'font.family':'sans-serif'})
    
    angles = np.linspace(0, 2*np.pi, num_models, endpoint=False)
    width = 2*np.pi / num_models
    bar_height = 0.8
    
    # Increase spacing between rings
    def ring_radius(i):
        return 1 + i*(bar_height+0.6)
    
    # Draw rings: each ring = a data scenario
    for i in range(num_data):
        radius = ring_radius(i)
        ring_values = mse_log[i,:]
        for j in range(num_models):
            val_log = ring_values[j]
            color = cmap(norm(val_log))
            ax.bar(angles[j], bar_height, width=width, bottom=radius, color=color,
                   edgecolor='white', linewidth=1, alpha=1.0)
    
    ax.set_ylim(0, ring_radius(num_data-1) + bar_height + 0.8)
    
    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.08)
    cbar.set_label('log$_{10}$(MSE)', fontsize=12)
    
    # Label each data scenario (DS) on the left side (angle = pi)
    for i, ds in enumerate(data_sizes):
        r_pos = ring_radius(i) + bar_height/2
        ax.text(np.pi, r_pos, f"DS={ds}", ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    
    # Label each model name once at the outer radius
    outer_radius = ring_radius(num_data-1) + bar_height + 0.4
    for j, m in enumerate(model_list):
        ang_center = angles[j] + width
        # Place model name outside at outer radius, no rotation, horizontal text
        ax.text(ang_center, outer_radius+0.3, m, ha='center', va='center', fontsize=11, color='black', rotation=0)
    
    plt.title("Circular Heat Map of log$_{10}$(MSE)\nRings = Datasets, Segments = Models", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 61 FAILED, continuing', flush=True)

_cell[0]=62
print('=== CELL 62 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Assume:
    # model_list = ["CauchyNet0","CauchyNet1","CauchyNet_l"]
    # data_sizes = [100, 200, 300, 400, 500]
    # mse_matrix shape: [num_data, num_models] with mse_matrix[i,j] = mean MSE for data_sizes[i], model_list[j]
    
    # Log transform MSE
    offset = 1e-8
    mse_log = np.log10(mse_matrix + offset)
    
    num_data = len(data_sizes)
    num_models = len(model_list)
    
    mse_min, mse_max = mse_log.min(), mse_log.max()
    norm = Normalize(vmin=mse_min, vmax=mse_max)
    cmap = cm.get_cmap("RdYlBu_r")
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    angles = np.linspace(0, 2*np.pi, num_models, endpoint=False)
    width = 2*np.pi / num_models
    bar_height = 0.8
    
    # Draw rings: each ring = a data scenario
    for i in range(num_data):
        radius = i + 1
        ring_values = mse_log[i,:]
        for j in range(num_models):
            val_log = ring_values[j]
            color = cmap(norm(val_log))
            ax.bar(angles[j], bar_height, width=width, bottom=radius, color=color,
                   edgecolor='white', linewidth=1, alpha=0.9)
    
    ax.set_ylim(0, num_data+1.5)
    
    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.08)
    cbar.set_label('log$_{10}$(MSE)', fontsize=12)
    
    plt.rcParams.update({'font.size': 12, 'font.family':'sans-serif'})
    
    # Label each data scenario (DS) on left side (angle = pi)
    for i, ds in enumerate(data_sizes):
        r_pos = i+1 + bar_height/2
        # Place label slightly inside the ring and horizontally
        ax.text(np.pi, r_pos, f"DS={ds}", ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    
    # Label each model
    # We'll place model names outside at top radius line:
    outer_radius = num_data + 1.3
    for j, m in enumerate(model_list):
        ang = angles[j] + width/2
        # We want model names to appear near the top. We'll place them close to angle=0 line.
        # Instead, let's place them evenly spaced along the top (like a legend),
        # or directly radially outward from each wedge but horizontally.
        # Here we try placing them slightly above, aligned horizontally:
        # Convert angle to Cartesian for label placement:
        # We'll place them outward with a small offset along that angle.
        # We'll just rotate text minimally (or not at all):
        x_pos = (outer_radius) * np.cos(-ang+np.pi/2)
        y_pos = (outer_radius) * np.sin(-ang+np.pi/2)
        # But we reversed direction: Let's simplify by placing them all around top:
        # Actually, let's just place them along the top circle. We'll keep angle measure:
        # We'll keep them at their angle but no rotation:
        # Angle in degrees for orientation if we wanted:
        # To keep horizontal text, rotation=0:
        # We'll just place them radially outward. 
        # If angle is near left, text might appear rotated. Let's no rotate text at all.
        
        # We'll place them slightly outside the circle at radius = num_data+1.3 along that angle:
        # Then no rotation in text, so some might appear angled around circle. 
        # Let's keep them horizontal and trust the user to identify them by position.
        x_label = (num_data+1.3)*np.cos(ang)
        y_label = (num_data+1.3)*np.sin(ang)
        ax.text(ang, num_data+1.3, m, ha='center', va='center', fontsize=11, color='black', rotation=0)
    
    plt.title("Circular Heat Map of log$_{10}$(MSE)\nRings = DataSets, Segments = Models", fontsize=14, y=1.08)
    
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 62 FAILED, continuing', flush=True)

_cell[0]=63
print('=== CELL 63 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Suppose you have:
    # model_list: list of models
    # data_sizes: list of dataset scenarios
    # mse_matrix: shape [num_models, num_data], mse_matrix[i,j] = mean MSE of model i on dataset j
    
    
    num_models = len(model_list)
    num_data = len(data_sizes)
    
    if mse_matrix.shape != (num_models, num_data):
        print("Warning: Mismatch in mse_matrix shape vs model_list/data_sizes lengths!")
        print(f"mse_matrix.shape = {mse_matrix.shape}, expected ({num_models},{num_data})")
    
    # Apply log transform to MSE values
    # Add a small value to avoid log(0)
    mse_log = np.log10(mse_matrix + 1e-8)
    
    mse_log_min, mse_log_max = mse_log.min(), mse_log.max()
    norm = Normalize(vmin=mse_log_min, vmax=mse_log_max)
    cmap = cm.viridis
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    angles = np.linspace(0, 2*np.pi, num_data, endpoint=False)
    width = 2*np.pi/num_data
    bar_height = 0.8  # thickness of each ring
    
    for i in range(num_models):
        radius = i + 1
        ring_values = mse_log[i,:]  # log values for this model
        for j in range(num_data):
            val_log = ring_values[j]
            color = cmap(norm(val_log))
            ax.bar(angles[j], bar_height, width=width, bottom=radius, color=color, edgecolor='black', alpha=0.9)
    
    # Set radius limits
    ax.set_ylim(0, num_models+1.5)
    
    # Add colorbar for log scale
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label('log10(MSE)', fontsize=10)
    
    # Label each ring (model)
    for i, model_name in enumerate(model_list):
        r_pos = i+1 + bar_height/2
        ax.text(np.pi, r_pos, model_name, ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    # Label each data scenario around the outer radius
    outer_radius = num_models + 1.3
    for j, ds in enumerate(data_sizes):
        ang = angles[j] + width/2
        ax.text(ang, outer_radius, str(ds), ha='center', va='center', fontsize=9, rotation=np.degrees(-ang+np.pi/2))
    
    plt.title("Circular Heat Map of log10(MSE) Means\n(Layers=Models, Segments=Datasets)", y=1.08, fontsize=14)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 63 FAILED, continuing', flush=True)

_cell[0]=64
print('=== CELL 64 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Assuming mse_matrix, model_list, and data_sizes are already defined,
    # and mse_matrix[i,j] = MSE mean for data_sizes[i] and model_list[j].
    # Also assuming model_list and data_sizes are lists of model names and data scenarios.
    
    # Example assumptions:
    # model_list = ["CauchyNet0","CauchyNet1","CauchyNet_t","CauchyNet_l","Cauchy_t1"]
    # data_sizes = [100,200,300,400,500]
    # mse_matrix shape: [num_data, num_models]
    
    num_models = len(model_list)
    num_data = len(data_sizes)
    
    # Create figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    
    angles = np.linspace(0, 2*np.pi, num_models, endpoint=False)
    width = 2*np.pi/num_models
    bar_height = 0.8  # thickness of each ring
    ax.set_xticks([])
    ax.set_yticks([])
    
    # We will use a list of distinct colormaps, one per dataset scenario.
    # If num_data > length of this list, add more colormaps.
    cmaps = [cm.get_cmap(name) for name in ["viridis","plasma","inferno","magma","cividis"]]
    while len(cmaps) < num_data:
        cmaps.append(cm.viridis)  # fallback if more data than cmaps
    
    # For layout of colorbars, we create a grid of axes at the top.
    # We'll place all colorbars horizontally at top.
    # We'll leave some space above for colorbars
    plt.subplots_adjust(top=0.75)
    
    # We'll keep track of colorbar axes to place them nicely
    cbar_axes = []
    cbar_height = 0.02
    cbar_width = 0.15
    cbar_spacing = 0.03
    left_start = 0.1
    for i in range(num_data):
        # Compute left position for each colorbar
        left = left_start + i*(cbar_width + cbar_spacing)
        cax = fig.add_axes([left, 0.9, cbar_width, cbar_height])
        cbar_axes.append(cax)
    
    for i in range(num_data):
        radius = i + 1
        ring_values = torch.log(mse_matrix[i,:])
        ring_min, ring_max = ring_values.min(), ring_values.max()
        norm = Normalize(vmin=ring_min, vmax=ring_max)
        cmap = cmaps[i]
    
        for j in range(num_models):
            val = ring_values[j]
            color = cmap(norm(val))
            # Draw a wedge segment as a bar
            # Use ax.bar for each segment
            ang_start = angles[j]
            ax.bar(ang_start, bar_height, width=width, bottom=radius, color=color, edgecolor='black', alpha=0.9)
    
        # Create ScalarMappable for the colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
    
        cbar = plt.colorbar(sm, cax=cbar_axes[i], orientation='horizontal')
        cbar.set_label(f'DS={data_sizes[i]} MSE', fontsize=9)
        cbar_axes[i].xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=8)
    
    # Set radius limits to fit all rings and some extra space
    ax.set_ylim(0, num_data+2)
    
    # Add ring labels on the left side (just an aesthetic choice)
    for i, ds in enumerate(data_sizes):
        # Place text to the left side (angle = 180 degrees)
        # We'll place label at angle pi (180 deg), radius=i+1 + bar_height/2
        r_pos = i+1 + bar_height/2
        ax.text(np.pi, r_pos, f"DS={ds}", ha='center', va='center', fontsize=12, fontweight='bold', color='black')
    
    # Label each model around the outer radius
    outer_radius = num_data + 1.7
    for j, m in enumerate(model_list):
        ang = angles[j] + width/2
        # Text rotated so that model names are readable
        ax.text(ang, outer_radius, m, ha='center', va='center', fontsize=9, rotation=np.degrees(-ang+np.pi/2))
    
    plt.title("Circular Heat Map of MSE Means\nEach Ring = Dataset, Each Segment = Model\nDistinct Colorbar per Dataset", y=1.05, fontsize=14)
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 64 FAILED, continuing', flush=True)

_cell[0]=65
print('=== CELL 65 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Assume we have:
    # model_list: list of model names, length = num_models
    # data_sizes: list of datasets, length = num_data
    # mse_matrix: shape [num_models, num_data], mse_matrix[i,j] = mean MSE of model i on dataset j
    
    num_models = len(model_list)
    num_data = len(data_sizes)
    
    # We'll create a polar plot:
    # Radius layers: models
    # Angular segments: data scenarios
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    angles = np.linspace(0, 2*np.pi, num_data, endpoint=False)
    width = 2*np.pi/num_data
    bar_height = 0.8  # thickness of each ring
    
    # Global normalization based on entire mse_matrix
    mse_min, mse_max = mse_matrix.min(), mse_matrix.max()
    norm = Normalize(vmin=mse_min, vmax=mse_max)
    cmap = cm.viridis
    
    for i in range(num_models):
        radius = i + 1
        ring_values = mse_matrix[i,:]
        for j in range(num_data):
            val = ring_values[j]
            color = cmap(norm(val))
            # draw segment:
            ang_start = angles[j]
            ax.bar(ang_start, bar_height, width=width, bottom=radius, color=color, edgecolor='black', alpha=0.9)
    
    # Set radius limits
    ax.set_ylim(0, num_models+1.5)
    
    # Add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label('MSE Mean', fontsize=10)
    
    # Label each ring (model)
    # We'll place model names on the left (angle = pi)
    for i, model_name in enumerate(model_list):
        r_pos = i+1 + bar_height/2
        ax.text(np.pi, r_pos, model_name, ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    # Label each data scenario around the outer radius
    outer_radius = num_models + 1.3
    for j, ds in enumerate(data_sizes):
        ang = angles[j] + width/2
        ax.text(ang, outer_radius, str(ds), ha='center', va='center', fontsize=9, rotation=np.degrees(-ang+np.pi/2))
    
    plt.title("Circular Heat Map of MSE Means\n(Layers=Models, Segments=Datasets)", y=1.08, fontsize=14)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 65 FAILED, continuing', flush=True)

_cell[0]=66
print('=== CELL 66 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Suppose you have:
    # model_list: list of models
    # data_sizes: list of dataset scenarios
    # mse_matrix: shape [num_models, num_data], mse_matrix[i,j] = mean MSE of model i on dataset j
    
    
    
    
    
    num_models = len(model_list)
    num_data = len(data_sizes)
    
    if mse_matrix.shape != (num_models, num_data):
        print("Warning: Mismatch in mse_matrix shape vs model_list/data_sizes lengths!")
        print(f"mse_matrix.shape = {mse_matrix.shape}, expected ({num_models},{num_data})")
    
    # Apply log transform to MSE values
    # Add a small value to avoid log(0)
    mse_log = np.log10(mse_matrix + 1e-8)
    
    mse_log_min, mse_log_max = mse_log.min(), mse_log.max()
    norm = Normalize(vmin=mse_log_min, vmax=mse_log_max)
    cmap = cm.viridis
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    angles = np.linspace(0, 2*np.pi, num_data, endpoint=False)
    width = 2*np.pi/num_data
    bar_height = 0.8  # thickness of each ring
    
    for i in range(num_models):
        radius = i + 1
        ring_values = mse_log[i,:]  # log values for this model
        for j in range(num_data):
            val_log = ring_values[j]
            color = cmap(norm(val_log))
            ax.bar(angles[j], bar_height, width=width, bottom=radius, color=color, edgecolor='black', alpha=0.9)
    
    # Set radius limits
    ax.set_ylim(0, num_models+1.5)
    
    # Add colorbar for log scale
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label('log10(MSE)', fontsize=10)
    
    # Label each ring (model)
    for i, model_name in enumerate(model_list):
        r_pos = i+1 + bar_height/2
        ax.text(np.pi, r_pos, model_name, ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    # Label each data scenario around the outer radius
    outer_radius = num_models + 1.3
    for j, ds in enumerate(data_sizes):
        ang = angles[j] + width/2
        ax.text(ang, outer_radius, str(ds), ha='center', va='center', fontsize=9, rotation=np.degrees(-ang+np.pi/2))
    
    plt.title("Circular Heat Map of log10(MSE) Means\n(Layers=Models, Segments=Datasets)", y=1.08, fontsize=14)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 66 FAILED, continuing', flush=True)

_cell[0]=67
print('=== CELL 67 ===', flush=True)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    
    SEED=10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class ReciprocalActivation(nn.Module):
        def forward(self, input):
            epsilon = 1e-8
            return 1.0 / (input + epsilon)
    
    # All models assume: input shape [batch_size, seq_len]
    # output shape [batch_size, seq_len, output_size]
    
    class CauchyNet_l(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet_l, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            batch_size, seq_len = x.size()
            # Make x complex: [batch, seq, 1]
            x = torch.complex(x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1)))
            
            lambda_expanded = self.lambda_.unsqueeze(0).unsqueeze(0)  # [1,1,hidden]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0)          # [1,1,hidden]
            
            x_expanded = x.expand(batch_size, seq_len, self.hidden_size)
            yz = (x_expanded * lambda_expanded) - xi_expanded
            
            activated = self.activation(yz) # [batch, seq, hidden]
            
            # sum over hidden
            output = torch.sum(activated, dim=2, keepdim=True) # [batch, seq, 1]
            return output.real, output.imag
    
    class Cauchy_t1(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Cauchy_t1, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            batch_size, seq_len = x.size()
            x = torch.complex(x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1)))
            
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0)  
            lambda_expanded = self.lambda_.unsqueeze(0).unsqueeze(0)
            
            x_expanded = x.expand(batch_size, seq_len, self.hidden_size)
            diff = x_expanded - xi_expanded
            diff_lambda = diff * lambda_expanded
            activated = self.activation(diff_lambda)
            
            # sum over hidden
            output = torch.sum(activated, dim=2, keepdim=True) # [batch, seq, 1]
            return output.real, output.imag
    
    class CauchyNet_t(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet_t, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1.0, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            batch_size, seq_len = x.size()
            x = torch.complex(x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1)))
            
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)
            diff = x - xi_expanded  
            
            activated = self.activation(diff)  # [batch, seq, hidden]
            
            # Flatten for matmul
            activated_flat = activated.view(batch_size*seq_len, self.hidden_size)
            y = torch.matmul(activated_flat, self.lambda_)  # [batch*seq, output_size]
            
            y = y.view(batch_size, seq_len, self.output_size)
            return y.real, y.imag
    
    class CauchyNet0(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet0, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            batch_size, seq_len = x.size()
            x = torch.complex(x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1)))
            
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)
            activated = self.activation(xi_expanded - x) # [batch, seq, hidden]
            
            activated_flat = activated.view(batch_size*seq_len, self.hidden_size)
            y = torch.matmul(activated_flat, self.lambda_)/self.hidden_size
            
            y = y.view(batch_size, seq_len, self.output_size)
            return y.real, y.imag
    
    class CauchyNet1(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet1, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            batch_size, seq_len = x.size()
            x = torch.complex(x.unsqueeze(-1), torch.zeros_like(x.unsqueeze(-1)))
            
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)
            activated = self.activation(xi_expanded - x)
            
            activated_flat = activated.view(batch_size * seq_len, self.hidden_size)
            y = torch.matmul(activated_flat, self.lambda_)/self.hidden_size
            
            y = y.view(batch_size, seq_len, self.output_size)
            return y.real, y.imag
    
    def loadData(X, Y, model_type='CauchyNet', lookBack=1, batchSize=64):
        full_dataset = TensorDataset(X, Y)  # X: [data_size], Y: [data_size]
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader, input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=200):
        model = model_class(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        start_time = time.time()
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                # x_batch: [batch_size], we want [batch_size, seq_len], seq_len=1
                x_batch = x_batch.unsqueeze(-1) # [batch_size,1]
                y_batch = y_batch.unsqueeze(-1) # [batch_size,1]
                
                optimizer.zero_grad()
                outputs_real, outputs_imag = model(x_batch) # [batch, seq=1, out=1]
                # Take last step if needed:
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                    outputs_imag = outputs_imag[:, -1, :]
                loss = criterion(outputs_real, y_batch) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    x_batch = x_batch.unsqueeze(-1)
                    y_batch = y_batch.unsqueeze(-1)
                    outputs_real, outputs_imag = model(x_batch)
                    if outputs_real.ndim == 3:
                        outputs_real = outputs_real[:, -1, :]
                        outputs_imag = outputs_imag[:, -1, :]
                    loss = criterion(outputs_real, y_batch) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        total_time = time.time() - start_time
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch = x_batch.unsqueeze(-1)
                y_batch = y_batch.unsqueeze(-1)
                outputs_real, outputs_imag = model(x_batch)
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                predictions.append(outputs_real.cpu().numpy())
                truths.append(y_batch.cpu().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        mse = mean_squared_error(truths, predictions)
        mae = mean_absolute_error(truths, predictions)
        
        return param_count, total_time, mse, mae
    
    data_sizes = [100, 200, 300, 400, 500]
    models_to_compare = {
        "CauchyNet0": CauchyNet0,
        "CauchyNet1": CauchyNet1,
        "CauchyNet_t": CauchyNet_t,
        "CauchyNet_l": CauchyNet_l,
        "Cauchy_t1": Cauchy_t1
    }
    
    input_size = 1
    hidden_size = 32
    output_size = 1
    lr = 0.001
    epochs = 300
    
    def compute_function(X):
        return X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1/(X**2 + 8)
    
    for data_size in data_sizes:
        # Generate data
        X = torch.linspace(-1, 1, data_size)
        Y = compute_function(X)
        # Normalize Y
        scaler = MinMaxScaler()
        Y_np = Y.numpy().reshape(-1, 1)
        Y_normalized = scaler.fit_transform(Y_np).reshape(-1)
        Y_scaled = torch.tensor(Y_normalized, dtype=torch.float32)
        
        train_loader, val_loader, test_loader = loadData(X, Y_scaled, 'CauchyNet', lookBack=1, batchSize=64)
        
        print(f"\nData Size: {data_size}")
        print("Model\tParameters\tTime(s)\tMSE\tMAE")
        for name, cls in models_to_compare.items():
            param_count, total_time, mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader,
                                                                        input_size=input_size, hidden_size=hidden_size,
                                                                        output_size=output_size, lr=lr, epochs=epochs)
            print(f"{name}\t{param_count}\t{total_time:.2f}\t{mse:.6f}\t{mae:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 67 FAILED, continuing', flush=True)

_cell[0]=68
print('=== CELL 68 ===', flush=True)
try:
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import datetime
    import matplotlib.pyplot as plt
    import math, time
    from math import sqrt
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import random
    import itertools
    import datetime
    from operator import itemgetter
    import time
    import torch.optim as optim
    import time
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from tqdm import tqdm
    
    SEED=10
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Define the Reciprocal Activation with a practical epsilon
    class ReciprocalActivation(nn.Module):
        def forward(self, input):
            epsilon = 1e-8  # Small value to prevent division by zero
            return 1.0 / (input + epsilon)
    class CauchyNet_l(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet_l, self).__init__()
            self.hidden_size = hidden_size
            
            # Complex parameters: lambda and xi
            # Treating lambda and xi as linear layer parameters:
            # y = (z * lambda) - xi
            # lambda: [hidden_size], xi: [hidden_size]
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch_size, seq_len]
            batch_size, seq_len = x.size()
            
            # Convert x to complex and shape [batch_size, seq_len, 1]
            x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
            x = torch.complex(x, torch.zeros_like(x))
            
            # Expand lambda and xi for broadcasting: [1,1,hidden_size]
            lambda_expanded = self.lambda_.unsqueeze(0).unsqueeze(0)  # [1,1,hidden_size]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0)          # [1,1,hidden_size]
            
            # Compute (lambda*z - xi)
            # First broadcast x along hidden dimension
            x_expanded = x.expand(batch_size, seq_len, self.hidden_size) # [batch_size, seq_len, hidden_size]
            yz = (x_expanded * lambda_expanded) - xi_expanded            # [batch_size, seq_len, hidden_size]
            
            # Apply reciprocal activation
            activated = self.activation(yz) # [batch_size, seq_len, hidden_size]
            
            # Sum over hidden dimension
            output = torch.sum(activated, dim=2, keepdim=True) # [batch_size, seq_len, 1]
            
            return output.real, output.imag
            
    class Cauchy_t1(nn.Module):
        def __init__(self, input_size, hidden_size,output_size):
            super(Cauchy_t1, self).__init__()
            self.hidden_size = hidden_size
            
            # Complex parameters
            # xi: [hidden_size], lambda: [hidden_size]
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch_size, seq_len]
            batch_size, seq_len = x.size()
            
            # Convert x to complex and reshape to [batch_size, seq_len, 1]
            x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
            x = torch.complex(x, torch.zeros_like(x))  # Make it complex: Real part = x, Imag part = 0
            
            # Expand xi and lambda_ for broadcasting:
            # xi: [hidden_size] -> [1,1,hidden_size]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(0)  # [1,1,hidden_size]
            lambda_expanded = self.lambda_.unsqueeze(0).unsqueeze(0) # [1,1,hidden_size]
            
            # Compute z - xi: shape [batch_size, seq_len, hidden_size]
            # We'll broadcast (z) and (xi) along last dim
            # We assume z is just x replicated along hidden dimension
            # If we want to broadcast x to match hidden_size, we do:
            x_expanded = x.expand(batch_size, seq_len, self.hidden_size)  # [batch_size, seq_len, hidden_size]
            
            diff = x_expanded - xi_expanded  # [batch_size, seq_len, hidden_size]
            
            # Multiply by lambda: [batch_size, seq_len, hidden_size]
            diff_lambda = diff * lambda_expanded
            
            # Apply reciprocal activation
            activated = self.activation(diff_lambda)  # [batch_size, seq_len, hidden_size]
            
            # Sum over hidden dimension: result [batch_size, seq_len, 1]
            output = torch.sum(activated, dim=2, keepdim=True)
            
            # Return real and imaginary parts
            return output.real, output.imag
            
    class CauchyNet_t(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet_t, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Parameters
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1.0, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch_size, seq_len]
            batch_size, seq_len = x.size()
            
            # Convert x to complex
            x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
            x = torch.complex(x, torch.zeros_like(x))
            
            # Expand xi for broadcasting: [1, 1, hidden_size]
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)
            
            # Compute z - xi
            diff = (x - xi_expanded)  # [batch_size, seq_len, hidden_size]
            
            # Apply reciprocal activation
            activated = self.activation(diff)  # [batch_size, seq_len, hidden_size]
            
            # Flatten for matrix multiplication
            activated = activated.view(batch_size * seq_len, self.hidden_size)  # [batch_size*seq_len, hidden_size]
            
            # Linear combination with lambda_
            y = torch.matmul(activated, self.lambda_)  # [batch_size*seq_len, output_size]
            
            # Reshape back
            y = y.view(batch_size, seq_len, self.output_size)  # [batch_size, seq_len, output_size]
    
            return y.real, y.imag
    
    
    class CauchyNet0(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet0, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            batch_size, seq_len = x.size()
            x = x.view(batch_size, seq_len, 1)
            x = torch.complex(x, torch.zeros_like(x))
            
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)
            activated = self.activation(xi_expanded - x)
            
            activated = activated.view(batch_size, -1)
            y = torch.matmul(activated, self.lambda_)/hidden_size
            
            return y.real, y.imag
    
    class CauchyNet1(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet1, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)
            )
            
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            batch_size, seq_len = x.size()
            x = x.view(batch_size, seq_len, 1)
            x = torch.complex(x, torch.zeros_like(x))
            
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)
            activated = self.activation(xi_expanded - x)
            
            activated = activated.view(batch_size * seq_len, -1)
            y = torch.matmul(activated, self.lambda_)/hidden_size
            
            y = y.view(batch_size, seq_len, -1)
            return y.real, y.imag
    
    
    
    def loadData(X, Y, model_type, lookBack=None, batchSize=64):
        if model_type in ['LSTM', 'GRU', 'Transformer']:
            assert lookBack is not None, "lookBack cannot be None for recurrent models."
            
            sequences = []
            for i in range(len(Y) - lookBack):
                sequence = Y[i: i + lookBack]
                label = Y[i + lookBack]
                sequences.append((sequence, label))
            
            sequence_tensors, label_tensors = zip(*sequences)
            sequence_tensors = torch.stack(sequence_tensors).unsqueeze(-1)
            label_tensors = torch.stack(label_tensors)
    
            full_dataset = TensorDataset(sequence_tensors, label_tensors)
        else:
            full_dataset = TensorDataset(X.unsqueeze(-1), Y.unsqueeze(-1))
        
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def compute_Y(X):
        return X**2 - X**3 + torch.sin(10*X) + 1 / (X**2 + 4)
    
    def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader, input_size=1, hidden_size=32, output_size=1, lr=0.001, epochs=200):
        model = model_class(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                sequences, labels = batch
                sequences, labels = sequences.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs_real, outputs_imag = model(sequences)
                
                # If output has shape [batch, seq_len, out], take last step only if needed:
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                    outputs_imag = outputs_imag[:, -1, :]
                
                loss = criterion(outputs_real, labels) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    sequences, labels = batch
                    sequences, labels = sequences.to(device), labels.to(device)
                    outputs_real, outputs_imag = model(sequences)
                    if outputs_real.ndim == 3:
                        outputs_real = outputs_real[:, -1, :]
                        outputs_imag = outputs_imag[:, -1, :]
                    loss = criterion(outputs_real, labels) + criterion(outputs_imag, torch.zeros_like(outputs_imag))
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test evaluation
        model.eval()
        predictions = []
        true_values = []
        with torch.no_grad():
            for batch in test_loader:
                sequences, labels = batch
                sequences, labels = sequences.to(device), labels.to(device)
                outputs_real, outputs_imag = model(sequences)
                if outputs_real.ndim == 3:
                    outputs_real = outputs_real[:, -1, :]
                predictions.append(outputs_real.cpu().numpy())
                true_values.append(labels.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        true_values = np.concatenate(true_values)
        
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        
        print(f"{model_name}: Test MSE: {mse:.4f}, MAE: {mae:.4f}")
        return mse, mae
    
    # Data generation and normalization
    data_size = 200
    X = torch.linspace(-1, 1, data_size)
    Y = X**2 +7*torch.cos(5*X**2) + 2*torch.sin(3*X) + 1 / (X**2 + 8)
    
    
    # Normalize Y
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1, 1)
    Y_normalized = scaler.fit_transform(Y_np).reshape(-1)
    Y = torch.tensor(Y_normalized, dtype=torch.float32)
    
    # Data loaders
    lookBack = 1
    batchSize = 2000
    train_loader, val_loader, test_loader = loadData(X, Y, 'CauchyNet', lookBack=lookBack, batchSize=batchSize)
    
    # Compare multiple models
    models_to_compare = {
        "CauchyNet0": CauchyNet0,
        "CauchyNet1": CauchyNet1,
         "CauchyNet_t": CauchyNet_t,
        "CauchyNet_l": CauchyNet_l
    }
    
    results = {}
    for name, cls in models_to_compare.items():
        mse, mae = train_and_evaluate_model(cls, name, train_loader, val_loader, test_loader, lr=0.001, epochs=1000)
        results[name] = (mse, mae)
    
    # Print comparison table
    print("\nModel Comparison:")
    print("Model\t\tMSE\t\tMAE")
    for name, (mse, mae) in results.items():
        print(f"{name}\t{mse:.6f}\t{mae:.6f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 68 FAILED, continuing', flush=True)

_cell[0]=69
print('=== CELL 69 ===', flush=True)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import random
    
    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
    def loadData(X, Y, model_type, lookBack=1, batchSize=64):
        # For simplicity, we assume non-recurrent model; if needed, adapt accordingly.
        # We'll just feed (X, Y) directly as the code above is mostly made for that scenario.
        # If needed to handle recurrent models, adapt code similarly as before.
        
        full_dataset = TensorDataset(X.unsqueeze(-1), Y.unsqueeze(-1))
        total_size = len(full_dataset)
        testSize = int(total_size * 0.25)
        valSize = testSize
        trainSize = total_size - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def compute_function(X):
        return X**2 - X**3 + torch.sin(20*X) + 1 / (X**2 + 4)
    
    def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            for batch in train_loader:
                seq, lbl = batch
                seq, lbl = seq.to(device), lbl.to(device)
                optimizer.zero_grad()
                out_real, out_imag = model(seq)
                if out_real.ndim == 3:
                    out_real = out_real[:, -1, :]
                    out_imag = out_imag[:, -1, :]
                loss = criterion(out_real, lbl) + criterion(out_imag, torch.zeros_like(out_imag))
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= len(train_loader)
            
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    seq, lbl = batch
                    seq, lbl = seq.to(device), lbl.to(device)
                    out_real, out_imag = model(seq)
                    if out_real.ndim == 3:
                        out_real = out_real[:, -1, :]
                        out_imag = out_imag[:, -1, :]
                    loss = criterion(out_real, lbl) + criterion(out_imag, torch.zeros_like(out_imag))
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(val_loader)
            
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state = model.state_dict()
        
        if best_state is not None:
            model.load_state_dict(best_state)
        return train_losses, val_losses
    
    def test_model(model, test_loader, device='cpu'):
        model.eval()
        predictions = []
        true_values = []
        with torch.no_grad():
            for batch in test_loader:
                seq, lbl = batch
                seq, lbl = seq.to(device), lbl.to(device)
                out_real, out_imag = model(seq)
                if out_real.ndim == 3:
                    out_real = out_real[:, -1, :]
                predictions.append(out_real.cpu().numpy())
                true_values.append(lbl.cpu().numpy())
        predictions = np.concatenate(predictions)
        true_values = np.concatenate(true_values)
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        return mse, mae
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_to_compare = {
        "CauchyNet0": CauchyNet0,
        "CauchyNet1": CauchyNet1,
          "CauchyNet_t": CauchyNet_t
    }
    
    data_sizes = [100, 200, 300, 400, 500]
    epochs = 200
    runs = 5
    hidden_size = 32
    output_size = 1
    lr = 0.001
    
    results = {}  # results[(model_name, data_size)] = list of (mse, mae)
    loss_records = {}  # loss_records[(model_name, data_size, 'train'/'val')] = (mean, std) arrays
    
    for model_name, model_class in models_to_compare.items():
        for data_size in data_sizes:
            # We will store train/val losses for all runs
            all_train_losses = []
            all_val_losses = []
            all_mse = []
            all_mae = []
            
            for seed in range(runs):
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                X = torch.linspace(-1,1,data_size)
                Y = compute_function(X)
                # Normalize Y
                scaler = MinMaxScaler()
                Y_np = Y.numpy().reshape(-1,1)
                Y_normalized = scaler.fit_transform(Y_np).reshape(-1)
                Y_scaled = torch.tensor(Y_normalized, dtype=torch.float32)
                
                train_loader, val_loader, test_loader = loadData(X, Y_scaled, 'CauchyNet', lookBack=1, batchSize=64)
                
                model = model_class(1, hidden_size, output_size).to(device)
                train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
                mse, mae = test_model(model, test_loader, device=device)
                
                all_train_losses.append(train_losses)
                all_val_losses.append(val_losses)
                all_mse.append(mse)
                all_mae.append(mae)
            
            # Compute mean/std of train/val losses across runs
            all_train_losses = np.array(all_train_losses)  # shape: [runs, epochs]
            all_val_losses = np.array(all_val_losses)
            
            mean_train = np.mean(all_train_losses, axis=0)
            std_train = np.std(all_train_losses, axis=0)
            mean_val = np.mean(all_val_losses, axis=0)
            std_val = np.std(all_val_losses, axis=0)
            
            # Store results
            results[(model_name, data_size)] = (np.mean(all_mse), np.std(all_mse), np.mean(all_mae), np.std(all_mae))
            loss_records[(model_name, data_size, 'train')] = (mean_train, std_train)
            loss_records[(model_name, data_size, 'val')] = (mean_val, std_val)
    
    # Plotting the losses
    # One plot per model, all data sizes
    for model_name in models_to_compare.keys():
        plt.figure(figsize=(12, 6))
        for data_size in data_sizes:
            mean_train, std_train = loss_records[(model_name, data_size, 'train')]
            mean_val, std_val = loss_records[(model_name, data_size, 'val')]
            
            epochs_range = np.arange(1, len(mean_train)+1)
            # Plot train
            plt.plot(epochs_range, mean_train, label=f"Train (size={data_size})")
            plt.fill_between(epochs_range, mean_train-std_train, mean_train+std_train, alpha=0.2)
            # Plot val
            plt.plot(epochs_range, mean_val, linestyle='--', label=f"Val (size={data_size})")
            plt.fill_between(epochs_range, mean_val-std_val, mean_val+std_val, alpha=0.2)
        
        plt.title(f"Loss Curves (mean ± std) for {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Print final comparison table
    print("Final Results (Mean ± Std) for MSE and MAE on Test Set:\n")
    print("Model\tDataSize\tMSE(mean±std)\tMAE(mean±std)")
    for (model_name, data_size), (mse_mean, mse_std, mae_mean, mae_std) in results.items():
        print(f"{model_name}\t{data_size}\t{mse_mean:.5f}±{mse_std:.5f}\t{mae_mean:.5f}±{mae_std:.5f}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 69 FAILED, continuing', flush=True)

_cell[0]=70
print('=== CELL 70 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Suppose `results` is defined as in the previous code:
    # results[(model_name, data_size)] = (mse_mean, mse_std, mae_mean, mae_std)
    
    # Extract model names and data sizes
    model_names = sorted({k[0] for k in results.keys()})
    data_sizes = sorted({k[1] for k in results.keys()})
    
    # Build a DataFrame for MSE means
    mse_data = []
    for model in model_names:
        row = []
        for size in data_sizes:
            mse_mean, mse_std, mae_mean, mae_std = results[(model, size)]
            row.append(mse_mean)
        mse_data.append(row)
    
    mse_df = pd.DataFrame(mse_data, index=model_names, columns=data_sizes)
    
    # Prepare figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We will create a "bubble plot":
    # X-axis: data_sizes
    # Y-axis: model_names
    # Each cell: circle size and color based on MSE
    
    xvals = np.arange(len(data_sizes))
    yvals = np.arange(len(model_names))
    
    # Flatten the data for scatter
    x_coords, y_coords = np.meshgrid(xvals, yvals)
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    
    mse_flat = mse_df.values.flatten()
    
    # Normalize MSE values for color mapping
    norm = plt.Normalize(mse_flat.min(), mse_flat.max())
    colors = plt.cm.viridis(norm(mse_flat))
    
    # We can scale the marker size by some factor
    # For example, larger MSE means larger circle.
    # Adjust scale_factor as needed
    scale_factor = 3000
    sizes = mse_flat * scale_factor
    
    # Create scatter plot
    scatter = ax.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=0.7, edgecolors='black')
    
    # Set ticks and labels
    ax.set_xticks(xvals)
    ax.set_xticklabels(data_sizes, rotation=45, ha='right')
    ax.set_yticks(yvals)
    ax.set_yticklabels(model_names)
    
    ax.set_xlabel("Data Size")
    ax.set_ylabel("Model")
    ax.set_title("MSE by Model and Data Size (Bubble Heat Map)")
    
    # Add a colorbar for MSE
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
    cbar.set_label("MSE Mean")
    
    # Optionally, add text annotations inside each bubble (optional)
    for (i, j), val in np.ndenumerate(mse_df.values):
        ax.text(j, i, f"{val:.3f}", ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 70 FAILED, continuing', flush=True)

_cell[0]=71
print('=== CELL 71 ===', flush=True)
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Assume you have a dictionary like this:
    # results[(model_name, data_size)] = (mse_mean, mse_std, mae_mean, mae_std)
    # where model_names and data_sizes are known.
    
    model_names = sorted({k[0] for k in results.keys()})
    data_sizes = sorted({k[1] for k in results.keys()})
    
    # Create a DataFrame for MSE and MAE comparison
    # Rows: data scenarios (data_sizes)
    # Columns: each model
    # We'll create two DataFrames: one for MSE, one for MAE
    mse_table = pd.DataFrame(index=data_sizes, columns=model_names)
    mae_table = pd.DataFrame(index=data_sizes, columns=model_names)
    
    for (model, size), (mse_mean, mse_std, mae_mean, mae_std) in results.items():
        mse_table.loc[size, model] = f"{mse_mean:.4f}±{mse_std:.4f}"
        mae_table.loc[size, model] = f"{mae_mean:.4f}±{mae_std:.4f}"
    
    print("MSE Comparison Table (Scenario as rows, Models as columns):")
    print(mse_table)
    print("\nMAE Comparison Table (Scenario as rows, Models as columns):")
    print(mae_table)
    
    # If you'd prefer a table without the ±std for clarity, you can store just the mse_mean and mae_mean:
    mse_mean_table = pd.DataFrame(index=data_sizes, columns=model_names)
    for (model, size), (mse_mean, mse_std, mae_mean, mae_std) in results.items():
        mse_mean_table.loc[size, model] = mse_mean
    
    print("\nMSE Mean Only:")
    print(mse_mean_table)
    
    # Plotting: For each scenario (data_size), plot a bar chart comparing all models
    fig, axes = plt.subplots(nrows=len(data_sizes), ncols=1, figsize=(8, 4 * len(data_sizes)), sharex=True)
    
    if len(data_sizes) == 1:
        axes = [axes]  # If there's only one scenario, ensure axes is a list
    
    for ax, size in zip(axes, data_sizes):
        # Extract MSE means for this scenario
        scenario_mse = []
        scenario_err = []
        for m in model_names:
            mse_mean, mse_std, mae_mean, mae_std = results[(m, size)]
            scenario_mse.append(mse_mean)
            scenario_err.append(mse_std)  # if you want error bars
    
        x = np.arange(len(model_names))
        width = 0.6
        bars = ax.bar(x, scenario_mse, width, yerr=scenario_err, capsize=5, alpha=0.7)
    
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel("MSE")
        ax.set_title(f"Scenario: Data Size = {size}")
        # Optionally highlight the best model for this scenario:
        best_index = np.argmin(scenario_mse)
        bars[best_index].set_edgecolor("red")
        bars[best_index].set_linewidth(3)
    
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 71 FAILED, continuing', flush=True)

_cell[0]=72
print('=== CELL 72 ===', flush=True)
try:
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import datetime
    import matplotlib.pyplot as plt
    import math, time
    from math import sqrt
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import random
    import itertools
    import datetime
    from operator import itemgetter
    import time
    import torch.optim as optim
    import time# Set device
    SEED=10
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm
    
    # Seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Define the Reciprocal Activation with a practical epsilon
    class ReciprocalActivation(nn.Module):
        def forward(self, input):
            epsilon = 1e-8  # Small value to prevent division by zero
            return 1.0 / (input + epsilon)
    
    # Define the CauchyNet class
    class CauchyNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Initialize lambda with appropriate dimensions
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            # Initialize xi with appropriate dimensions
            angles = 2 * np.pi * torch.rand(hidden_size)  # Random angles
            real_part = 2 * np.pi * torch.cos(angles)     # Radius of 2pi for real part
            imaginary_part = torch.sin(angles)            # Radius of 1 for imaginary part
            self.xi = nn.Parameter(torch.complex(real_part, imaginary_part))
            
            # Define the custom activation function
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch_size, seq_len]
            batch_size, seq_len = x.size()
            
            # Reshape and convert to complex
            x = x.view(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]
            x = torch.complex(x, torch.zeros_like(x))  # [batch_size, seq_len, 1]
            
            # Expand xi for broadcasting
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)  # [1, 1, hidden_size]
            
            # Activation
            activated = self.activation(xi_expanded - x)  # [batch_size, seq_len, hidden_size]
            
            # Flatten for matrix multiplication
            activated = activated.view(batch_size * seq_len, -1)  # [batch_size * seq_len, hidden_size]
            
            # Matrix multiplication
            y = torch.matmul(activated, self.lambda_)  # [batch_size * seq_len, output_size]
            
            # Reshape back to [batch_size, seq_len, output_size]
            y = y.view(batch_size, seq_len, -1)
            
            # Extract the last time step's output
            y_last = y[:, -1, :]  # [batch_size, output_size]
            
            # Return real and imaginary parts
            return y_last.real, y_last.imag
    
    # Define the CauchyNet0 class
    class CauchyNet0(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet0, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Initialize lambda with appropriate dimensions
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            # Initialize xi with appropriate dimensions
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)  # [hidden_size]
            )
            
            # Define the custom activation function
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch_size, seq_len]
            batch_size, seq_len = x.size()
            
            # Reshape and convert to complex
            x = x.view(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]
            x = torch.complex(x, torch.zeros_like(x))  # [batch_size, seq_len, 1]
            
            # Expand xi for broadcasting
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)  # [1, 1, hidden_size]
            
            # Activation
            activated = self.activation(xi_expanded - x)  # [batch_size, hidden_size, 1]
            
            # Flatten for matrix multiplication
            activated = activated.view(batch_size, -1)  # [batch_size, hidden_size]
            
            # Matrix multiplication
            y = torch.matmul(activated, self.lambda_)  # [batch_size, output_size]
            
            # Return real and imaginary parts
            return y.real, y.imag
    
    #This is betterL
    
    class CauchyNet1(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet1, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Initialize lambda with appropriate dimensions
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            # Initialize xi with appropriate dimensions
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)  # [hidden_size]
            )
            
            # Define the custom activation function
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch_size, seq_len]
            batch_size, seq_len = x.size()
            
            # Reshape and convert to complex
            x = x.view(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]
            x = torch.complex(x, torch.zeros_like(x))  # [batch_size, seq_len, 1]
            
            # Expand xi for broadcasting
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)  # [1, 1, hidden_size]
            
            # Activation
            activated = self.activation(xi_expanded - x)  # [batch_size, seq_len, hidden_size]
            
            # Flatten for matrix multiplication
            activated = activated.view(batch_size * seq_len, -1)  # [batch_size * seq_len, hidden_size]
            
            # Matrix multiplication
            y = torch.matmul(activated, self.lambda_)  # [batch_size * seq_len, output_size]
            
            # Reshape back to [batch_size, seq_len, output_size]
            y = y.view(batch_size, seq_len, -1)
            
            # Extract the last time step's output
            y_last = y#[:, -1, :]  # [batch_size, output_size]
            
            # Return real and imaginary parts
            return y_last.real, y_last.imag
    
    
    # Define the loadData function
    def loadData(X, Y, model_type, lookBack=None, batchSize=32):
        if model_type in ['LSTM', 'GRU', 'Transformer']:
            assert lookBack is not None, "lookBack cannot be None for recurrent models."
            
            # Prepare data for sequence models
            sequences = []
            for i in range(len(Y) - lookBack):
                sequence = Y[i: i + lookBack]
                label = Y[i + lookBack]
                sequences.append((sequence, label))
            
            # Convert list of tuples to two tensors
            sequence_tensors, label_tensors = zip(*sequences)
            sequence_tensors = torch.stack(sequence_tensors).unsqueeze(-1)  # [num_samples, lookBack, 1]
            label_tensors = torch.stack(label_tensors).unsqueeze(-1)        # [num_samples, 1]
            
            full_dataset = TensorDataset(sequence_tensors, label_tensors)
        elif model_type == 'NBeats':
            # Prepare data for NBeats similarly if needed
            sequences = []
            for i in range(len(Y) - lookBack):
                sequence = Y[i: i + lookBack]
                label = Y[i + lookBack]
                sequences.append((sequence, label))
            
            sequence_tensors, label_tensors = zip(*sequences)
            sequence_tensors = torch.stack(sequence_tensors)               # [num_samples, lookBack]
            label_tensors = torch.stack(label_tensors).unsqueeze(-1)      # [num_samples, 1]
            
            full_dataset = TensorDataset(sequence_tensors, label_tensors)
        else:
            # For non-sequence models like FNN, CauchyNet, CauchyNet0
            full_dataset = TensorDataset(X.unsqueeze(-1), Y.unsqueeze(-1))  # [num_samples, 1]
        
        # Splitting the dataset
        total_size = len(full_dataset)
        trainSize = int(total_size * 0.5)
        valSize = int(total_size * 0.25)
        testSize = total_size - trainSize - valSize
        
        # Use random_split for consistency
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    # Generate data
    data_size = 100
    X = torch.linspace(0, 1, data_size).float()
    
    def compute_Y(X):
        return X**2 - X**3 + torch.sin(10*X) + 1 / (X**2 + 4)
    
    def compute_Y_np(X):
        return X**2 - X**3 + np.sin(10*X) + 1 / (X**2 + 4)
    
    Y = compute_Y(X).float()
    
    # Normalize Y
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1, 1)
    Y_normalized = scaler.fit_transform(Y_np).reshape(-1)
    Y = torch.tensor(Y_normalized, dtype=torch.float32)
    
    # Load the data
    lookBack = 1
    batchSize = 64  # Use a reasonable batch size
    train_loader, val_loader, test_loader = loadData(X, Y, 'CauchyNet', lookBack=lookBack, batchSize=batchSize)
    
    # Initialize the model, loss function, and optimizer
    input_size = 1
    hidden_size = 32
    output_size = 1
    
    model = CauchyNet0(input_size, hidden_size, output_size).to('cpu')  # Use 'cuda' if GPU is available
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight_decay for regularization
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)  # Added scheduler
    
    # Training loop with Early Stopping
    num_epochs = 500
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 50  # Number of epochs to wait before early stopping
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            sequences, labels = batch
            optimizer.zero_grad()
            
            # Forward pass
            outputs, output_imag = model(sequences)  # [batch_size, output_size], [batch_size, output_size]
            
            # Calculate loss
            loss = criterion(outputs, labels) + criterion(output_imag, torch.zeros_like(output_imag))
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sequences, labels = batch
                outputs, output_imag = model(sequences)
                loss = criterion(outputs, labels) + criterion(output_imag, torch.zeros_like(output_imag))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Step the scheduler
        scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print losses every 50 epochs and the first epoch
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model state based on validation loss.")
    
    # Testing and plotting
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences, labels = batch
            outputs, _ = model(sequences)
            predictions.append(outputs)
            true_values.append(labels)
    
    predictions = torch.cat(predictions).cpu().numpy()
    true_values = torch.cat(true_values).cpu().numpy()
    
    # Inverse transform the predictions and true values to original scale
    predictions_original = scaler.inverse_transform(predictions)
    true_values_original = scaler.inverse_transform(true_values)
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(true_values_original, label='True Values')
    plt.plot(predictions_original, label='Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predictions vs True Values on Test Data')
    plt.legend()
    plt.show()
    
    
    
    
    class CauchyNet2(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(CauchyNet2, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            # Initialize lambda with appropriate dimensions
            self.lambda_ = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat)
            )
            
            # Initialize xi with appropriate dimensions
            self.xi = nn.Parameter(
                torch.normal(mean=0.0, std=0.5, size=(hidden_size,), dtype=torch.cfloat)  # [hidden_size]
            )
            
            # Define the custom activation function
            self.activation = ReciprocalActivation()
    
        def forward(self, x):
            # x: [batch_size, seq_len]
            batch_size, seq_len = x.size()
            
            # Reshape and convert to complex
            x = x.view(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]
            x = torch.complex(x, torch.zeros_like(x))  # [batch_size, seq_len, 1]
            
            # Expand xi for broadcasting
            xi_expanded = self.xi.unsqueeze(0).unsqueeze(1)  # [1, 1, hidden_size]
            
            # Activation
            activated = self.activation(xi_expanded - x)  # [batch_size, seq_len, hidden_size]
            
            # Flatten for matrix multiplication
            activated = activated.view(batch_size * seq_len, -1)  # [batch_size * seq_len, hidden_size]
            
            # Matrix multiplication
            y = torch.matmul(activated, self.lambda_)  # [batch_size * seq_len, output_size]
            
            # Reshape back to [batch_size, seq_len, output_size]
            y = y.view(batch_size, seq_len, -1)
            
            # Extract the last time step's output
            y_last = y#[:, -1, :]  # [batch_size, output_size]
            
            # Return real and imaginary parts
            return y_last.real, y_last.imag
    # Define the loadData function
    
    def loadData(X, Y, model_type, lookBack=None, batchSize=1000):
        if model_type in ['LSTM', 'GRU', 'Transformer']:
            assert lookBack is not None, "lookBack cannot be None for recurrent models."
            
            # Prepare data for LSTM with lookbacks
            sequences = []
            for i in range(len(Y) - lookBack):
                sequence = Y[i: i + lookBack]
                label = Y[i + lookBack]
                sequences.append((sequence, label))
            
            # Convert list of tuples to two tensors
            sequence_tensors, label_tensors = zip(*sequences)
            sequence_tensors = torch.stack(sequence_tensors).unsqueeze(-1)  # Add feature dimension
            label_tensors = torch.stack(label_tensors)
    
            full_dataset = TensorDataset(sequence_tensors, label_tensors)
        else:
            # For models that don't need look-back, use the input data directly
            full_dataset = TensorDataset(X.unsqueeze(-1), Y.unsqueeze(-1))  # Add feature dimension
    
        # Splitting the dataset
        testSize = int(np.round(0.25 * len(full_dataset)))
        valSize = testSize
        trainSize = len(full_dataset) - testSize - valSize
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [trainSize, valSize, testSize])
    
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
    
        return train_loader, val_loader, test_loader
    
    # Generate data
    data_size = 300
    X = torch.linspace(-1,1, data_size)
    Y = X**2 - X**3 + torch.sin(20*X) + 1 / (X**2 + 4)
    
    # Load the data
    lookBack = 1
    batchSize = 500
    train_loader, val_loader, test_loader = loadData(X, Y, 'CauchyNet', lookBack=lookBack, batchSize=batchSize)
    
    # Initialize the model, loss function, and optimizer
    input_size = 1
    hidden_size = 32
    output_size = 1
    
    model = CauchyNet(input_size, hidden_size, output_size).to('cpu')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            sequences, labels = batch
            optimizer.zero_grad()
            outputs,outputs1 = model(sequences)
            outputs = outputs[:, -1]  # Take the last time step's output
            outputs1 = outputs1[:, -1]
            loss = criterion(outputs, labels)+ criterion(outputs1, torch.zeros_like(outputs1))
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sequences, labels = batch
                outputs,_ = model(sequences)
                outputs = outputs[:, -1]  # Take the last time step's output
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")
    
    # Testing and plotting
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences, labels = batch
            outputs,_ = model(sequences)
            outputs = outputs[:, -1]  # Take the last time step's output
            predictions.append(outputs)
            true_values.append(labels)
    
    predictions = torch.cat(predictions).cpu().numpy()
    true_values = torch.cat(true_values).cpu().numpy()
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 72 FAILED, continuing', flush=True)
