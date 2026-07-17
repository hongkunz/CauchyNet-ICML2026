import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback
_n=[0]
def _show(*a,**k):
    plt.gcf().savefig(f'exp5_fig_{_n[0]:02d}.png', dpi=110, bbox_inches='tight'); _n[0]+=1; plt.close('all')
plt.show=_show

print('=== CELL 0 ===', flush=True)
try:
    #!pip install --upgrade torch torchvision torchaudio
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 0 FAILED, continuing', flush=True)

print('=== CELL 1 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import pytorch_warmup as warmup
    import torch.optim as optim
    import time
    # Define the custom activation function for complex numbers
    import math
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    import os
    
    from sklearn.preprocessing import MinMaxScaler
    
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    
        
    import torch
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import torch
    import numpy as np
    from models import CauchyNet0,CauchyNet, SIREN, RBFNetwork, FNN_ReLU,  NBeats,  MinimalTransformer
    from utils4 import train_model, loadData
    from modles_Cauchys import  CauchyNet0,  CauchyNet1,  CauchyNet, CauchyNet_RealActivation, CauchyNet_NoImagPenalty,CauchyNet_PurelyRealParams,CauchyNet_NonHolomorphic
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 1 FAILED, continuing', flush=True)

print('=== CELL 2 ===', flush=True)
try:
    import pandas as pd
    # Load the CSV file into a DataFrame
    
    df = pd.read_csv('M4_trend_data.csv')
    df
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 2 FAILED, continuing', flush=True)

print('=== CELL 3 ===', flush=True)
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
    plt.scatter(X_mapped, Y_mapped, color="blue", s=5, label="f(x)")
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
    print('!! CELL 3 FAILED, continuing', flush=True)

print('=== CELL 4 ===', flush=True)
try:
    def challenging_function(x):
        """
        A highly non-symmetric function designed to highlight differences between CauchyNet
        and sine-activation-based models like SIREN.
        """
        return (
            1 / ((x + 0.6)**2 + 0.005)   # Sharp positive peak at x=0.1
                        - 40*torch.exp(-2 * (x + 0.4)**2)  # Localized Gaussian bump at x=0.4
                     +x**2 / ((x - 0.8)**4 + 0.01)     # Sharp negative peak at x=0.8
                + 50 * torch.sign(x) * torch.abs(torch.sin(3*x)+0.8)**1.5 * torch.sin(10*x) # Asymmetric cubic-like feature
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
    plt.savefig("toyf.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 4 FAILED, continuing', flush=True)

print('=== CELL 5 ===', flush=True)
try:
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def train_and_evaluate_model(
        model_constructor,
        input_size, hidden_size, output_size,
        train_loader, val_loader, test_loader,
        lr=0.005,
        epochs=10000,
        device=None,
        scaler=None,
        scheduler_step_size=20000,
        scheduler_gamma=0.9,
        model_name="MyModel",
    ):
        """
        Wraps your original 'train_model' logic into a single function that builds
        the model, trains it with Adam + StepLR, then evaluates on the test set.
        Returns results in the same style as your second snippet.
    
        Parameters
        ----------
        model_constructor : Callable
            E.g. def MyNet(in_dim, hidden_dim, out_dim): return ...
        input_size, hidden_size, output_size : int
            Dimensions for the model.
        train_loader, val_loader, test_loader : DataLoader
            Dataloaders for training, validation, and testing data.
        lr : float
            Learning rate for Adam.
        epochs : int
            Number of training epochs.
        device : torch.device or None
            If None, chooses CUDA if available, else CPU.
        scaler : object or None
            If provided, used to inverse-transform test predictions/targets.
        scheduler_step_size : int
            Step size for StepLR.
        scheduler_gamma : float
            Gamma (decay factor) for StepLR.
        model_name : str
            Will be used for printing & saving checkpoints.
    
        Returns
        -------
        model : nn.Module
            The best (checkpoint-loaded) trained model.
        train_losses : list of float
            Per-epoch average training loss.
        val_losses : list of float
            Per-epoch average validation loss.
        test_mse : float
            MSE on test set (in *unscaled* domain if scaler is provided).
        test_mae : float
            MAE on test set (in *unscaled* domain if scaler is provided).
        preds_all_unscaled : np.ndarray
            Final test predictions (1D), unscaled if `scaler` was given.
        truths_all_unscaled : np.ndarray
            Final test targets (1D), unscaled if `scaler` was given.
        training_time : float
            Time (seconds) for the full training loop.
        num_params : int
            Number of trainable parameters in the model.
        """
    
        # ------------------- Device Setup -------------------
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # ------------------- Model Construction -------------------
        model = model_constructor(input_size, hidden_size, output_size).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        # ------------------- Optimizer & Scheduler -------------------
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )
        criterion = nn.MSELoss()
    
        train_losses = []
        val_losses   = []
    
        best_val_loss = float('inf')
        best_state    = None
    
        start_time = time.time()
    
        # ==================== Training Loop ====================
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
    
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
    
                # If model expects [B,1], do unsqueeze
                x_batch = x_batch.unsqueeze(-1)  # => [B,1]
                y_batch = y_batch.unsqueeze(-1)  # => [B,1]
    
                optimizer.zero_grad()
                out = model(x_batch)  # could be a single Tensor or (real, imag)
    
                # Same logic as your original train_model
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    zeros_imag = torch.zeros_like(y_imag)
                    loss = criterion(y_real, y_batch) + criterion(y_imag, zeros_imag)
                else:
                    loss = criterion(out, y_batch)
    
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
            # Step the scheduler (every epoch, not val-based)
            scheduler.step()
    
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
    
            # ==================== Validation Loop ====================
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
    
                    x_val = x_val.unsqueeze(-1)
                    y_val = y_val.unsqueeze(-1)
    
                    val_out = model(x_val)
                    if isinstance(val_out, tuple):
                        vr, vi = val_out
                        zeros_im = torch.zeros_like(vi)
                        val_loss = criterion(vr, y_val) + criterion(vi, zeros_im)
                    else:
                        val_loss = criterion(val_out, y_val)
    
                    val_loss_sum += val_loss.item()
    
            avg_val_loss = val_loss_sum / len(val_loader)
            val_losses.append(avg_val_loss)
    
            # Best model checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state    = model.state_dict()
    
            # Optional print every N epochs
            if (epoch + 1) % 500 == 0:
                print(
                    f"{model_name} Epoch {epoch+1}/{epochs}, "
                    f"Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}"
                )
    
        training_time = time.time() - start_time
        print(f"\nFinished Training {model_name}. Total Time: {training_time:.2f} seconds")
    
        # ==================== Load Best Model & Save ====================
        if best_state is not None:
            model.load_state_dict(best_state)
            checkpoint_path = f"{model_name}_best.pth"
            torch.save(best_state, checkpoint_path)
            print(f"Best {model_name} model saved with val_loss={best_val_loss:.6f}.\n")
        else:
            print("No improvement found, best model not saved.\n")
    
        # ==================== Test Loop ====================
        model.eval()
        preds_list   = []
        targets_list = []
    
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)
    
                x_test = x_test.unsqueeze(-1)  # => [B,1]
                y_test = y_test.unsqueeze(-1)
    
                out_test = model(x_test)
                if isinstance(out_test, tuple):
                    y_r, y_i = out_test
                    preds_scaled = y_r.cpu().numpy().flatten()
                else:
                    preds_scaled = out_test.cpu().numpy().flatten()
    
                truths_scaled = y_test.cpu().numpy().flatten()
    
                # If scaler is provided, inverse-transform to original domain
                if scaler is not None:
                    preds_unscaled  = scaler.inverse_transform(
                        preds_scaled.reshape(-1,1)
                    ).flatten()
                    truths_unscaled = scaler.inverse_transform(
                        truths_scaled.reshape(-1,1)
                    ).flatten()
                else:
                    preds_unscaled  = preds_scaled
                    truths_unscaled = truths_scaled
    
                preds_list.append(preds_unscaled)
                targets_list.append(truths_unscaled)
    
        preds_all_unscaled  = np.concatenate(preds_list)
        truths_all_unscaled = np.concatenate(targets_list)
    
        test_mse = mean_squared_error(truths_all_unscaled, preds_all_unscaled)
        test_mae = mean_absolute_error(truths_all_unscaled, preds_all_unscaled)
        print(
            f"{model_name} => Test MSE={test_mse:.6f}, "
            f"Test MAE={test_mae:.6f}"
        )
    
        # Return in second-snippet style
        return (
            model,            # The best (checkpoint-loaded) model
            train_losses,     # list of training losses
            val_losses,       # list of validation losses
            test_mse,         # final test MSE
            test_mae,         # final test MAE
            preds_all_unscaled,  # final test predictions
            truths_all_unscaled, # final test targets
            training_time,    # total training time in seconds
            num_params        # number of trainable model parameters
        )
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 5 FAILED, continuing', flush=True)

print('=== CELL 6 ===', flush=True)
try:
    import torch
    import pickle
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from sklearn.preprocessing import MinMaxScaler
    import math
    
    # Import utility functions (ensure they exist in your environment)
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
    
    # 1) Configure environment style
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    # 2) Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3) Generate or load your data
    data_size = 500
    X = torch.linspace(-1, 1, data_size)
    Y = challenging_function(X)  # Make sure challenging_function is defined somewhere
    
    # 4) Scale Y
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1, 1)  # shape (data_size,1)
    Y_norm = scaler.fit_transform(Y_np).flatten()  # shape (data_size,)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32)
    
    # 5) Build dataset + splits
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    # 6) Models dictionary
    models_dict = {
       "CauchyNet":  CauchyNet1,
       # "CauchyNet0": CauchyNet,
       # "CauchyNet1": CauchyNet0,
       # ...
    }
    
    # 7) Create color map for plotting
    model_names = list(models_dict.keys())
    color_map   = get_model_colors(model_names, emphasize_model="CauchyNet")
    
    # 8) Training configuration
    input_size   = 1
    hidden_size  = 128
    output_size  = 1
    lr           = 0.01
    epochs       = 4000
    num_repeats  = 1
    
    # 9) Dictionaries to log intervals and results
    interval_logs = {}
    results_dict  = {}
    
    # 10) Train multiple runs per model & collect logs
    last_model_name = None  # We'll track which model name was last in the loop
    
    for model_name, constructor in models_dict.items():
        last_model_name = model_name  # Track so we know which model was last
        print(f"\n=== Training {model_name} ===")
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        Num_Params = None
    
        for _ in range(num_repeats):
            # NOTE: the function returns exactly 9 items
            (model,
             train_losses,
             val_losses,
             test_mse,
             test_mae,
             preds_unscaled,
             truths_unscaled,
             training_time,
             Num_Params) = train_and_evaluate_model(
                model_constructor  = constructor,
                input_size         = input_size,
                hidden_size        = hidden_size,
                output_size        = output_size,
                train_loader       = train_loader,
                val_loader         = val_loader,
                test_loader        = test_loader,
                lr                 = lr,
                epochs             = epochs,
                device             = device,
                scaler             = scaler,
                model_name         = model_name
            )
    
            runs_mse.append(test_mse)
            runs_mae.append(test_mae)
            runs_time.append(training_time)
            train_runs.append(train_losses)
            val_runs.append(val_losses)
    
        # Mean ± Std across repeats
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
    
    # 11) Plot training curves (with confidence intervals)
    plot_training_curves_with_confidence(
        interval_logs,
        model_names,
        color_map,
        filename="training_curves1.pdf"
    )
    
    # 12) Load the last trained model's checkpoint
    if last_model_name is not None:
        final_model = models_dict[last_model_name](input_size, hidden_size, output_size).to(device)
        checkpoint_path = f"{last_model_name}_best.pth"
    
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            final_model.load_state_dict(state_dict)
            print(f"[INFO] Loaded best checkpoint for {last_model_name}.")
        else:
            print(f"[WARNING] No checkpoint found for {last_model_name}, using fresh model.")
    
        final_model.eval()
        print(f"Final model is: {last_model_name}")
    else:
        print("[WARN] No models in models_dict => skipping final checkpoint load.")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 6 FAILED, continuing', flush=True)

print('=== CELL 7 ===', flush=True)
try:
    import torch
    import numpy as np
    import os
    
    # -------------------------------------------------------------------------
    # 0) Ensure the following variables / objects already exist in your session:
    #    - device: a torch.device (cpu/cuda)
    #    - models_dict: dict mapping "CauchyNet" etc. to their constructors
    #    - scaler: a fitted MinMaxScaler (already used in training)
    #    - Y_np: the original unscaled Y in shape [N,1]
    #    - test_loader: your test DataLoader
    #    - challenging_function: a function for generating the "true" Y
    #    - color_map: a dict of color mappings for your model names
    #    - The plotting functions from utils2 (subplot_predictions, plot_box_errors, plot_error_curves)
    #    - The dictionary `results_dict` if you want to print final stats
    # -------------------------------------------------------------------------
    
    # Example dictionary of checkpoints. Adjust to match your actual file names.
    checkpoint_paths = {
        "CauchyNet": "CauchyNet_best.pth",
        # e.g. "CauchyNet1": "CauchyNet1_best.pth",
        # etc.
    }
    
    # 1) Create fresh model instances, load their state_dicts
    loaded_models = {}
    for model_name, ckpt_path in checkpoint_paths.items():
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] checkpoint not found => {ckpt_path}")
            continue
    
        # Build model with the SAME hidden_size you used when training
        constructor = models_dict[model_name] 
        model = constructor(input_size=1, hidden_size=128, output_size=1)  
        # If your checkpoint was originally trained with hidden_size=128, 
        # then you must do 128 here, not 1000.
    
        # Load the saved weights (use weights_only=True to address the FutureWarning)
        print(f"[INFO] Loading {model_name} from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    
        loaded_models[model_name] = model
    
    # 2) Evaluate predictions on a uniform grid for final overlay
    X_plot = torch.linspace(-0.95, 0.95, 200).to(device)
    Y_plot_true = challenging_function(X_plot.cpu()).numpy()  # unscaled ground truth
    
    overlay_preds = {}
    for model_name, model in loaded_models.items():
        with torch.no_grad():
            out = model(X_plot.unsqueeze(-1).float())
            if isinstance(out, tuple):
                y_real, _ = out
                preds_scaled = y_real.squeeze(-1).cpu().numpy()
            else:
                preds_scaled = out.squeeze(-1).cpu().numpy()
    
        preds_unscaled = scaler.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).flatten()
        overlay_preds[model_name] = preds_unscaled
    
    # (Optional) Subplot predictions
    subplot_predictions(
        X_plot.cpu().numpy(),
        Y_plot_true,
        overlay_preds,
        color_map,
        filename="subplot_predictions_from_saved_models.pdf"
    )
    
    # 3) Box plot of errors on test set (unscaled)
    error_dict = {}
    for mname, model in loaded_models.items():
        abs_errors = []
        with torch.no_grad():
            for x_batch, y_batch_scaled in test_loader:
                x_batch = x_batch.to(device)
                y_batch_scaled = y_batch_scaled.to(device)
    
                out = model(x_batch.unsqueeze(-1))
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_scaled = out.cpu().numpy().flatten()
    
                truths_scaled = y_batch_scaled.cpu().numpy().flatten()
    
                preds_unscaled  = scaler.inverse_transform(
                    preds_scaled.reshape(-1, 1)
                ).flatten()
                truths_unscaled = scaler.inverse_transform(
                    truths_scaled.reshape(-1, 1)
                ).flatten()
    
                abs_errors.extend(np.abs(preds_unscaled - truths_unscaled))
    
        error_dict[mname] = np.array(abs_errors)
    
    plot_box_errors(
        error_dict,
        color_map,
        filename="box_errors_from_saved_models.pdf"
    )
    
    # 4) (Optional) Print summary if 'results_dict' is in memory
    print("\n=== Summary from loaded models ===")
    for mname in loaded_models.keys():
        stats = results_dict.get(mname, None)  # if results_dict was filled after training
        if stats is not None:
            print(f"{mname:12s} => "
                  f"MSE={stats['MSE_mean']:.6f}±{stats['MSE_std']:.4f}, "
                  f"MAE={stats['MAE_mean']:.4f}±{stats['MAE_std']:.4f}, "
                  f"Time={stats['Time_mean']:.2f}±{stats['Time_std']:.2f}s, "
                  f"#Params={stats['Num_Params']}")
        else:
            print(f"{mname:12s} => No stats found in results_dict")
    
    # 5) (Optional) Plot error curves
    plot_error_curves(
        X_plot.cpu().numpy(),
        Y_plot_true,
        overlay_preds,
        color_map,
        filename="error_curves_from_saved_models.pdf"
    )
    
    print("\n=== Done with loaded models! ===")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 7 FAILED, continuing', flush=True)

print('=== CELL 8 ===', flush=True)
try:
    import torch
    import numpy as np
    import os
    
    # -------------------------------------------------------------------------
    # 0) Ensure the following variables / objects already exist in your session:
    #    - device: a torch.device (cpu/cuda)
    #    - models_dict: dict mapping "CauchyNet", etc. to their constructors
    #    - scaler: a fitted MinMaxScaler (already used in training)
    #    - Y_np: the original unscaled Y in shape [N,1]
    #    - test_loader: your test DataLoader
    #    - challenging_function: a function for generating the "true" Y
    #    - color_map: a dict of color mappings for your model names
    #    - The plotting functions from utils2: subplot_predictions, plot_box_errors, plot_error_curves
    #    - The dictionary `results_dict` if you want to print final stats
    #
    # If any are missing, define them or import them before this snippet.
    # -------------------------------------------------------------------------
    
    # Example dictionary of checkpoints. Adjust to match your actual file names.
    checkpoint_paths = {
        "CauchyNet":   "CauchyNet_best.pth",
       # "CauchyNet1":  "CauchyNet1_best.pth",
       # "CauchyNet0":  "CauchyNet0_best.pth",
       # "CauchyNet3":  "CauchyNet3_best.pth",
       # "CauchyNet4":  "CauchyNet4_best.pth",
        # ...
    }
    
    # 1) Create fresh model instances, load their state_dicts
    loaded_models = {}
    for model_name, ckpt_path in checkpoint_paths.items():
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] checkpoint not found => {ckpt_path}")
            continue
    
        # Create a fresh instance with the same architecture
        # NOTE: If your original training used hidden_size=1000, do the same here.
        constructor = models_dict[model_name] 
        model = constructor(input_size=1, hidden_size=1000, output_size=1)
    
        # Load the saved weights
        print(f"[INFO] Loading {model_name} from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    
        loaded_models[model_name] = model
    
    # 2) Evaluate predictions on a uniform grid for final overlay
    X_plot = torch.linspace(-0.95, 0.95, 200).to(device)
    Y_plot_true = challenging_function(X_plot.cpu()).numpy()  # unscaled "ground truth"
    
    # Re-fit (or re-use) the same scaler on original Y => ensures consistent inverse transform
    # If 'scaler' is already fit from your previous code, you might not need to call 'fit' again.
    # But if you want to confirm, do:
    # scaler.fit(Y_np)
    
    overlay_preds = {}
    for model_name, model in loaded_models.items():
        with torch.no_grad():
            out = model(X_plot.unsqueeze(-1).float())
            if isinstance(out, tuple):
                y_real, _ = out
                preds_scaled = y_real.squeeze(-1).cpu().numpy()
            else:
                preds_scaled = out.squeeze(-1).cpu().numpy()
    
        # Inverse transform
        preds_2d = preds_scaled.reshape(-1, 1)
        preds_unscaled = scaler.inverse_transform(preds_2d).flatten()
    
        overlay_preds[model_name] = preds_unscaled
    
    # (Optional) Subplot predictions
    subplot_predictions(
        X_plot.cpu().numpy(),
        Y_plot_true,
        overlay_preds,
        color_map,
        filename="subplot_predictions_from_saved_models.pdf"
    )
    
    # 3) Box plot of errors on test set (unscaled)
    error_dict = {}
    for mname, model in loaded_models.items():
        abs_errors = []
        with torch.no_grad():
            for x_batch, y_batch_scaled in test_loader:
                x_batch = x_batch.to(device)
                y_batch_scaled = y_batch_scaled.to(device)
    
                out = model(x_batch.unsqueeze(-1))
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_scaled = out.cpu().numpy().flatten()
    
                truths_scaled = y_batch_scaled.cpu().numpy().flatten()
    
                # Inverse-transform both
                preds_unscaled  = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
                truths_unscaled = scaler.inverse_transform(truths_scaled.reshape(-1,1)).flatten()
    
                abs_errors.extend(np.abs(preds_unscaled - truths_unscaled))
    
        error_dict[mname] = np.array(abs_errors)
    
    plot_box_errors(
        error_dict,
        color_map,
        filename="box_errors_from_saved_models.pdf"
    )
    
    # 4) (Optional) Print summary if 'results_dict' is in memory
    print("\n=== Summary from loaded models ===")
    for mname in loaded_models.keys():
        stats = results_dict.get(mname, None)  # if results_dict was filled after training
        if stats is not None:
            print(f"{mname:12s} => "
                  f"MSE={stats['MSE_mean']:.6f}±{stats['MSE_std']:.4f}, "
                  f"MAE={stats['MAE_mean']:.4f}±{stats['MAE_std']:.4f}, "
                  f"Time={stats['Time_mean']:.2f}±{stats['Time_std']:.2f}s, "
                  f"#Params={stats['Num_Params']}")
        else:
            print(f"{mname:12s} => No stats found in results_dict")
    
    # 5) (Optional) Plot error curves
    plot_error_curves(
        X_plot.cpu().numpy(),
        Y_plot_true,
        overlay_preds,
        color_map,
        filename="error_curves_from_saved_models.pdf"
    )
    
    print("\n=== Done with loaded models! ===")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 8 FAILED, continuing', flush=True)

print('=== CELL 9 ===', flush=True)
try:
    import torch
    import numpy as np
    import os
    
    # Suppose you have the following saved model checkpoints
    # in the same directory or a "checkpoints/" folder:
    # "CauchyNet_trained.pth", "SIREN_trained.pth", "NBeats_trained.pth", "FNN_trained.pth", etc.
    # We'll do a dictionary to match checkpoint paths to model classes:
    checkpoint_paths = {
        "CauchyNet":   "CauchyNet_trained.pth",
        "CauchyNet1":       "CauchyNet1_trained.pth",
        "CauchyNet2":      "CauchyNet2_trained.pth",
        "CauchyNet3":         "CauchyNet3_trained.pth",
        "CauchyNet4":         "CauchyNet4_trained.pth",
       # "CauchyNet5": "CauchyNet5_trained.pth"
    }
    
    # 1) Create fresh instances, load their state_dicts
    loaded_models = {}
    for model_name, ckpt_path in checkpoint_paths.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: checkpoint not found => {ckpt_path}")
            continue
    
        # Create a fresh instance with the same architecture
        # (same input_size, hidden_size, output_size, etc.)
        constructor = models_dict[model_name]  # from your original dictionary
        model = constructor(input_size=1, hidden_size=128, output_size=1)  # or however you init
    
        # Load the saved weights
        print(f"Loading {model_name} from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    
        loaded_models[model_name] = model
    
    # 2) Evaluate predictions on a uniform grid for final overlay
    #    We re-fit the same scaler on original Y to ensure consistent inverses
    X_plot = torch.linspace(-0.95, 0.95, 200).to(device)
    Y_plot_true = challenging_function(X_plot.cpu()).numpy()   # unscaled
    
    scaler.fit(Y_np)  # re-fit on original Y
    
    overlay_preds = {}
    for model_name, model in loaded_models.items():
        model.eval()
        with torch.no_grad():
            out = model(X_plot.unsqueeze(-1).float())
            if isinstance(out, tuple):
                y_real, _ = out
                preds_scaled = y_real.cpu().numpy().flatten()
            else:
                preds_scaled = out.cpu().numpy().flatten()
    
        preds_2d       = preds_scaled.reshape(-1, 1)
        preds_unscaled = scaler.inverse_transform(preds_2d).flatten()
        overlay_preds[model_name] = preds_unscaled
    
    # (Optional) Subplot predictions
    subplot_predictions(
        X_plot.cpu().numpy(),
        Y_plot_true,
        overlay_preds,
        color_map,
        filename="subplot_predictions_from_saved_models.pdf"
    )
    
    # 3) Box plot of errors on test set (unscaled)
    error_dict = {}
    for mname, model in loaded_models.items():
        model.eval()
        abs_errors = []
        with torch.no_grad():
            for x_batch, y_batch_scaled in test_loader:
                x_batch, y_batch_scaled = x_batch.to(device), y_batch_scaled.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_scaled = out.cpu().numpy().flatten()
    
                truths_scaled = y_batch_scaled.cpu().numpy().flatten()
                # Inverse transform both
                preds_unscaled  = scaler.inverse_transform(
                                    preds_scaled.reshape(-1,1)
                                  ).flatten()
                truths_unscaled = scaler.inverse_transform(
                                    truths_scaled.reshape(-1,1)
                                  ).flatten()
    
                abs_errors.extend(np.abs(preds_unscaled - truths_unscaled))
        error_dict[mname] = np.array(abs_errors)
    
    plot_box_errors(
        error_dict,
        color_map,
        filename="box_errors_from_saved_models.pdf"
    )
    
    # 4) (Optional) Print summary if you have 'results_dict' saved
    #    If you also saved results_dict to a file, load it similarly and print it again.
    print("\n=== Summary from loaded models ===")
    for mname in loaded_models.keys():
        stats = results_dict.get(mname, None)  # If 'results_dict' is in memory
        if stats is not None:
            print(f"{mname:12s} => "
                  f"MSE={stats['MSE_mean']:.6f}±{stats['MSE_std']:.4f}, "
                  f"MAE={stats['MAE_mean']:.4f}±{stats['MAE_std']:.4f}, "
                  f"Time={stats['Time_mean']:.2f}±{stats['Time_std']:.2f}s, "
                  f"#Params={stats['Num_Params']}")
        else:
            print(f"{mname:12s} => No stats found in results_dict")
    
    # 5) (Optional) Plot error curves
    plot_error_curves(
        X_plot.cpu().numpy(),
        Y_plot_true,
        overlay_preds,
        color_map,
        filename="error_curves_from_saved_models.pdf"
    )
    
    print("\n=== Done with loaded models! ===")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 9 FAILED, continuing', flush=True)

print('=== CELL 10 ===', flush=True)
try:
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from sklearn.preprocessing import MinMaxScaler
    import math
    # Import utility functions (ensure they exist in your environment)
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
    
    # =============== Transform Data ===============
    # Ensure `transform_time_series_to_domain` is defined
    Y_vals = df['Y']  # Assuming df is a pandas DataFrame
    X_mapped, Y_mapped = transform_time_series_to_domain(Y_vals)
    
    # Convert to Torch tensors
    X = torch.tensor(X_mapped, dtype=torch.float32)
    Y = torch.tensor(Y_mapped, dtype=torch.float32)
    
    # Normalize Y
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1, 1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32)
    
    # =============== Data Splits ===============
    # Load data splits (ensure `loadData` is correctly defined)
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    # =============== Model Dictionary ===============
    # Replace these placeholders with actual model classes
    models_dict = {     "CauchyNet": CauchyNet1,
                  # "CauchyNet0": CauchyNet,
        "CauchyNet1": CauchyNet0,
        "CauchyNet2": CauchyNet_NonHolomorphic,
        "CauchyNet3": CauchyNet_RealActivation,
        "CauchyNet4": CauchyNet_NoImagPenalty,
        "CauchyNet5": CauchyNet_PurelyRealParams,
    }
    # Generate color map for models
    model_names = list(models_dict.keys())
    color_map = get_model_colors(model_names, emphasize_model="CauchyNet")
    
    # =============== Train Configuration ===============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size, hidden_size, output_size = 1, 128, 1
    lr, epochs, num_repeats = 0.005, 200, 5
    
    # =============== Train Multiple Runs & Collect Logs ===============
    interval_logs = {}
    results_dict = {}
    
    for model_name, constructor in models_dict.items():
        print(f"=== Training {model_name} ===")
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        num_params = None
    
        for _ in range(num_repeats):
            model, train_losses, val_losses, test_mse, test_mae, preds_scaled, truths_scaled, training_time, num_params = train_and_evaluate_model(
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
    
        # Compute statistics
        results_dict[model_name] = {
            'MSE_mean': np.mean(runs_mse),
            'MSE_std': np.std(runs_mse),
            'MAE_mean': np.mean(runs_mae),
            'MAE_std': np.std(runs_mae),
            'Time_mean': np.mean(runs_time),
            'Time_std': np.std(runs_time),
            'Num_Params': num_params
        }
        interval_logs[model_name] = {'train': train_runs, 'val': val_runs}
    
    # =============== Plot Training Curves ===============
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves2.pdf")
    
    # =============== Final Run for Each Model ===============
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
    print('!! CELL 10 FAILED, continuing', flush=True)

print('=== CELL 11 ===', flush=True)
try:
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from sklearn.preprocessing import MinMaxScaler
    import math
    # Import utility functions (ensure they exist in your environment)
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
    
    # =============== Transform Data ===============
    # # Ensure `transform_time_series_to_domain` is defined
    # Y_vals = df['Y']  # Assuming df is a pandas DataFrame
    # X_mapped, Y_mapped = transform_time_series_to_domain(Y_vals)
    
    # # Convert to Torch tensors
    # X = torch.tensor(X_mapped, dtype=torch.float32)
    # Y = torch.tensor(Y_mapped, dtype=torch.float32)
    X = torch.linspace(-0.95, 0.95, 200).to(device)
    Y = challenging_function(X.cpu())#.numpy()   # unscaled
    
    scaler.fit(Y_np)  # re-fit on original Y
    
    # Normalize Y
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1, 1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32)
    
    # =============== Data Splits ===============
    # Load data splits (ensure `loadData` is correctly defined)
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    # =============== Model Dictionary ===============
    # Replace these placeholders with actual model classes
    models_dict = { "CauchyNet": CauchyNet1,
        #"CauchyNet0": CauchyNet0,
        "CauchyNet1": CauchyNet0,
        "CauchyNet2": CauchyNet_NonHolomorphic,
        "CauchyNet3": CauchyNet_RealActivation,
        "CauchyNet4": CauchyNet_NoImagPenalty,
       # "CauchyNet5": CauchyNet_PurelyRealParams,
    }
    
    # Generate color map for models
    model_names = list(models_dict.keys())
    color_map = get_model_colors(model_names, emphasize_model="CauchyNet")
    
    # =============== Train Configuration ===============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size, hidden_size, output_size = 1, 128, 1
    lr, epochs, num_repeats = 0.01, 200, 5
    
    # =============== Train Multiple Runs & Collect Logs ===============
    interval_logs = {}
    results_dict = {}
    
    for model_name, constructor in models_dict.items():
        print(f"=== Training {model_name} ===")
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        num_params = None
    
        for _ in range(num_repeats):
            model, train_losses, val_losses, test_mse, test_mae, preds_scaled, truths_scaled, training_time, num_params = train_and_evaluate_model(
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
    
        # Compute statistics
        results_dict[model_name] = {
            'MSE_mean': np.mean(runs_mse),
            'MSE_std': np.std(runs_mse),
            'MAE_mean': np.mean(runs_mae),
            'MAE_std': np.std(runs_mae),
            'Time_mean': np.mean(runs_time),
            'Time_std': np.std(runs_time),
            'Num_Params': num_params
        }
        interval_logs[model_name] = {'train': train_runs, 'val': val_runs}
    
    # =============== Plot Training Curves ===============
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves3.pdf")
    
    # =============== Final Run for Each Model ===============
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
    print('!! CELL 11 FAILED, continuing', flush=True)

print('=== CELL 12 ===', flush=True)
try:
    # =============== Plot Training Curves ===============
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves3.pdf")
    
    # =============== Final Run for Each Model ===============
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
    print('!! CELL 12 FAILED, continuing', flush=True)

print('=== CELL 13 ===', flush=True)
try:
    import math
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.data import TensorDataset, DataLoader
    
    #################### 1) Our function and derivative ####################
    def f(x: torch.Tensor) -> torch.Tensor:
        """
        Example:  f(x) = sin(2x) + 0.05/((x-1)^2 + 0.1) + 0.01/((x+0.5)^2 + 0.05)
                   - 0.01*x^2 + 0.01*x^3
    
        x: shape [N] or [N,1]
        returns shape [N].
        """
        sinusoidal = torch.sin(2*(x-2)) + 0.5*torch.cos(5*(x-1))
        rational1  = 0.05 / ((x - 1)**2 + 0.1)
        rational2  = 0.01 / ((x + 0.5)**2 + 0.05)
        polynomial = -0.01*(x**2) + 0.01*(x**3)
        return sinusoidal + rational1 + rational2 + polynomial
    
    def df(x: torch.Tensor) -> torch.Tensor:
        """
        Derivative of f(x).
        """
        # derivative of sin(2x) + 0.5 cos(5x) => 2 cos(2x) - 2.5 sin(5x)
        d_sin2x = 2*torch.cos(2*(x-2)) - 2.5*torch.sin(5*(x-1))
    
        # derivative(0.05 / ((x-1)^2 + 0.1))
        denom1  = (x - 1)**2 + 0.1
        d_rat1  = 0.05 * (-2*(x - 1)) / (denom1**2)
    
        # derivative(0.01 / ((x+0.5)^2 + 0.05))
        denom2  = (x + 0.5)**2 + 0.05
        d_rat2  = 0.01 * (-2*(x+0.5)) / (denom2**2)
    
        # derivative of polynomial => -0.02*x + 0.03*x^2
        d_poly  = -0.02*x + 0.03*(x**2)
        return d_sin2x + d_rat1 + d_rat2 + d_poly
    
    
    #################### 2) Identify turning points ####################
    sample_size  = 2000
    X1 = torch.linspace(-2, 2, sample_size)  # scanning range
    with torch.no_grad():
        dY1 = df(X1)                          # approximate derivative
    
    # We'll treat pointsdd where |dY1| < 0.15 as turning points or "sensitive" regions
    turn_mask      = torch.abs(dY1) < 0.15
    turning_points = X1[turn_mask]
    
    
    #################### 3) Sample test data near turning points ####################
    # We want ~250 test points, all within 0.15 of some turning point
    test_size = 250
    test_X_list = []
    while len(test_X_list) < test_size:
        samp = torch.normal(0., 1., size=(1,))*2.
        samp = torch.clamp(samp, -2, 2)
        # If close to *any* turning point => put in test set
        if any(torch.abs(turning_points - samp) < 0.15):
            test_X_list.append(samp)
    
    test_X = torch.cat(test_X_list)
    test_Y = f(test_X)
    
    
    #################### 4) Sample “standard” train data, all >0.15 from turning points ####################
    train_size_standard = 350  # We want a total of e.g. 350 training points
    train_X_list = []
    while len(train_X_list) < train_size_standard:
        samp = torch.normal(0., 1., size=(1,))*2.
        samp = torch.clamp(samp, -2, 2)
        # If not near turning points => put in train
        if not any(torch.abs(turning_points - samp) < 0.15):
            train_X_list.append(samp)
    
    train_X = torch.cat(train_X_list)  # shape [350]
    train_Y = f(train_X)
    
    #################### 5) Validation set, also away from turning points ####################
    val_size = 100
    val_X_list = []
    while len(val_X_list) < val_size:
        samp = torch.normal(0., 1., size=(1,))*2.
        samp = torch.clamp(samp, -2, 2)
        if not any(torch.abs(turning_points - samp) < 0.15):
            val_X_list.append(samp)
    
    val_X = torch.cat(val_X_list)
    val_Y = f(val_X)
    
    print(f"Final:\nTrain: {len(train_X)}\nVal:   {len(val_X)}\nTest:  {len(test_X)}")
    
    #################### 6) Build DataLoaders ####################
    from torch.utils.data import TensorDataset, DataLoader
    
    train_ds = TensorDataset(train_X, train_Y)
    val_ds   = TensorDataset(val_X,   val_Y)
    test_ds  = TensorDataset(test_X,  test_Y)
    
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    
    #################### 7) Quick Visualization of sample distribution ####################
    X_plot = torch.linspace(-2, 2, 500)
    Y_plot = f(X_plot)
    
    plt.figure(figsize=(6,4))
    plt.plot(X_plot.numpy(), Y_plot.numpy(), 'k-', alpha=0.4, lw=2, label="f(x)")
    
    plt.scatter(train_X.numpy(), train_Y.numpy(),
                color='blue', marker='o', s=20, alpha=0.8, label="Train")
    plt.scatter(val_X.numpy(), val_Y.numpy(),
                color='green', marker='s', s=20, alpha=0.8, label="Validation")
    plt.scatter(test_X.numpy(), test_Y.numpy(),
                color='red', marker='^', s=20, alpha=0.8, label="Test (near TPs)")
    
    #plt.title("Data Splits:\nTrain, Val, Test", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.ylabel("Y", fontsize=11)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 13 FAILED, continuing', flush=True)

print('=== CELL 14 ===', flush=True)
try:
    def challenging_function(x):
        """
        A highly non-symmetric function designed to highlight differences between CauchyNet
        and sine-activation-based models like SIREN.
        """
        return (
            1 / ((x + 0.6)**2 + 0.005)   # Sharp positive peak at x=0.1
                        - 40*torch.exp(-2 * (x + 0.4)**2)  # Localized Gaussian bump at x=0.4
                     +x**2 / ((x - 0.8)**4 + 0.01)     # Sharp negative peak at x=0.8
                + 50 * torch.sign(x) * torch.abs(torch.sin(3*x)+0.8)**1.5 * torch.sin(10*x) # Asymmetric cubic-like feature
        )
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 14 FAILED, continuing', flush=True)

print('=== CELL 15 ===', flush=True)
try:
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from sklearn.preprocessing import MinMaxScaler
    import math
    
    # Import utility functions (ensure they exist in your environment)
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
    
    # =============== Transform Data ===============
    # Ensure `transform_time_series_to_domain` is defined
    # Y_vals = df['Y']  # Assuming df is a pandas DataFrame
    # X_mapped, Y_mapped = transform_time_series_to_domain(Y_vals)
    
    # # Convert to Torch tensors
    # X = torch.tensor(X_mapped, dtype=torch.float32)
    # Y = torch.tensor(Y_mapped, dtype=torch.float32)
    
    # Normalize Y
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1, 1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32)
    
    # =============== Data Splits ===============
    # Load data splits (ensure `loadData` is correctly defined)
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    # =============== Model Dictionary ===============
    # Replace these placeholders with actual model classes
    models_dict = {
       "CauchyNet": CauchyNet1,
       # "CauchyNet0": CauchyNet0,
        "CauchyNet1": CauchyNet0,
        "CauchyNet2": CauchyNet_NonHolomorphic,
        "CauchyNet3": CauchyNet_RealActivation,
        "CauchyNet4": CauchyNet_NoImagPenalty,
        # "CauchyNet0": CauchyNet0,
        # "CauchyNet1": CauchyNet1
    }
    
    # Generate color map for models
    model_names = list(models_dict.keys())
    color_map = get_model_colors(model_names, emphasize_model="CauchyNet")
    
    # =============== Train Configuration ===============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size, hidden_size, output_size = 1, 128, 1
    lr, epochs, num_repeats = 0.01, 200, 5
    
    # Number of random seeds to test
    random_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # =============== Train Multiple Runs & Collect Logs ===============
    # We'll store MSE for each seed in a dictionary: final_mse_by_seed[model_name] = [mse_seed0, mse_seed1, ...]
    final_mse_by_seed = {model_name: [] for model_name in model_names}
    
    # Also store your usual aggregated results
    results_dict = {}
    interval_logs = {}
    
    for model_name, constructor in models_dict.items():
        print(f"=== Training {model_name} ===")
    
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs = [], []
        num_params = None
    
        for seed in random_seeds:
            # Set the random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
    
            # Train and evaluate model
            model, train_losses, val_losses, test_mse, test_mae, preds_scaled, truths_scaled, training_time, num_params = train_and_evaluate_model(
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
    
        # Store the per-seed MSE results
        final_mse_by_seed[model_name] = runs_mse
    
        # Compute aggregate statistics
        results_dict[model_name] = {
            'MSE_mean': np.mean(runs_mse),
            'MSE_std': np.std(runs_mse),
            'MAE_mean': np.mean(runs_mae),
            'MAE_std': np.std(runs_mae),
            'Time_mean': np.mean(runs_time),
            'Time_std': np.std(runs_time),
            'Num_Params': num_params
        }
        interval_logs[model_name] = {'train': train_runs, 'val': val_runs}
    
    # =============== Print Aggregated Results ===============
    for model_name in model_names:
        print(f"{model_name}: {results_dict[model_name]}")
    
    # =============== Plot MSE vs. Random Seed ===============
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    plt.figure(figsize=(10, 6))
    
    for model_name in model_names:
        mse_values = final_mse_by_seed[model_name]
        plt.plot(
            random_seeds, 
            mse_values, 
            marker='o', 
            label=model_name, 
            color=color_map[model_name]
        )
    
    plt.xlabel("Random Seed")
    plt.ylabel("Test MSE")
    plt.title("Final Test MSE vs. Random Seed")
    plt.legend()
    plt.tight_layout()
    output_path = "figures/final_test_mse_vs_random_seed.pdf"
    plt.savefig(output_path)
    plt.show()
    
    # =============== Optionally Plot Training Curves with Confidence ===============
    plot_training_curves_with_confidence(interval_logs, model_names, color_map, filename="training_curves3.pdf")
    
    # =============== Final Run for Each Model (Optional) ===============
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
    print('!! CELL 15 FAILED, continuing', flush=True)

print('=== CELL 16 ===', flush=True)
try:
    # =============== Plot MSE vs. Random Seed ===============
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    plt.figure(figsize=(10, 6))
    
    for model_name in model_names:
        mse_values = final_mse_by_seed[model_name]
        plt.plot(
            random_seeds, 
            mse_values, 
            marker='o', 
            label=model_name, 
            color=color_map[model_name]
        )
    
    plt.xlabel("Random Seed")
    plt.ylabel("Test MSE")
    plt.title("Final Test MSE vs. Random Seed")
    plt.legend()
    plt.tight_layout()
    output_path = "figures/final_test_mse_vs_random_seed.pdf"
    plt.savefig(output_path)
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 16 FAILED, continuing', flush=True)

print('=== CELL 17 ===', flush=True)
try:
    import random
    import torch
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, Subset
    
    # ============ 1) Experimental Setup ============
    
    # Suppose these are defined somewhere:
    # - models_dict: {"CauchyNet": CauchyNetConstructor, "CauchyNet1": CauchyNet1Constructor, ...}
    # - train_and_evaluate_model: your training function returning (model, train_losses, val_losses, test_mse, ...)
    # - ds_test: your dataset for demonstration, from which we sample
    # - val_loader, test_loader: already built
    # - device: CPU or CUDA
    # - input_size, output_size: typically 1 if it's a 1D input => 1D output
    # - random, torch, etc. already imported
    
    model_names = ["CauchyNet"]  # or however many you want
    models_dict = {
        "CauchyNet": CauchyNet1, 
        # "CauchyNet1": AnotherConstructor,
        # ...
    }
    
    data_sizes   = [100, 300, 600, 1200]  # as in your example
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    epoch_list   = [2000]
    
    input_size   = 1
    output_size  = 1
    lr           = 0.01
    num_repeats  = 1
    
    # We'll store results like: results_grid[model][data_size][hidden_size][epochs] = { "MSE":..., "MAE":..., ... }
    results_grid = {
        model_name: {
            ds: {hs: {} for hs in hidden_sizes} 
            for ds in data_sizes
        }
        for model_name in model_names
    }
    
    # ============ 2) Run Experiments & Populate results_grid ============
    for ds in data_sizes:
        print(f"\n=== Testing with data_size={ds} ===")
    
        # We'll sample a random subset of ds_test to create a "train set"
        effective_data_size = min(ds, len(ds_test)) 
        subset_indices = random.sample(range(len(ds_test)), effective_data_size)
        subset_train_loader = DataLoader(Subset(ds_test, subset_indices), batch_size=32, shuffle=True)
    
        for hs in hidden_sizes:
            for ep in epoch_list:
                print(f"  >> hidden_size={hs}, epochs={ep}")
                for model_name in model_names:
                    constructor = models_dict[model_name]
                    print(f"       >> Training {model_name}")
                    (
                        model,
                        train_losses,
                        val_losses,
                        test_mse,
                        test_mae,
                        preds_scaled,
                        truths_scaled,
                        training_time,
                        num_params
                    ) = train_and_evaluate_model(
                        model_constructor = constructor,
                        input_size        = input_size,
                        hidden_size       = hs,
                        output_size       = output_size,
                        train_loader      = subset_train_loader,
                        val_loader        = val_loader,
                        test_loader       = test_loader,
                        lr                = lr,
                        epochs            = ep,
                        device            = device
                    )
    
                    # Store results in results_grid
                    results_grid[model_name][ds][hs][ep] = {
                        "MSE": test_mse,
                        "MAE": test_mae,
                        "Time": training_time,
                        "Num_Params": num_params
                    }
    
    # ============ 3) Convert results_grid to DataFrame for Plotting ============
    plot_data = []
    for model_name in model_names:
        for ds in data_sizes:
            for hs in hidden_sizes:
                for ep in epoch_list:
                    stats = results_grid[model_name][ds][hs].get(ep)
                    if stats is None:
                        # If no result stored, skip
                        continue
                    plot_data.append({
                        "Model":       model_name,
                        "Data Size":   ds,
                        "Hidden Size": hs,
                        "Epochs":      ep,
                        "MSE":         stats["MSE"],
                        "MAE":         stats["MAE"],
                        "Time":        stats["Time"],
                        "Num_Params":  stats["Num_Params"]
                    })
    
    df = pd.DataFrame(plot_data)
    # e.g. columns => Model, Data Size, Hidden Size, Epochs, MSE, MAE, Time, Num_Params
    
    # ============ 4) Create Heatmaps for MSE ============
    
    # Suppose you want a heatmap for each model, pivoted by "Data Size" vs "Hidden Size"
    # We'll assume there's only one 'epoch' in epoch_list for simplicity. 
    # If you have multiple epochs, you could pivot on that as well.
    
    for model in model_names:
        df_model = df[df["Model"] == model]
        
        # pivot => rows=Hidden Size, cols=Data Size, values=MSE
        pivot_data = df_model.pivot(index="Hidden Size", columns="Data Size", values="MSE")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".6f",
            cmap="viridis",
            cbar_kws={"label": "MSE"},
        )
        plt.title(f"Heatmap of MSE for {model}", fontsize=16)
        plt.xlabel("Data Size", fontsize=14)
        plt.ylabel("Hidden Size", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # ============ 5) Print or Summarize Results ============
    print("\n===== Summary of Results =====")
    for model_name in model_names:
        print(f"\n--- {model_name} ---")
        for ds in data_sizes:
            for hs in hidden_sizes:
                for ep in epoch_list:
                    stats = results_grid[model_name][ds][hs].get(ep)
                    if stats is None:
                        continue
                    print(
                        f"Data Size={ds}, Hidden={hs}, Epochs={ep} | "
                        f"MSE={stats['MSE']:.6f}, MAE={stats['MAE']:.6f}, "
                        f"Time={stats['Time']:.2f}s, Num_Params={stats['Num_Params']}"
                    )
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 17 FAILED, continuing', flush=True)

print('=== CELL 18 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LogNorm
    
    # Suppose 'df' is your DataFrame with columns [Model, Data Size, Hidden Size, MSE, ...]
    
    for model in model_names:
        df_model = df[df["Model"] == model]
        # pivot => rows=Hidden Size, cols=Data Size, values=MSE
        pivot_data = df_model.pivot(index="Hidden Size", columns="Data Size", values="MSE")
    
        plt.figure(figsize=(8, 6))
    
        # If you know a typical min/max range of your MSE, set them in LogNorm
        # For example, vmin=1e-6, vmax=1e-1 if your MSEs range from ~1e-6 to 1e-1
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2e",            # scientific notation
            cmap="magma",         # or "plasma", "viridis", etc.
            norm=LogNorm(vmin=1e-6, vmax=1e-1),
            cbar_kws={"label": "MSE (log scale)"}
        )
        plt.title(f"Heatmap of MSE for {model} (Log Scale)", fontsize=16)
        plt.xlabel("Data Size", fontsize=14)
        plt.ylabel("Hidden Size", fontsize=14)
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 18 FAILED, continuing', flush=True)

print('=== CELL 19 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LogNorm
    
    # Suppose `df` is your DataFrame with columns:
    # [ "Model", "Data Size", "Hidden Size", "MSE", ... ]
    # and we want to plot a heatmap of MSE for each model.
    
    for model in model_names:
        df_model = df[df["Model"] == model]
        
        # Pivot => rows=Hidden Size, cols=Data Size, values=MSE
        pivot_data = df_model.pivot(index="Hidden Size", columns="Data Size", values="MSE")
    
        plt.figure(figsize=(8, 6))
    
        # If MSE spans multiple orders of magnitude, use LogNorm:
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2e",          # scientific notation (e.g. 1.23e-4)
            cmap="magma",       # color-blind-friendly colormap
            norm=LogNorm(vmin=1e-6, vmax=1e-1),  # adjust vmin/vmax to your MSE range
            cbar_kws={"label": "MSE (log scale)"}
        )
        plt.title(f"Heatmap of MSE for {model} (Log Scale)", fontsize=14)
        plt.xlabel("Data Size", fontsize=12)
        plt.ylabel("Hidden Size", fontsize=12)
    
        # Save at 300 DPI, tight layout for publication
        plt.savefig(f"{model}_heatmap.png", dpi=300, bbox_inches="tight")
    
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 19 FAILED, continuing', flush=True)

print('=== CELL 20 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import matplotlib.colors as mcolors
    
    # Suppose 'df' is your DataFrame with columns:
    #    ["Model", "Data Size", "Hidden Size", "MSE"]
    # and you only plot for one model, e.g. "CauchyNet".
    # If you have multiple models, loop over each model similarly.
    
    model = "CauchyNet"
    df_model = df[df["Model"] == model]
    
    # Pivot => rows=Hidden Size, cols=Data Size, values=MSE
    pivot_data = df_model.pivot(index="Hidden Size", columns="Data Size", values="MSE")
    
    plt.figure(figsize=(8, 6))
    
    # PowerNorm with gamma < 1 => stretches small differences, e.g. 0.5
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt="0.2e",            # scientific notation for small MSE
        cmap="magma",          # color-blind-friendly
        norm=mcolors.PowerNorm(gamma=0.2, vmin=1e-6, vmax=1e-3),
        cbar=False             # <-- No color bar
    )
    
    plt.title(f"Heatmap of MSE for {model} (Power Norm)", fontsize=14)
    plt.xlabel("Data Size", fontsize=12)
    plt.ylabel("Hidden Size", fontsize=12)
    
    # Save figure at 300 DPI for publication
    plt.savefig(f"{model}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 20 FAILED, continuing', flush=True)

print('=== CELL 21 ===', flush=True)
try:
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import matplotlib.colors as mcolors
    
    # Suppose 'df' is your DataFrame with columns:
    #    ["Model", "Data Size", "Hidden Size", "MSE"]
    # and you only plot for one model, e.g. "CauchyNet".
    # If you have multiple models, loop over each model similarly.
    
    model = "CauchyNet"
    df_model = df[df["Model"] == model]
    
    # Pivot => rows=Hidden Size, cols=Data Size, values=MSE
    pivot_data = df_model.pivot(index="Hidden Size", columns="Data Size", values="MSE")
    
    plt.figure(figsize=(8, 6))
    
    # PowerNorm with gamma < 1 => stretches small differences, e.g. 0.5
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt="0.2e",            # scientific notation for small MSE
        cmap="magma",         # color-blind-friendly
        norm=mcolors.PowerNorm(gamma=0.2, vmin=1e-6, vmax=1e-3),
        cbar_kws={"label": "MSE (power scale)"}
    )
    
    plt.title(f"Heatmap of MSE for {model} (Power Norm)", fontsize=14)
    plt.xlabel("Data Size", fontsize=12)
    plt.ylabel("Hidden Size", fontsize=12)
    
    # Save figure at 300 DPI for publication
    plt.savefig(f"{model}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 21 FAILED, continuing', flush=True)

print('=== CELL 22 ===', flush=True)
try:
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def train_and_evaluate_model(
        model_constructor, 
        input_size, hidden_size, output_size,
        train_loader, val_loader, test_loader,
        lr=0.01,
        epochs=200,
        device=None,
        scaler=None,
        weight_decay=1e-10,   # <--- Exposed parameter for weight decay
         model_name="MyModel",
    ):
        """
        Simple training loop with Adam + MSE, with optional weight decay regularization.
        Returns model + logs + test metrics, where test errors are computed on
        the ORIGINAL (unscaled) data if 'scaler' is provided.
    
        Params:
        -------
        model_constructor : Callable
            A function/class that, given (input_size, hidden_size, output_size),
            returns an nn.Module (e.g. standard net or CauchyNet).
        input_size, hidden_size, output_size : int
            Network dimensions.
        train_loader, val_loader, test_loader : DataLoader
            Data loaders for training, validation, and test sets.
        lr : float
            Learning rate for Adam.
        epochs : int
            Number of training epochs.
        device : torch.device or None
            If None, we auto-select CUDA if available, else CPU.
        scaler : object or None
            A fitted scaler with 'inverse_transform()' for Y.
            If None, assume data is already unscaled.
        weight_decay : float
            L2 regularization strength (default=1e-10).
            Set to 0.0 if you want no weight decay.
    
        Returns
        -------
        model : nn.Module
            The trained model (best checkpoint loaded by val_loss).
        train_losses : list of float
            Per-epoch training MSE loss.
        val_losses : list of float
            Per-epoch validation MSE loss.
        test_mse : float
            Test MSE, computed on unscaled data if 'scaler' is provided.
        test_mae : float
            Test MAE, computed on unscaled data if 'scaler' is provided.
        preds_all_unscaled : np.ndarray
            Final test predictions in original domain if scaler is provided.
        truths_all_unscaled : np.ndarray
            Final test labels in original domain if scaler is provided.
        training_time : float
            Total training time (seconds).
        Num_Params : int
            Number of trainable parameters in the model.
        """
    
        # ------------------ Device Setup ------------------
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # ------------------ Model Construction ------------------
        model = model_constructor(input_size, hidden_size, output_size).to(device)
    
        criterion = nn.MSELoss()
        # Use the weight_decay parameter here
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
        # We can keep the same plateau scheduler as your original code
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
                    loss = (
                        criterion(y_real, y_batch_scaled.unsqueeze(-1)) 
                        + criterion(y_imag, torch.zeros_like(y_imag))
                    )
                else:
                    loss = criterion(out, y_batch_scaled.unsqueeze(-1))
    
                loss.backward()
                # Gradient clipping is optional; keep if you like
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
                        val_loss = (
                            criterion(y_real, y_batch_scaled.unsqueeze(-1))
                            + criterion(y_imag, torch.zeros_like(y_imag))
                        )
                    else:
                        val_loss = criterion(out, y_batch_scaled.unsqueeze(-1))
    
                    val_loss_accum += val_loss.item()
    
            val_loss = val_loss_accum / len(val_loader)
            val_losses.append(val_loss)
    
            # Store best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = model.state_dict()
    
            # Step the scheduler (ReduceLROnPlateau expects a metric)
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
                x_batch       = x_batch.to(device)
                y_batch_scaled = y_batch_scaled.to(device)
    
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_scaled = out.cpu().numpy().flatten()
    
                truths_scaled = y_batch_scaled.cpu().numpy().flatten()
    
                # Inverse-transform if we have a scaler
                if scaler is not None:
                    preds_unscaled  = scaler.inverse_transform(
                                        preds_scaled.reshape(-1, 1)
                                     ).flatten()
                    truths_unscaled = scaler.inverse_transform(
                                        truths_scaled.reshape(-1, 1)
                                     ).flatten()
                else:
                    preds_unscaled  = preds_scaled
                    truths_unscaled = truths_scaled
    
                preds_list_unscaled.append(preds_unscaled)
                truths_list_unscaled.append(truths_unscaled)
    
        preds_all_unscaled  = np.concatenate(preds_list_unscaled)
        truths_all_unscaled = np.concatenate(truths_list_unscaled)
    
        # Compute test metrics in the original domain
        test_mse = mean_squared_error(truths_all_unscaled, preds_all_unscaled)
        test_mae = mean_absolute_error(truths_all_unscaled, preds_all_unscaled)
    
        return (
            model,
            train_losses,
            val_losses,
            test_mse,
            test_mae,
            preds_all_unscaled,   # return unscaled predictions
            truths_all_unscaled,  # return unscaled truths
            training_time,
            Num_Params
        )
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 22 FAILED, continuing', flush=True)

print('=== CELL 23 ===', flush=True)
try:
    import random
    import torch
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # ============ 1) Experimental Setup ============
    
    # Suppose these are defined somewhere:
    # - models_dict: {"CauchyNet": CauchyNetConstructor, "CauchyNetAblation": CauchyNetAblationConstructor, ...}
    # - train_and_evaluate_model: your training function returning
    #       (model, train_losses, val_losses, test_mse, test_mae, preds_unscaled, truths_unscaled, training_time, num_params)
    # - train_loader, val_loader, test_loader: DataLoaders
    # - device: CPU or CUDA
    # - input_size, hidden_size, output_size: for your model
    # - random, torch, etc. already imported if needed
    
    model_names = ["CauchyNet"]  # or however many you want from models_dict
    models_dict = {
        "CauchyNet": CauchyNet1, 
        # "CauchyNetAblation": AnotherConstructor,
    }
    
    # We'll vary learning_rate and weight_decay
    learning_rates = [1e-3, 1e-2, 1e-1]
    weight_decays  = [0.0, 1e-5, 1e-4]
    
    epochs    = 400  # example, can be changed
    input_size  = 1
    hidden_size = 128
    output_size = 1
    num_repeats = 1    # how many times to repeat each experiment
    
    # We'll store results like results_grid[model][lr][wd] = { "MSE":..., "MAE":..., ... }
    results_grid = {
        model_name: {
            lr: {wd: {} for wd in weight_decays}
            for lr in learning_rates
        }
        for model_name in model_names
    }
    
    # ============ 2) Run Experiments & Populate results_grid ============
    for model_name in model_names:
        print(f"\n=== Training/Testing Model: {model_name} ===")
        constructor = models_dict[model_name]
        
        for lr in learning_rates:
            for wd in weight_decays:
                print(f"  >> LR={lr}, WD={wd} for {model_name}")
                
                # (Optional) repeat the same setting multiple times if you want an average
                runs_mse, runs_mae, runs_time = [], [], []
                for _ in range(num_repeats):
                    # Train & evaluate
                    (
                        model,
                        train_losses,
                        val_losses,
                        test_mse,
                        test_mae,
                        preds_scaled,
                        truths_scaled,
                        training_time,
                        num_params
                    ) = train_and_evaluate_model(
                        model_constructor = constructor,
                        input_size        = input_size,
                        hidden_size       = hidden_size,
                        output_size       = output_size,
                        train_loader      = train_loader,
                        val_loader        = val_loader,
                        test_loader       = test_loader,
                        lr                = lr,
                        weight_decay      = wd,
                        epochs            = epochs,
                        device            = device
                    )
                    runs_mse.append(test_mse)
                    runs_mae.append(test_mae)
                    runs_time.append(training_time)
                
                # If num_repeats > 1, you can store mean & std; for now, we store just the last or average
                mean_mse = sum(runs_mse)  / num_repeats
                mean_mae = sum(runs_mae)  / num_repeats
                mean_time= sum(runs_time) / num_repeats
                
                # Save in results_grid
                results_grid[model_name][lr][wd] = {
                    "MSE":        mean_mse,
                    "MAE":        mean_mae,
                    "Time":       mean_time,
                    "Num_Params": num_params
                }
    
    # ============ 3) Convert results_grid to DataFrame for Plotting ============
    plot_data = []
    for model_name in model_names:
        for lr in learning_rates:
            for wd in weight_decays:
                stats = results_grid[model_name][lr][wd]
                if not stats:
                    continue  # skip empty
                plot_data.append({
                    "Model":        model_name,
                    "Learning Rate":lr,
                    "Weight Decay": wd,
                    "MSE":          stats["MSE"],
                    "MAE":          stats["MAE"],
                    "Time":         stats["Time"],
                    "Num_Params":   stats["Num_Params"]
                })
    
    df = pd.DataFrame(plot_data)
    # e.g. columns => Model, Learning Rate, Weight Decay, MSE, MAE, Time, Num_Params
    
    # ============ 4) Create Heatmap for MSE ============
    
    # Suppose you want a heatmap for each model, pivoted by "Learning Rate" vs "Weight Decay"
    for model in model_names:
        df_model = df[df["Model"] == model]
        
        # pivot => rows=Weight Decay, cols=Learning Rate, values=MSE
        pivot_data = df_model.pivot(index="Weight Decay", columns="Learning Rate", values="MSE")
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".6f",
            cmap="viridis",
            cbar_kws={"label": "MSE"},
        )
        plt.title(f"Heatmap of MSE for {model}\n(epochs={epochs}, hidden={hidden_size})")
        plt.xlabel("Learning Rate")
        plt.ylabel("Weight Decay")
        plt.tight_layout()
        plt.show()
    
    # ============ 5) Print or Summarize Results ============
    print("\n===== Summary of Results =====")
    for model_name in model_names:
        print(f"\n--- {model_name} ---")
        for lr in learning_rates:
            for wd in weight_decays:
                stats = results_grid[model_name][lr][wd]
                if not stats:
                    continue
                print(
                    f"LR={lr}, WD={wd} | "
                    f"MSE={stats['MSE']:.6f}, MAE={stats['MAE']:.6f}, "
                    f"Time={stats['Time']:.2f}s, Num_Params={stats['Num_Params']}"
                )
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 23 FAILED, continuing', flush=True)

print('=== CELL 24 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import matplotlib.colors as mcolors
    
    # Suppose 'df' is your DataFrame with columns:
    #    ["Model", "Learning Rate", "Weight Decay", "MSE"]
    # and you only want to plot for one model, e.g. "CauchyNet".
    # If you have multiple models, loop similarly or filter each one in turn.
    
    model = "CauchyNet"
    df_model = df[df["Model"] == model]
    
    # Pivot => rows=Weight Decay, columns=Learning Rate, values=MSE
    pivot_data = df_model.pivot(index="Weight Decay", columns="Learning Rate", values="MSE")
    
    plt.figure(figsize=(8, 6))
    
    # Use PowerNorm(gamma=0.2) to stretch small differences in MSE
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt="0.2e",             # scientific notation for small MSE
        cmap="magma",           # color-blind-friendly
        norm=mcolors.PowerNorm(gamma=0.2, vmin=1e-6, vmax=1e-4),
        cbar=False 
    )
    
    plt.title(f"Heatmap of MSE for {model} (Power Norm)", fontsize=14)
    plt.xlabel("Learning Rate", fontsize=12)
    plt.ylabel("Weight Decay", fontsize=12)
    
    # Save at 300 DPI for publication
    plt.savefig(f"{model}_heatmap2.png", dpi=300, bbox_inches="tight")
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 24 FAILED, continuing', flush=True)

print('=== CELL 25 ===', flush=True)
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # We'll assume 'df' has columns: "Data Size", "Hidden Size", "MSE", "Model"
    # For bubble size, we pass "size=MSE" and set a range with `sizes=(min_size, max_size)`.
    
    for model in model_names:
        df_model = df[df["Model"] == model]
        
        plt.figure(figsize=(8, 6))
        scatter = sns.scatterplot(
            data=df_model,
            x="Data Size",
            y="Hidden Size",
            size="MSE",
            hue="MSE",            # color by MSE as well
            palette="magma",      # or another colormap
            sizes=(20, 2000),     # scale bubble sizes to a suitable range
            alpha=0.7
        )
    
        # Optionally force log-scale on color if values vary widely
        # E.g., you can map "MSE" to a LogNorm by manually creating a color mapping,
        # but that is more advanced. This simpler example uses a linear scale.
    
        plt.title(f"Bubble Plot of MSE for {model}")
        plt.xlabel("Data Size")
        plt.ylabel("Hidden Size")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 25 FAILED, continuing', flush=True)

print('=== CELL 26 ===', flush=True)
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Suppose your DataFrame 'df' has multiple runs per (Model, Data Size, Hidden Size),
    # each row has a single 'MSE'. Then:
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x="Model",
        y="MSE",
        hue="Data Size",  # optional, to see separate violins per data size
        split=True        # puts them side by side within each category
    )
    plt.yscale("log")  # If MSE spans multiple orders of magnitude, log scale helps
    plt.title("Violin Plot of MSE distributions", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 26 FAILED, continuing', flush=True)

print('=== CELL 27 ===', flush=True)
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # We'll assume 'df' has columns: "Data Size", "Hidden Size", "MSE", "Model"
    # For bubble size, we pass "size=MSE" and set a range with `sizes=(min_size, max_size)`.
    
    for model in model_names:
        df_model = df[df["Model"] == model]
        
        plt.figure(figsize=(8, 6))
        scatter = sns.scatterplot(
            data=df_model,
            x="Data Size",
            y="Hidden Size",
            size="MSE",
            hue="MSE",            # color by MSE as well
            palette="magma",      # or another colormap
            sizes=(20, 2000),     # scale bubble sizes to a suitable range
            alpha=0.7
        )
    
        # Optionally force log-scale on color if values vary widely
        # E.g., you can map "MSE" to a LogNorm by manually creating a color mapping,
        # but that is more advanced. This simpler example uses a linear scale.
    
        plt.title(f"Bubble Plot of MSE for {model}")
        plt.xlabel("Data Size")
        plt.ylabel("Hidden Size")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 27 FAILED, continuing', flush=True)

print('=== CELL 28 ===', flush=True)
try:
    # Line Plot: MSE Across Hidden Sizes and Data Sizes
    plt.figure(figsize=(14, 8))
    
    # Plot each model's MSE for all data sizes and hidden sizes
    for model in models:
        for i, data_size in enumerate(data_sizes):
            plt.plot(
                hidden_sizes,
                mse_values[model][i],
                label=f"{model} (Data Size={data_size})",
                marker="o",
                linestyle="-",
            )
    
    # Add title and axis labels
    plt.title("MSE Across Hidden Sizes and Data Sizes", fontsize=16)
    plt.xlabel("Hidden Sizes", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(hidden_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models and Data Sizes", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 28 FAILED, continuing', flush=True)

print('=== CELL 29 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Prepare data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    data_sizes = [100, 300, 600, 1200]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # Example MSE values for visualization
    mse_values = {
        "CauchyNet": [
            [0.000685, 0.000146, 0.003650, 0.000496, 0.000342, 0.021072],  # Data Size 100
            [0.000222, 0.000170, 0.000237, 0.000015, 0.001560, 0.001591],  # Data Size 300
            [0.000226, 0.000022, 0.000004, 0.000006, 0.000024, 0.000102],  # Data Size 600
            [0.000017, 0.000018, 0.000014, 0.000010, 0.000024, 0.000110],  # Data Size 1200
        ],
        "CauchyNet1": [
            [0.044191, 0.040490, 0.042729, 0.045892, 0.055015, 0.061345],  # Data Size 100
            [0.038939, 0.037371, 0.040976, 0.045169, 0.050542, 0.058900],  # Data Size 300
            [0.040437, 0.038464, 0.041653, 0.044253, 0.051950, 0.059588],  # Data Size 600
            [0.038273, 0.040054, 0.041914, 0.044812, 0.050814, 0.060149],  # Data Size 1200
        ],
    }
    
    # ====================== Heatmap ======================
    plt.figure(figsize=(12, 8))
    
    # Convert data into a DataFrame for heatmap
    data = []
    for i, data_size in enumerate(data_sizes):
        for j, hidden_size in enumerate(hidden_sizes):
            for model in models:
                mse = mse_values[model][i][j]
                data.append({"Model": model, "Data Size": data_size, "Hidden Size": hidden_size, "MSE": mse})
    
    df = pd.DataFrame(data)
    
    # Plot heatmaps for both models
    for model in models:
        plt.figure(figsize=(10, 6))
        pivot_data = df[df["Model"] == model].pivot("Hidden Size", "Data Size", "MSE")
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".6f",
            cmap="viridis",
            cbar_kws={"label": "MSE"},
        )
        plt.title(f"Heatmap of MSE for {model}", fontsize=16)
        plt.xlabel("Data Size", fontsize=14)
        plt.ylabel("Hidden Size", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # ====================== Line Plot ======================
    plt.figure(figsize=(14, 8))
    
    # Plot each model's MSE for all data sizes and hidden sizes
    for model in models:
        for i, data_size in enumerate(data_sizes):
            plt.plot(
                hidden_sizes,
                mse_values[model][i],
                label=f"{model} (Data Size={data_size})",
                marker="o",
                linestyle="-",
            )
    
    # Add title and axis labels
    plt.title("MSE Across Hidden Sizes and Data Sizes", fontsize=16)
    plt.xlabel("Hidden Sizes", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(hidden_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models and Data Sizes", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 29 FAILED, continuing', flush=True)

print('=== CELL 30 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.colors import PowerNorm
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    data_sizes = [100, 300, 600, 1200]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # Example MSE values
    mse_values = {
        "CauchyNet": [
            [0.000685, 0.000146, 0.003650, 0.000496, 0.000342, 0.021072],  # Data Size 100
            [0.000222, 0.000170, 0.000237, 0.000015, 0.001560, 0.001591],  # Data Size 300
            [0.000226, 0.000022, 0.000004, 0.000006, 0.000024, 0.000102],  # Data Size 600
            [0.000017, 0.000018, 0.000014, 0.000010, 0.000024, 0.000110],  # Data Size 1200
        ],
        "CauchyNet1": [
            [0.044191, 0.040490, 0.042729, 0.045892, 0.055015, 0.061345],  # Data Size 100
            [0.038939, 0.037371, 0.040976, 0.045169, 0.050542, 0.058900],  # Data Size 300
            [0.040437, 0.038464, 0.041653, 0.044253, 0.051950, 0.059588],  # Data Size 600
            [0.038273, 0.040054, 0.041914, 0.044812, 0.050814, 0.060149],  # Data Size 1200
        ],
    }
    
    # Prepare data for heatmap
    data = []
    for i, data_size in enumerate(data_sizes):
        for j, hidden_size in enumerate(hidden_sizes):
            for model in models:
                mse = mse_values[model][i][j]
                data.append({"Model": model, "Data_Size": data_size, "Hidden_Size": hidden_size, "MSE": mse})
    
    df = pd.DataFrame(data)
    
    # Plot heatmaps for both models
    for model in models:
        plt.figure(figsize=(10, 6))
        pivot_data = df[df["Model"] == model].pivot(index="Hidden_Size", columns="Data_Size", values="MSE")
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".6f",
            cmap="viridis",
            norm=PowerNorm(gamma=0.2),  # Adjust the color scale for better differentiation of small values
            cbar_kws={"label": "MSE"},
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(f"Heatmap of MSE for {model} (Enhanced Color Sensitivity)", fontsize=16)
        plt.xlabel("Data Size", fontsize=14)
        plt.ylabel("Hidden Size", fontsize=14)
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 30 FAILED, continuing', flush=True)

print('=== CELL 31 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    data_sizes = [100, 300, 600, 1200]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # Example MSE values for visualization
    mse_values = {
        "CauchyNet": [
            [0.000685, 0.000146, 0.003650, 0.000496, 0.000342, 0.021072],  # Data Size 100
            [0.000222, 0.000170, 0.000237, 0.000015, 0.001560, 0.001591],  # Data Size 300
            [0.000226, 0.000022, 0.000004, 0.000006, 0.000024, 0.000102],  # Data Size 600
            [0.000017, 0.000018, 0.000014, 0.000010, 0.000024, 0.000110],  # Data Size 1200
        ],
        "CauchyNet1": [
            [0.044191, 0.040490, 0.042729, 0.045892, 0.055015, 0.061345],  # Data Size 100
            [0.038939, 0.037371, 0.040976, 0.045169, 0.050542, 0.058900],  # Data Size 300
            [0.040437, 0.038464, 0.041653, 0.044253, 0.051950, 0.059588],  # Data Size 600
            [0.038273, 0.040054, 0.041914, 0.044812, 0.050814, 0.060149],  # Data Size 1200
        ],
    }
    
    # Prepare data for heatmap
    data = []
    for i, data_size in enumerate(data_sizes):
        for j, hidden_size in enumerate(hidden_sizes):
            for model in models:
                mse = mse_values[model][i][j]
                data.append({"Model": model, "Data_Size": data_size, "Hidden_Size": hidden_size, "MSE": mse})
    
    df = pd.DataFrame(data)
    
    # Prepare a custom colormap for slow-changing green shades
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_green", ["#d9f0d3", "#a6d96a", "#1a9850", "#006837"], N=256
    )
    
    # Plot heatmaps for both models
    for model in models:
        plt.figure(figsize=(10, 6))
        pivot_data = df[df["Model"] == model].pivot(index="Hidden_Size", columns="Data_Size", values="MSE")
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".6f",
            cmap=custom_cmap,
            cbar_kws={"label": "MSE"},
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(f"Heatmap of MSE for {model} (Slow-Changing Green Colors)", fontsize=16)
        plt.xlabel("Data Size", fontsize=14)
        plt.ylabel("Hidden Size", fontsize=14)
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 31 FAILED, continuing', flush=True)

print('=== CELL 32 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    data_sizes = [100, 300, 600, 1200]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # Example MSE values for visualization
    mse_values = {
        "CauchyNet": [
            [0.000685, 0.000146, 0.003650, 0.000496, 0.000342, 0.021072],  # Data Size 100
            [0.000222, 0.000170, 0.000237, 0.000015, 0.001560, 0.001591],  # Data Size 300
            [0.000226, 0.000022, 0.000004, 0.000006, 0.000024, 0.000102],  # Data Size 600
            [0.000017, 0.000018, 0.000014, 0.000010, 0.000024, 0.000110],  # Data Size 1200
        ],
        "CauchyNet1": [
            [0.044191, 0.040490, 0.042729, 0.045892, 0.055015, 0.061345],  # Data Size 100
            [0.038939, 0.037371, 0.040976, 0.045169, 0.050542, 0.058900],  # Data Size 300
            [0.040437, 0.038464, 0.041653, 0.044253, 0.051950, 0.059588],  # Data Size 600
            [0.038273, 0.040054, 0.041914, 0.044812, 0.050814, 0.060149],  # Data Size 1200
        ],
    }
    
    # Prepare data for heatmap
    data = []
    for i, data_size in enumerate(data_sizes):
        for j, hidden_size in enumerate(hidden_sizes):
            for model in models:
                mse = mse_values[model][i][j]*100
                data.append({"Model": model, "Data_Size": data_size, "Hidden_Size": hidden_size, "MSE": mse})
    
    df = pd.DataFrame(data)
    
    # Prepare a custom colormap to emphasize small MSE values
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_green", ["#006837", "#1a9850", "#a6d96a", "#d9f0d3"], N=156
    )
    
    # Plot heatmaps for both models
    for model in models:
        plt.figure(figsize=(10, 6))
        pivot_data = df[df["Model"] == model].pivot(index="Hidden_Size", columns="Data_Size", values="MSE")
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".6f",
            cmap=custom_cmap,
            cbar_kws={"label": "MSE"},
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(f"Heatmap of MSE for {model} (Small MSE Highlighted)", fontsize=16)
        plt.xlabel("Data Size", fontsize=14)
        plt.ylabel("Hidden Size", fontsize=14)
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 32 FAILED, continuing', flush=True)

print('=== CELL 33 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.ticker import MaxNLocator
    
    # Recreate the data for MSE values
    models = ["CauchyNet", "CauchyNet1"]
    data_sizes = [100, 300, 600, 1200]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # Example MSE values for visualization
    mse_values = {
        "CauchyNet": [
            [0.000685, 0.000146, 0.003650, 0.000496, 0.000342, 0.021072],  # Data Size 100
            [0.000222, 0.000170, 0.000237, 0.000015, 0.001560, 0.001591],  # Data Size 300
            [0.000226, 0.000022, 0.000004, 0.000006, 0.000024, 0.000102],  # Data Size 600
            [0.000017, 0.000018, 0.000014, 0.000010, 0.000024, 0.000110],  # Data Size 1200
        ],
        "CauchyNet1": [
            [0.044191, 0.040490, 0.042729, 0.045892, 0.055015, 0.061345],  # Data Size 100
            [0.038939, 0.037371, 0.040976, 0.045169, 0.050542, 0.058900],  # Data Size 300
            [0.040437, 0.038464, 0.041653, 0.044253, 0.051950, 0.059588],  # Data Size 600
            [0.038273, 0.040054, 0.041914, 0.044812, 0.050814, 0.060149],  # Data Size 1200
        ],
    }
    
    # Prepare the data for plotting
    data = []
    for i, data_size in enumerate(data_sizes):
        for j, hidden_size in enumerate(hidden_sizes):
            for model in models:
                mse = mse_values[model][i][j]
                data.append({"Model": model, "Data Size": data_size, "Hidden Size": hidden_size, "MSE": mse})
    
    df = pd.DataFrame(data)
    
    # ====================== Combined Line Plot ======================
    plt.figure(figsize=(14, 8))
    colors = {"CauchyNet": "blue", "CauchyNet1": "orange"}
    markers = {"CauchyNet": "o", "CauchyNet1": "s"}
    
    # Plot MSE values across data sizes and hidden sizes
    for model in models:
        for data_size in data_sizes:
            subset = df[(df["Model"] == model) & (df["Data Size"] == data_size)]
            plt.plot(
                subset["Hidden Size"],
                subset["MSE"],
                label=f"{model} (Data Size={data_size})",
                color=colors[model],
                marker=markers[model],
                linestyle="-",
                markersize=8,
                linewidth=2,
            )
    
    # Add axis labels and title
    plt.title("MSE Across Hidden Sizes and Data Sizes", fontsize=20)
    plt.xlabel("Hidden Sizes", fontsize=16)
    plt.ylabel("MSE", fontsize=16)
    plt.xscale("log")  # Logarithmic scale for hidden sizes
    plt.yscale("log")  # Logarithmic scale for MSE
    plt.xticks(hidden_sizes, labels=hidden_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models and Data Sizes", fontsize=12, title_fontsize=14, loc="best")
    
    # Add grid for better readability
    plt.grid(alpha=0.3, linestyle="--")
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # ====================== Cylinder Plot ======================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Cylinder positions
    x_positions = range(len(hidden_sizes))
    width = 0.3  # Bar width
    
    for model_index, model in enumerate(models):
        for data_index, data_size in enumerate(data_sizes):
            x_offset = model_index * width + data_index * width * len(models)
            heights = mse_values[model][data_index]
    
            # Bar plot for each combination
            ax.bar(
                [x + x_offset for x in x_positions],
                heights,
                width=width,
                label=f"{model} (Data Size={data_size})",
                color=colors[model],
                alpha=0.7,
                edgecolor="k",
            )
    
    # Add axis labels and title
    ax.set_title("Cylinder Plot of MSE by Hidden and Data Sizes", fontsize=20)
    ax.set_xlabel("Hidden Sizes", fontsize=16)
    ax.set_ylabel("MSE", fontsize=16)
    ax.set_xticks([x + width * len(models) for x in x_positions])
    ax.set_xticklabels(hidden_sizes, fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune="both"))
    
    # Add legend
    ax.legend(title="Models and Data Sizes", fontsize=12, title_fontsize=14, loc="best")
    
    # Add gridlines for readability
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 33 FAILED, continuing', flush=True)

print('=== CELL 34 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    data_sizes = [100, 300, 600, 1200]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # Example MSE values
    mse_values = {
        "CauchyNet": [
            [0.000685, 0.000146, 0.003650, 0.000496, 0.000342, 0.021072],  # Data Size 100
            [0.000222, 0.000170, 0.000237, 0.000015, 0.001560, 0.001591],  # Data Size 300
            [0.000226, 0.000022, 0.000004, 0.000006, 0.000024, 0.000102],  # Data Size 600
            [0.000017, 0.000018, 0.000014, 0.000010, 0.000024, 0.000110],  # Data Size 1200
        ],
        "CauchyNet1": [
            [0.044191, 0.040490, 0.042729, 0.045892, 0.055015, 0.061345],  # Data Size 100
            [0.038939, 0.037371, 0.040976, 0.045169, 0.050542, 0.058900],  # Data Size 300
            [0.040437, 0.038464, 0.041653, 0.044253, 0.051950, 0.059588],  # Data Size 600
            [0.038273, 0.040054, 0.041914, 0.044812, 0.050814, 0.060149],  # Data Size 1200
        ],
    }
    
    # Improved Cylinder Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Cylinder positions
    x_positions = np.arange(len(hidden_sizes))
    width = 0.18  # Narrower bar width
    
    # Colors for each combination
    colors = sns.color_palette("tab10", n_colors=len(models) * len(data_sizes))
    
    # Plot grouped bars
    for model_index, model in enumerate(models):
        for data_index, data_size in enumerate(data_sizes):
            x_offset = model_index * width + data_index * width * len(models)
            heights = mse_values[model][data_index]
    
            # Bar plot for each combination
            ax.bar(
                [x + x_offset for x in x_positions],
                heights,
                width=width,
                label=f"{model} (Data Size={data_size})",
                color=colors[model_index * len(data_sizes) + data_index],
                alpha=0.8,
                edgecolor="k",
            )
    
    # Add axis labels and title
    ax.set_title("Improved Cylinder Plot of MSE by Hidden and Data Sizes", fontsize=20)
    ax.set_xlabel("Hidden Sizes", fontsize=16)
    ax.set_ylabel("MSE (Log Scale)", fontsize=16)
    ax.set_xticks(x_positions + (len(models) * len(data_sizes) * width) / 2 - width)
    ax.set_xticklabels(hidden_sizes, fontsize=12)
    ax.set_yscale("log")  # Logarithmic scale for better visibility of MSE differences
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune="both"))
    
    # Add legend
    ax.legend(title="Models and Data Sizes", fontsize=10, title_fontsize=12, loc="best", ncol=2)
    
    # Add gridlines for readability
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 34 FAILED, continuing', flush=True)

print('=== CELL 35 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    data_sizes = [100, 300, 600, 1200]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # Example MSE values
    mse_values = {
        "CauchyNet": [
            [0.000685, 0.000146, 0.003650, 0.000496, 0.000342, 0.021072],  # Data Size 100
            [0.000222, 0.000170, 0.000237, 0.000015, 0.001560, 0.001591],  # Data Size 300
            [0.000226, 0.000022, 0.000004, 0.000006, 0.000024, 0.000102],  # Data Size 600
            [0.000017, 0.000018, 0.000014, 0.000010, 0.000024, 0.000110],  # Data Size 1200
        ],
        "CauchyNet1": [
            [0.044191, 0.040490, 0.042729, 0.045892, 0.055015, 0.061345],  # Data Size 100
            [0.038939, 0.037371, 0.040976, 0.045169, 0.050542, 0.058900],  # Data Size 300
            [0.040437, 0.038464, 0.041653, 0.044253, 0.051950, 0.059588],  # Data Size 600
            [0.038273, 0.040054, 0.041914, 0.044812, 0.050814, 0.060149],  # Data Size 1200
        ],
    }
    
    # Plot each model's results separately
    colors = sns.color_palette("tab10", n_colors=len(data_sizes))
    markers = ["o", "s", "d", "^"]
    
    for model in models:
        plt.figure(figsize=(12, 8))
    
        # Plot lines for each data size
        for i, data_size in enumerate(data_sizes):
            mse = mse_values[model][i]
            plt.plot(
                hidden_sizes,
                mse,
                label=f"Data Size={data_size}",
                color=colors[i],
                marker=markers[i],
                linestyle="-",
                linewidth=2,
                markersize=8,
            )
    
        # Add axis labels and title
        plt.title(f"MSE Across Hidden Sizes for {model}", fontsize=20)
        plt.xlabel("Hidden Sizes", fontsize=16)
        plt.ylabel("MSE", fontsize=16)
        plt.xscale("log")  # Logarithmic scale for hidden sizes
        plt.yscale("log")  # Logarithmic scale for MSE values
        plt.xticks(hidden_sizes, labels=hidden_sizes, fontsize=12)
        plt.yticks(fontsize=12)
    
        # Add legend
        plt.legend(title="Data Sizes", fontsize=12, title_fontsize=14, loc="best")
    
        # Add gridlines for readability
        plt.grid(alpha=0.3, linestyle="--")
    
        # Show plot
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 35 FAILED, continuing', flush=True)

print('=== CELL 36 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    data_sizes = [100, 300, 600, 1200]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # Example MSE values for visualization
    mse_values = {
        "CauchyNet": [
            [0.000685, 0.000146, 0.003650, 0.000496, 0.000342, 0.021072],  # Data Size 100
            [0.000222, 0.000170, 0.000237, 0.000015, 0.001560, 0.001591],  # Data Size 300
            [0.000226, 0.000022, 0.000004, 0.000006, 0.000024, 0.000102],  # Data Size 600
            [0.000017, 0.000018, 0.000014, 0.000010, 0.000024, 0.000110],  # Data Size 1200
        ],
        "CauchyNet1": [
            [0.044191, 0.040490, 0.042729, 0.045892, 0.055015, 0.061345],  # Data Size 100
            [0.038939, 0.037371, 0.040976, 0.045169, 0.050542, 0.058900],  # Data Size 300
            [0.040437, 0.038464, 0.041653, 0.044253, 0.051950, 0.059588],  # Data Size 600
            [0.038273, 0.040054, 0.041914, 0.044812, 0.050814, 0.060149],  # Data Size 1200
        ],
    }
    
    # Prepare data for heatmap
    data = []
    for i, data_size in enumerate(data_sizes):
        for j, hidden_size in enumerate(hidden_sizes):
            for model in models:
                mse = mse_values[model][i][j]
                data.append({"Model": model, "Data_Size": data_size, "Hidden_Size": hidden_size, "MSE": mse})
    
    df = pd.DataFrame(data)
    
    # Prepare a custom colormap for lighter colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_viridis", ["orange", "green", "blue", "purple"], N=256)
    
    # Plot heatmaps for both models
    for model in models:
        plt.figure(figsize=(10, 6))
        pivot_data = df[df["Model"] == model].pivot(index="Hidden_Size", columns="Data_Size", values="MSE")
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".6f",
            cmap=custom_cmap,
            cbar_kws={"label": "MSE"},
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(f"Heatmap of MSE for {model} (Lighter Colors)", fontsize=16)
        plt.xlabel("Data Size", fontsize=14)
        plt.ylabel("Hidden Size", fontsize=14)
        plt.tight_layout()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 36 FAILED, continuing', flush=True)

print('=== CELL 37 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from sklearn.preprocessing import MinMaxScaler
    import math
    
    # Import utility functions (ensure they exist in your environment)
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
    
    # =============== Transform Data ===============
    X = torch.linspace(-0.95, 0.95, 2000).to(device)
    Y = challenging_function(X.cpu())#.numpy()   # unscaled
    
    scaler.fit(Y_np)  # re-fit on original Y
    
    # Normalize Y
    scaler = MinMaxScaler()
    Y_np = Y.numpy().reshape(-1, 1)
    Y_norm = scaler.fit_transform(Y_np).reshape(-1)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32)
    
    # =============== Data Splits ===============
    # Load data splits (ensure `loadData` is correctly defined)
    train_loader, val_loader, test_loader, ds_test = loadData(X, Y_t, batchSize=32)
    
    # =============== Model Dictionary ===============
    # Replace these placeholders with your actual model classes
    models_dict = {
    "CauchyNet": CauchyNet1,
       # "CauchyNet1": CauchyNet0,
       # "CauchyNet1": CauchyNet1,
       # "CauchyNet2": CauchyNet_NonHolomorphic,
       # "CauchyNet3": CauchyNet_RealActivation,
      #  "CauchyNet4": CauchyNet_NoImagPenalty,
    }
    
    # Generate color map for models
    model_names = list(models_dict.keys())
    color_map = get_model_colors(model_names, emphasize_model="CauchyNet")
    
    # =============== Train Configuration ===============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size, output_size = 1, 1
    lr = 0.01
    #num_repeats =5
    
    # Define the range of hidden sizes and epochs to test
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    epoch_list   = [400]
    
    # This dictionary will hold results for each model at each (hidden_size, epochs) pair:
    # results_grid[model_name][hidden_size][epochs] = {
    #     'MSE': ..., 'MAE': ..., 'Time': ..., 'Num_Params': ...
    # }
    results_grid = {
        model_name: {h: {} for h in hidden_sizes}
        for model_name in model_names
    }
    
    # If you want to track training curves for each run, keep a separate data structure
    training_logs = {
        model_name: {h: {} for h in hidden_sizes}
        for model_name in model_names
    }
    import random
    from torch.utils.data import Subset
    
    # Define data sizes to test
    data_sizes = [100, 300, 600, 1200]
    
    # Initialize results grid for both hidden sizes and data sizes
    results_grid = {
        model_name: {data_size: {h: {} for h in hidden_sizes} for data_size in data_sizes}
        for model_name in model_names
    }
    
    # =============== Run over all (data_size, hidden_size, epochs) combinations ===============
    for data_size in data_sizes:
        print(f"\n=== Testing with data_size={data_size} ===")
        
        # Randomly sample a subset of the training dataset
        subset_indices = random.sample(range(len(ds_test)), data_size)
        subset_train_loader = DataLoader(Subset(ds_test, subset_indices), batch_size=32, shuffle=True)
        
        for hsize in hidden_sizes:
            for ep in epoch_list:
                print(f"    >> Testing hidden_size={hsize}, epochs={ep}")
                for model_name, constructor in models_dict.items():
                    print(f"       >> Training {model_name}")
    
                    # Train and evaluate the model
                    model, train_losses, val_losses, test_mse, test_mae, preds_scaled, truths_scaled, training_time, num_params = train_and_evaluate_model(
                        constructor,
                        input_size, hsize, output_size,
                        subset_train_loader, val_loader, test_loader,
                        lr=lr, epochs=ep, device=device
                    )
    
                    # Store results
                    results_grid[model_name][data_size][hsize][ep] = {
                        'MSE': test_mse,
                        'MAE': test_mae,
                        'Time': training_time,
                        'Num_Params': num_params
                    }
    
    # =============== Print or Summarize Results ===============
    print("\n===== Summary of Results =====")
    for model_name in model_names:
        print(f"\n--- {model_name} ---")
        for data_size in data_sizes:
            for hsize in hidden_sizes:
                for ep in epoch_list:
                    stats = results_grid[model_name][data_size][hsize][ep]
                    print(
                        f"Data Size={data_size}, Hidden={hsize}, Epochs={ep} | "
                        f"MSE={stats['MSE']:.6f}, MAE={stats['MAE']:.6f}, "
                        f"Time={stats['Time']:.2f}s, Num_Params={stats['Num_Params']}"
                    )
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 37 FAILED, continuing', flush=True)

print('=== CELL 38 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    hidden_sizes = [32, 64, 128, 256, 612]
    
    # MSE values
    mse_values = {
        "CauchyNet": [0.000957, 0.000237, 0.000101, 0.000083, 0.000185],
        "CauchyNet1": [0.017606, 0.015176, 0.015883, 0.018431, 0.022242],
    }
    
    # Line Plot: Loss as a Function of Hidden Dimensions
    plt.figure(figsize=(10, 6))
    
    # Define symbols for each model
    symbols = {"CauchyNet": "o-", "CauchyNet1": "s-"}
    colors = {"CauchyNet": "blue", "CauchyNet1": "orange"}
    
    # Plot each model's loss as a function of hidden dimensions
    for model in models:
        plt.plot(
            hidden_sizes,
            mse_values[model],
            symbols[model],
            label=model,
            color=colors[model],
            markersize=8,
            linewidth=2,
        )
    
    # Add title and axis labels
    plt.title("Loss as a Function of Hidden Dimensions", fontsize=16)
    plt.xlabel("Hidden Dimensions", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(hidden_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 38 FAILED, continuing', flush=True)

print('=== CELL 39 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # MSE and time values
    mse_values = {
        "CauchyNet": [0.000957, 0.000237, 0.000101, 0.000083, 0.000185, 0.000837],
        "CauchyNet1": [0.017606, 0.015176, 0.015883, 0.018431, 0.022242, 0.028740],
    }
    time_values = {
        "CauchyNet": [2.11, 2.18, 2.34, 2.63, 3.90, 4.77],
        "CauchyNet1": [1.82, 1.92, 2.01, 2.34, 2.90, 3.81],
    }
    
    # Enhanced plot
    plt.figure(figsize=(14, 8))
    
    colors = {"CauchyNet": "#1f77b4", "CauchyNet1": "#ff7f0e"}  # Use JMLR-like colors
    
    # Create bubbles for each model
    for model in models:
        if len(hidden_sizes) == len(mse_values[model]) == len(time_values[model]):
            bubble_sizes = [t * 300 for t in time_values[model]]  # Scale bubble size
            plt.scatter(
                hidden_sizes,
                mse_values[model],
                s=bubble_sizes,
                label=model,
                color=colors[model],
                alpha=0.7,
                edgecolor="k"
            )
    
    # Customize the style for JMLR-like appearance
    plt.title("Hidden Dimensions vs. MSE (Bubble Size = Time)", fontsize=20, pad=15)
    plt.xlabel("Hidden Dimensions", fontsize=16, labelpad=10)
    plt.ylabel("MSE", fontsize=16, labelpad=10)
    plt.xticks(hidden_sizes, fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add legend with a polished look
    plt.legend(
        title="Models", fontsize=14, title_fontsize=16,
        loc="upper left", frameon=True, shadow=False, facecolor="white", edgecolor="k"
    )
    
    # Add gridlines with light style
    plt.grid(alpha=0.5, linestyle="--", linewidth=0.7)
    
    # Finalize the layout for publication quality
    plt.tight_layout()
    output_path = "figures/bubble_width.pdf"
    plt.savefig(output_path)
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 39 FAILED, continuing', flush=True)

print('=== CELL 40 ===', flush=True)
try:
    # Bubble Plot: Hidden Dimensions vs. MSE (Bubble Size = Time)
    plt.figure(figsize=(12, 8))
    
    for model in models:
        # Ensure time values match hidden sizes and mse_values
        if len(hidden_sizes) == len(mse_values[model]) == len(time_values[model]):
            bubble_sizes = [t * 300 for t in time_values[model]]  # Scale bubble size by time
            plt.scatter(
                hidden_sizes,
                mse_values[model],
                s=bubble_sizes,
                label=model,
                color=colors[model],
                alpha=0.8,
                edgecolor="k"
            )
        else:
            print(f"Mismatch in data lengths for model {model}: "
                  f"{len(hidden_sizes)} hidden sizes, "
                  f"{len(mse_values[model])} mse values, "
                  f"{len(time_values[model])} time values.")
    
    # Add labels to each bubble
    for model in models:
        for i, hsize in enumerate(hidden_sizes):
            plt.text(
                hsize,
                mse_values[model][i],
                f"{mse_values[model][i]:.6f}",
                fontsize=9,
                ha="center",
                va="center",
                color="black"
            )
    
    # Add title and axis labels
    plt.title("Hidden Dimensions vs. MSE (Bubble Size = Time)", fontsize=16)
    plt.xlabel("Hidden Dimensions", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(hidden_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 40 FAILED, continuing', flush=True)

print('=== CELL 41 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # MSE values
    mse_values = {
        "CauchyNet": [0.000957, 0.000237, 0.000101, 0.000083, 0.000185, 0.000837],
        "CauchyNet1": [0.017606, 0.015176, 0.015883, 0.018431, 0.022242, 0.028740],
    }
    
    # Prepare data for the cylinder plot
    x_positions = np.arange(len(hidden_sizes))  # Position of each hidden size on the x-axis
    width = 0.4  # Width of each cylinder
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot cylinders for each model
    colors = {"CauchyNet": "blue", "CauchyNet1": "orange"}
    for i, model in enumerate(models):
        ax.bar(
            x_positions + (i * width),  # Adjust position for each model
            mse_values[model],
            width=width,
            label=model,
            color=colors[model],
            edgecolor="black"
        )
    
    # Add labels on top of each cylinder
    for i, model in enumerate(models):
        for j, mse in enumerate(mse_values[model]):
            ax.text(
                x_positions[j] + (i * width),
                mse,
                f"{mse:.6f}",
                ha="center",
                va="bottom",
                fontsize=9
            )
    
    # Set axis labels and title
    ax.set_title("Cylinder Plot of MSE by Hidden Dimensions", fontsize=16)
    ax.set_xlabel("Hidden Dimensions", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_xticks(x_positions + width / 2)  # Set x-ticks in the middle of grouped bars
    ax.set_xticklabels(hidden_sizes, fontsize=12)
    
    # Add legend
    ax.legend(title="Models", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3, axis="y")
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 41 FAILED, continuing', flush=True)

print('=== CELL 42 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # MSE values
    mse_values = {
        "CauchyNet": [0.000957, 0.000237, 0.000101, 0.000083, 0.000185, 0.000837],
        "CauchyNet1": [0.017606, 0.015176, 0.015883, 0.018431, 0.022242, 0.028740],
    }
    
    # Cylinder Plot
    x_positions = np.arange(len(hidden_sizes))  # Position of each hidden size on the x-axis
    width = 0.4  # Width of each cylinder
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"CauchyNet": "blue", "CauchyNet1": "orange"}
    
    # Plot cylinders for each model
    for i, model in enumerate(models):
        ax.bar(
            x_positions + (i * width),
            mse_values[model],
            width=width,
            label=model,
            color=colors[model],
            edgecolor="black",
        )
    
    # Add axis labels and title
    ax.set_title("Cylinder Plot of MSE by Hidden Dimensions", fontsize=16)
    ax.set_xlabel("Hidden Dimensions", fontsize=14)
    ax.set_ylabel("MSE", fontsize=14)
    ax.set_xticks(x_positions + width / 2)
    ax.set_xticklabels(hidden_sizes, fontsize=12)
    ax.legend(title="Models", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 42 FAILED, continuing', flush=True)

print('=== CELL 43 ===', flush=True)
try:
    import seaborn as sns
    import pandas as pd
    
    # Prepare data for the violin plot
    violin_data = []
    for model in models:
        for hsize, mse in zip(hidden_sizes, mse_values[model]):
            violin_data.append({"Model": model, "Hidden_Size": hsize, "MSE": mse})
    
    # Convert to DataFrame for Seaborn
    df_violin = pd.DataFrame(violin_data)
    
    # Violin Plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x="Hidden_Size",
        y="MSE",
        hue="Model",
        data=df_violin,
        palette={"CauchyNet": "blue", "CauchyNet1": "orange"},
        split=True,
        scale="width",
    )
    plt.title("Violin Plot of MSE by Hidden Dimensions", fontsize=16)
    plt.xlabel("Hidden Dimensions", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Models", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 43 FAILED, continuing', flush=True)

print('=== CELL 44 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    hidden_sizes = [32, 64, 128, 256, 612, 1224]
    
    # MSE values for histogram
    mse_values = {
        "CauchyNet": [0.000957, 0.000237, 0.000101, 0.000083, 0.000185, 0.000837],
        "CauchyNet1": [0.017606, 0.015176, 0.015883, 0.018431, 0.022242, 0.028740],
    }
    
    # Prepare data for histogram
    mse_data = [mse_values[model] for model in models]
    
    # Set up the histogram
    plt.figure(figsize=(12, 6))
    bins = np.linspace(0, max(max(mse_data[0]), max(mse_data[1])), 15)  # Define bins based on the MSE range
    
    # Plot histograms for each model
    colors = {"CauchyNet": "blue", "CauchyNet1": "orange"}
    for i, model in enumerate(models):
        plt.hist(
            mse_data[i],
            bins=bins,
            alpha=0.6,
            label=model,
            color=colors[model],
            edgecolor="k"
        )
    
    # Add title and axis labels
    plt.title("Histogram of MSE Values Across Hidden Dimensions", fontsize=16)
    plt.xlabel("MSE", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 44 FAILED, continuing', flush=True)

print('=== CELL 45 ===', flush=True)
try:
    # Bubble Plot: Hidden Dimensions vs. MSE (Bubble Size = Time)
    plt.figure(figsize=(12, 8))
    
    for model in models:
        # Ensure time values match hidden sizes and mse_values
        bubble_sizes = [t * 50 for t in time_values[model]]  # Scale bubble size by time
        plt.scatter(
            hidden_sizes,
            mse_values[model],
            s=bubble_sizes,
            label=model,
            color=colors[model],
            alpha=0.8,
            edgecolor="k"
        )
    
    # Add labels to each bubble
    for model in models:
        for i, hsize in enumerate(hidden_sizes):
            plt.text(
                hsize,
                mse_values[model][i],
                f"{mse_values[model][i]:.6f}",
                fontsize=9,
                ha="center",
                va="center",
                color="black"
            )
    
    # Add title and axis labels
    plt.title("Hidden Dimensions vs. MSE (Bubble Size = Time)", fontsize=16)
    plt.xlabel("Hidden Dimensions", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(hidden_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 45 FAILED, continuing', flush=True)

print('=== CELL 46 ===', flush=True)
try:
    # Bubble Plot: Hidden Dimensions vs. MSE (Bubble Size = Time)
    time_values = {
        "CauchyNet": [2.11, 2.18, 2.34, 2.63, 3.90, 4.77],
        "CauchyNet1": [1.82, 1.92, 2.01, 2.34, 2.90, 3.81],
    }
    
    plt.figure(figsize=(12, 8))
    
    for model in models:
        plt.scatter(
            hidden_sizes,
            mse_values[model],
            s=[t * 50 for t in time_values[model]],  # Scale bubble size by time
            label=model,
            color=colors[model],
            alpha=0.8,
            edgecolor="k"
        )
    
    # Add labels to each bubble
    for model in models:
        for i, hsize in enumerate(hidden_sizes):
            plt.text(
                hsize,
                mse_values[model][i],
                f"{mse_values[model][i]:.6f}",
                fontsize=9,
                ha="center",
                va="center",
                color="black"
            )
    
    # Add title and axis labels
    plt.title("Hidden Dimensions vs. MSE (Bubble Size = Time)", fontsize=16)
    plt.xlabel("Hidden Dimensions", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(hidden_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 46 FAILED, continuing', flush=True)

print('=== CELL 47 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    
    # Data for visualization
    models = ["CauchyNet", "CauchyNet1"]
    hidden_sizes = [128, 256, 612, 1224]
    
    # MSE and time values
    mse_values = {
        "CauchyNet": [0.000183, 0.000186, 0.000184, 0.000743],
        "CauchyNet1": [0.021479, 0.022887, 0.027304, 0.034002],
    }
    time_values = {
        "CauchyNet": [1.19, 1.31, 1.71, 2.35],
        "CauchyNet1": [1.01, 1.11, 1.39, 1.81],
    }
    
    # Bubble Plot
    plt.figure(figsize=(12, 8))
    colors = {"CauchyNet": "blue", "CauchyNet1": "orange"}
    
    for model in models:
        plt.scatter(
            hidden_sizes,
            mse_values[model],
            s=[t * 50 for t in time_values[model]],  # Scale bubble size by time
            label=model,
            color=colors[model],
            alpha=0.8,
            edgecolor="k"
        )
    
    # Add labels to each bubble
    for model in models:
        for i, hsize in enumerate(hidden_sizes):
            plt.text(
                hsize,
                mse_values[model][i],
                f"{mse_values[model][i]:.6f}",
                fontsize=9,
                ha="center",
                va="center",
                color="black"
            )
    
    # Title and axis labels
    plt.title("Hidden Dimensions vs. MSE (Bubble Size = Time)", fontsize=16)
    plt.xlabel("Hidden Dimensions", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(hidden_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 47 FAILED, continuing', flush=True)

print('=== CELL 48 ===', flush=True)
try:
    
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    for model_name in model_names:
        # Prepare a 2D matrix for MSE (rows = hidden_sizes, cols = epochs)
        mse_matrix = np.zeros((len(hidden_sizes), len(epoch_list)))
        for i, h in enumerate(hidden_sizes):
            for j, e in enumerate(epoch_list):
                mse_matrix[i, j] = results_grid[model_name][h][e]['MSE']
    
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            mse_matrix,
            annot=True,
            xticklabels=epoch_list,
            yticklabels=hidden_sizes,
            cmap="viridis",
            fmt=".4f"
        )
        plt.title(f"MSE Heatmap for {model_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Hidden Size")
        plt.tight_layout()
        plt.show()
    
    # =============== (Optional) Plot Individual Training Curves ===============
    # You could also plot the training and validation loss curves for a specific combination 
    # of hidden_size and epochs across different models, or vice versa.
    # Example: plot training curves for hidden_size=128 and epochs=200.
    
    target_hsize = 128
    target_ep    = 200
    
    # plot_data_for = {
    #     model_name: {
    #         'train': [training_logs[model_name][target_hsize][target_ep]['train_losses']],
    #         'val':   [training_logs[model_name][target_hsize][target_ep]['val_losses']]
    #     }
    #     for model_name in model_names
    # }
    
    # Access and verify training logs for the target hidden size and epoch
    plot_data_for = {}
    for model_name in model_names:
        if target_hsize in training_logs[model_name]:
            if target_ep in training_logs[model_name][target_hsize]:
                plot_data_for[model_name] = {
                    'train': [training_logs[model_name][target_hsize][target_ep]['train_losses']],
                    'val':   [training_logs[model_name][target_hsize][target_ep]['val_losses']]
                }
            else:
                print(f"Warning: Epoch {target_ep} not found for {model_name} with hidden size {target_hsize}")
        else:
            print(f"Warning: Hidden size {target_hsize} not found for {model_name}")
    
    plot_training_curves_with_confidence(plot_data_for, model_names, color_map,
                                         filename="training_curves_example.pdf")
    
    # =============== Done ===============
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 48 FAILED, continuing', flush=True)

print('=== CELL 49 ===', flush=True)
try:
    # =============== Print or Summarize Results ===============
    print("\n===== Summary of Results =====")
    for model_name in model_names:
        print(f"\n--- {model_name} ---")
        for hsize in hidden_sizes:
            for ep in epoch_list:
                stats = results_grid[model_name][hsize][ep]
                print(
                    f"Hidden={hsize}, Epochs={ep} | "
                    f"MSE={stats['MSE']:.6f}, MAE={stats['MAE']:.6f}, "
                    f"Time={stats['Time']:.2f}s, Num_Params={stats['Num_Params']}"
                )
    
    # =============== (Optional) Example of Plotting Results ===============
    # You might want to visualize how MSE changes with hidden_size and epochs for each model.
    # Below is a simple example of a heatmap. Many other visualizations are possible.
    
    # Convert results into a matrix for each model
    # One dimension: hidden_size, Another dimension: epochs
    # Then we can do a heatmap of MSE, for instance.
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    for model_name in model_names:
        # Prepare a 2D matrix for MSE (rows = hidden_sizes, cols = epochs)
        mse_matrix = np.zeros((len(hidden_sizes), len(epoch_list)))
        for i, h in enumerate(hidden_sizes):
            for j, e in enumerate(epoch_list):
                mse_matrix[i, j] = results_grid[model_name][h][e]['MSE']
    
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            mse_matrix,
            annot=True,
            xticklabels=epoch_list,
            yticklabels=hidden_sizes,
            cmap="viridis",
            fmt=".3f"
        )
        plt.title(f"MSE Heatmap for {model_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Hidden Size")
        plt.tight_layout()
        plt.show()
    
    # =============== (Optional) Plot Individual Training Curves ===============
    # You could also plot the training and validation loss curves for a specific combination 
    # of hidden_size and epochs across different models, or vice versa.
    # Example: plot training curves for hidden_size=128 and epochs=200.
    
    target_hsize = 1224
    target_ep    = 1500
    
    plot_data_for = {
        model_name: {
            'train': [training_logs[model_name][target_hsize][target_ep]['train_losses']],
            'val':   [training_logs[model_name][target_hsize][target_ep]['val_losses']]
        }
        for model_name in model_names
    }
    
    plot_training_curves_with_confidence(plot_data_for, model_names, color_map,
                                         filename="training_curves_example.pdf")
    
    # =============== Done ===============
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 49 FAILED, continuing', flush=True)

print('=== CELL 50 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    # # Prepare data for the visualizations (manually extracted for epoch=1500)
    # models = ["CauchyNet", "CauchyNet0", "CauchyNet1"]
    # hidden_sizes = [64, 128, 256, 612, 1224]
    # fixed_epoch = 1500
    
    # # Manually extracted MSE and time data for epoch=1500
    # mse_values = {
    #     "CauchyNet": [0.002229, 0.000228, 0.000195, 0.000346, 0.003909],
    #     "CauchyNet0": [0.016332, 0.017005, 0.018731, 0.021288, 0.027158],
    #     "CauchyNet1": [0.001003, 0.000248, 0.000134, 0.000211, 0.000353],
    # }
    # time_values = {
    #     "CauchyNet": [7.71, 8.38, 9.43, 13.52, 18.45],
    #     "CauchyNet0": [6.71, 7.28, 8.01, 10.47, 13.76],
    #     "CauchyNet1": [7.74, 8.33, 9.39, 13.29, 18.79],
    # }
    
    # Convert the data into DataFrames
    df_mse = pd.DataFrame(mse_values, index=hidden_sizes)
    df_time = pd.DataFrame(time_values, index=hidden_sizes)
    
    # ====================== Bar Chart: MSE Across Models ======================
    plt.figure(figsize=(10, 6))
    df_mse.plot(kind="bar", figsize=(10, 6), colormap="viridis")
    plt.title(f"MSE Across Models for Fixed Epochs ({fixed_epoch})")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("MSE")
    plt.xticks(rotation=0)
    plt.legend(title="Models")
    plt.tight_layout()
    plt.show()
    
    # ====================== Bubble Plot: Hidden Dimensions vs. MSE with Time ======================
    plt.figure(figsize=(10, 6))
    for model in models:
        plt.scatter(
            hidden_sizes, df_mse[model], s=[t * 50 for t in df_time[model]], label=model, alpha=0.7
        )
    
    plt.title("Hidden Dimensions vs. MSE (Bubble Size = Time)")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("MSE")
    plt.legend(title="Models")
    plt.tight_layout()
    plt.show()
    
    # ====================== Heatmap: Models x Hidden Dimensions ======================
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_mse.T, annot=True, fmt=".4f", cmap="viridis", cbar_kws={"label": "MSE"})
    plt.title(f"Heatmap of MSE (Models x Hidden Dimensions, Epochs={fixed_epoch})")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("Models")
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 50 FAILED, continuing', flush=True)

print('=== CELL 51 ===', flush=True)
try:
    bar_chart_path = "figure/mse_across_models_bar_chart.png"
    df_mse.plot(kind="bar", figsize=(10, 6), colormap="viridis")
    plt.title(f"MSE Across Models for Fixed Epochs ({fixed_epoch})")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("MSE")
    plt.xticks(rotation=0)
    plt.legend(title="Models")
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    plt.close()
    
    # ====================== Bubble Plot: Hidden Dimensions vs. MSE with Time ======================
    bubble_plot_path = "figure/mse_vs_time_bubble_plot.png"
    plt.figure(figsize=(10, 6))
    for model in models:
        plt.scatter(
            hidden_sizes, df_mse[model], s=[t * 50 for t in df_time[model]], label=model, alpha=0.7
        )
    
    plt.title("Hidden Dimensions vs. MSE (Bubble Size = Time)")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("MSE")
    plt.legend(title="Models")
    plt.tight_layout()
    plt.savefig(bubble_plot_path)
    plt.close()
    
    # ====================== Heatmap: Models x Hidden Dimensions ======================
    heatmap_path = "figure/mse_models_hidden_dimensions_heatmap.png"
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_mse.T, annot=True, fmt=".4f", cmap="viridis", cbar_kws={"label": "MSE"})
    plt.title(f"Heatmap of MSE (Models x Hidden Dimensions, Epochs={fixed_epoch})")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("Models")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    
    (bar_chart_path, bubble_plot_path, heatmap_path)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 51 FAILED, continuing', flush=True)

print('=== CELL 52 ===', flush=True)
try:
    plt.figure(figsize=(10, 6))
    df_mse.plot(kind="bar", figsize=(10, 6), colormap="viridis")
    plt.title(f"MSE Across Models for Fixed Epochs ({fixed_epoch})")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("MSE")
    plt.xticks(rotation=0)
    plt.legend(title="Models")
    plt.tight_layout()
    plt.show()
    
    # ====================== Bubble Plot: Hidden Dimensions vs. MSE with Time ======================
    plt.figure(figsize=(10, 6))
    for model in models:
        plt.scatter(
            hidden_sizes, df_mse[model], s=[t * 50 for t in df_time[model]], label=model, alpha=0.7
        )
    
    plt.title("Hidden Dimensions vs. MSE (Bubble Size = Time)")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("MSE")
    plt.legend(title="Models")
    plt.tight_layout()
    plt.show()
    
    # ====================== Heatmap: Models x Hidden Dimensions ======================
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_mse.T, annot=True, fmt=".4f", cmap="viridis", cbar_kws={"label": "MSE"})
    plt.title(f"Heatmap of MSE (Models x Hidden Dimensions, Epochs={fixed_epoch})")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("Models")
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 52 FAILED, continuing', flush=True)

print('=== CELL 53 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    
    # Polished Bubble Plot: Hidden Dimensions vs. MSE with Time as Bubble Size
    plt.figure(figsize=(12, 8))
    
    # Define colors for each model
    colors = {"CauchyNet": "blue", "CauchyNet0": "green", "CauchyNet1": "orange"}
    
    # Iterate over models to plot bubbles
    for model in models:
        plt.scatter(
            hidden_sizes,
            df_mse[model],
            s=[t * 70 for t in df_time[model]],  # Scale bubble size for better visibility
            label=model,
            alpha=0.8,
            color=colors[model],
            edgecolor="k",  # Add black edges to bubbles for clarity
        )
    
    # Add labels to each bubble
    for model in models:
        for i, hsize in enumerate(hidden_sizes):
            plt.text(
                hsize,
                df_mse[model][hsize],
                f"{df_mse[model][hsize]:.4f}",
                fontsize=9,
                ha="center",
                va="center",
                color="black",
            )
    
    # Title and axis labels
    plt.title("Hidden Dimensions vs. MSE (Bubble Size = Time)", fontsize=16)
    plt.xlabel("Hidden Dimensions", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(hidden_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    plt.legend(title="Models", fontsize=12, title_fontsize=14)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    # Tight layout and show the plot
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 53 FAILED, continuing', flush=True)

print('=== CELL 54 ===', flush=True)
try:
    models = ["CauchyNet", "CauchyNet0", "CauchyNet1"]
    hidden_sizes = [64, 128, 256, 612, 1224]
    fixed_epoch = 1500
    
    # Manually extracted MSE data for epoch=1500
    mse_values = {
        "CauchyNet": [0.002229, 0.000228, 0.000195, 0.000346, 0.003909],
        "CauchyNet0": [0.016332, 0.017005, 0.018731, 0.021288, 0.027158],
        "CauchyNet1": [0.001003, 0.000248, 0.000134, 0.000211, 0.000353],
    }
    
    # Convert the data into a matrix (rows=models, columns=hidden_sizes)
    mse_matrix = np.array([mse_values[model] for model in models])
    
    # Transform the data into polar coordinates
    model_radii = np.linspace(0, 1, len(models))  # Normalize models as radial levels
    hidden_angles = np.linspace(0, 2 * np.pi, len(hidden_sizes), endpoint=False)  # Angles for hidden sizes
    
    # Create a grid of points
    r, theta = np.meshgrid(model_radii, hidden_angles)
    points = np.array([(r_, t_) for r_, t_ in zip(r.flatten(), theta.flatten())])
    values = mse_matrix.T.flatten()  # Transpose to match rows=hidden_sizes and cols=models
    
    # Interpolate to create a smooth polar heatmap
    grid_r = np.linspace(0, 1, 200)
    grid_theta = np.linspace(0, 2 * np.pi, 200)
    grid_rr, grid_tt = np.meshgrid(grid_r, grid_theta)
    grid_values = griddata(points, values, (grid_rr, grid_tt), method='cubic', fill_value=0)
    
    # Plot the polar heatmap with updated configuration
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection="polar")
    
    # Plot the MSE values with the custom colormap
    c = ax.pcolormesh(grid_theta, grid_r, grid_values.T, cmap="plasma")
    plt.colorbar(c, ax=ax, pad=0.1, label="MSE")
    
    # Add radial grid lines (circular rings) for models
    for radius in np.linspace(0, 1, len(models)):
        ax.plot(np.linspace(0, 2 * np.pi, 100), [radius] * 100, color="gray", linestyle="--", alpha=0.6)
    
    # Add angular grid lines for hidden sizes
    for angle in hidden_angles:
        ax.plot([angle, angle], [0, 1], color="gray", linestyle="--", alpha=0.6)
    
    # Replace radial labels with model names
    ax.set_yticks(np.linspace(0, 1, len(models)))
    ax.set_yticklabels(models)
    ax.set_ylabel("Models", labelpad=20)
    
    # Replace angular labels with hidden sizes
    ax.set_xticks(hidden_angles)
    ax.set_xticklabels(hidden_sizes)
    ax.set_xlabel("Hidden Dimensions", labelpad=20)
    
    # Add title
    ax.set_title(f"Circular Heatmap of MSE (Models vs. Hidden Dimensions, Epochs={fixed_epoch})", va='bottom')
    
    # Show the updated plot
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 54 FAILED, continuing', flush=True)

print('=== CELL 55 ===', flush=True)
try:
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection="polar")
    
    # Use a colormap with greens and blues for better visual appeal
    c = ax.pcolormesh(grid_theta, grid_r, grid_values.T, cmap="viridis")
    plt.colorbar(c, ax=ax, pad=0.1, label="MSE")
    
    # Add radial grid lines (circular rings) for models
    for radius in np.linspace(0, 1, len(models)):
        ax.plot(np.linspace(0, 2 * np.pi, 100), [radius] * 100, color="gray", linestyle="--", alpha=0.6)
    
    # Add angular grid lines for hidden sizes
    for angle in hidden_angles:
        ax.plot([angle, angle], [0, 1], color="gray", linestyle="--", alpha=0.6)
    
    # Replace radial labels with model names
    ax.set_yticks(np.linspace(0, 1, len(models)))
    ax.set_yticklabels(models)
    ax.set_ylabel("Models", labelpad=20)
    
    # Replace angular labels with hidden sizes
    ax.set_xticks(hidden_angles)
    ax.set_xticklabels(hidden_sizes)
    ax.set_xlabel("Hidden Dimensions", labelpad=20)
    
    # Add MSE values at the center of each sector
    for i, model_name in enumerate(models):
        for j, hsize in enumerate(hidden_sizes):
            radius = np.linspace(0, 1, len(models))[i] + 0.1  # Offset to position within the ring
            angle = hidden_angles[j]
            ax.text(angle, radius, f"{mse_matrix[i, j]:.4f}", color="black", ha="center", va="center", fontsize=8)
    
    # Add title
    ax.set_title(f"Circular Heatmap of MSE (Models vs. Hidden Dimensions, Epochs={fixed_epoch})", va='bottom')
    
    # Show the updated plot
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 55 FAILED, continuing', flush=True)

print('=== CELL 56 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Example data for MSE matrix (replace with your actual data)
    hidden_sizes = [64, 128, 256, 612, 1224]
    epoch_list = [100, 200, 400, 800, 1500]
    
    
    # Heatmap plot
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        mse_matrix,
        annot=True,
        xticklabels=epoch_list,
        yticklabels=hidden_sizes,
        cmap="viridis",
        fmt=".3f"
    )
    plt.title("MSE Heatmap for CauchyNet0")
    plt.xlabel("Epochs")
    plt.ylabel("Hidden Size")
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = "figures/mse_heatmap_cauchynet0.pdf"
    plt.savefig(heatmap_path)
    plt.show()
    
    # Generate a circular heatmap
    from math import pi
    import pandas as pd
    
    # Prepare data for circular heatmap
    data = pd.DataFrame(mse_matrix, index=hidden_sizes, columns=epoch_list)
    data_melted = data.reset_index().melt(id_vars="index")
    data_melted.columns = ["Hidden Size", "Epochs", "MSE"]
    
    # Circular heatmap plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Calculate the angles and radii
    angles = np.linspace(0, 2 * pi, len(epoch_list), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    for i, hsize in enumerate(hidden_sizes):
        values = data.iloc[i].tolist()
        values += values[:1]  # Close the loop for each hidden size
        ax.plot(angles, values, label=f"Hidden Size {hsize}")
    
    ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(epoch_list)
    ax.set_title("Circular Heatmap of MSE")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    
    # Save circular heatmap
    circular_heatmap_path = "figures/circular_mse_heatmap.pdf"
    plt.savefig(circular_heatmap_path)
    plt.show()
    
    (heatmap_path, circular_heatmap_path)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 56 FAILED, continuing', flush=True)

print('=== CELL 57 ===', flush=True)
try:
    # Plot the polar heatmap with a custom colormap that uses a light color at zero
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create a custom colormap with a light base color (but not white) for zero
    colors = ["#f5f5dc", "yellow", "orange", "red", "purple"]  # Light beige to dark
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection="polar")
    
    # Plot the MSE values with the custom colormap
    c = ax.pcolormesh(grid_theta, grid_r, grid_values.T, cmap=custom_cmap)
    plt.colorbar(c, ax=ax, pad=0.1, label="MSE")
    
    # Add radial grid lines (circular rings) for hidden dimensions
    for radius in np.linspace(0, 1, len(hidden_sizes)):
        ax.plot(np.linspace(0, 2 * np.pi, 100), [radius] * 100, color="gray", linestyle="--", alpha=0.6)
    
    # Add angular grid lines for epochs
    for angle in epoch_angles:
        ax.plot([angle, angle], [0, 1], color="gray", linestyle="--", alpha=0.6)
    
    # Replace radial labels with hidden dimensions
    ax.set_yticks(np.linspace(0, 1, len(hidden_sizes)))
    ax.set_yticklabels(hidden_sizes)
    ax.set_ylabel("Hidden Dimensions", labelpad=20)
    
    # Replace angular labels with epochs
    ax.set_xticks(epoch_angles)
    ax.set_xticklabels(epoch_list)
    ax.set_xlabel("Number of Epochs", labelpad=20)
    
    # Add title
    ax.set_title(f"Circular Heatmap of MSE (Hidden Dimensions vs. Epochs)", va='bottom')
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 57 FAILED, continuing', flush=True)

print('=== CELL 58 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    # # Example data structure for results_grid (replace this with your actual results data)
    # hidden_sizes = [64, 128, 256, 612, 1224]
    # epoch_list = [100, 200, 400, 800, 1500]
    # model_name = "CauchyNet0"  # Replace with your desired model name
    
    # # Replace this example with your actual mse_matrix extraction logic
    # # Example results_grid data for illustration purposes
    # results_grid = {
    #     "CauchyNet0": {
    #         h: {e: {'MSE': np.random.uniform(0.01, 0.03)} for e in epoch_list}
    #         for h in hidden_sizes
    #     }
    # }
    
    # # Prepare the MSE matrix
    # mse_matrix = np.zeros((len(hidden_sizes), len(epoch_list)))
    # for i, h in enumerate(hidden_sizes):
    #     for j, e in enumerate(epoch_list):
    #         mse_matrix[i, j] = results_grid[model_name][h][e]['MSE']
    
    # Transform the data into polar coordinates
    hidden_sizes_normalized = np.linspace(0, 1, len(hidden_sizes))  # Normalize hidden sizes (radii)
    epoch_angles = np.linspace(0, 2 * np.pi, len(epoch_list), endpoint=False)  # Angles for epochs
    
    # Create a grid of points
    r, theta = np.meshgrid(hidden_sizes_normalized, epoch_angles)
    points = np.array([(r_, t_) for r_, t_ in zip(r.flatten(), theta.flatten())])
    values = mse_matrix.T.flatten()  # Transpose to match rows=epochs and cols=hidden_sizes
    
    # Interpolate to create a smooth polar heatmap
    grid_r = np.linspace(0, 1, 200)
    grid_theta = np.linspace(0, 2 * np.pi, 200)
    grid_rr, grid_tt = np.meshgrid(grid_r, grid_theta)
    grid_values = griddata(points, values, (grid_rr, grid_tt), method='cubic', fill_value=0)
    
    # Plot the polar heatmap
    plt.figure(figsize=(4, 4))
    ax = plt.subplot(projection="polar")
    c = ax.pcolormesh(grid_theta, grid_r, grid_values.T, cmap="viridis")
    plt.colorbar(c, ax=ax, pad=0.1, label="MSE")
    
    # Add title and labels
    ax.set_title(f"Circular Heatmap of MSE for {model_name}", va='bottom')
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 58 FAILED, continuing', flush=True)

print('=== CELL 59 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    # Transform the data into polar coordinates
    hidden_sizes_normalized = np.linspace(0, 1, len(hidden_sizes))  # Normalize hidden sizes (radii)
    epoch_angles = np.linspace(0, 2 * np.pi, len(epoch_list), endpoint=False)  # Angles for epochs
    
    # Create a grid of points
    r, theta = np.meshgrid(hidden_sizes_normalized, epoch_angles)
    points = np.array([(r_, t_) for r_, t_ in zip(r.flatten(), theta.flatten())])
    values = mse_matrix.T.flatten()  # Transpose to match rows=epochs and cols=hidden_sizes
    
    # Interpolate to create a smooth polar heatmap
    grid_r = np.linspace(0, 1, 200)
    grid_theta = np.linspace(0, 2 * np.pi, 200)
    grid_rr, grid_tt = np.meshgrid(grid_r, grid_theta)
    grid_values = griddata(points, values, (grid_rr, grid_tt), method='cubic', fill_value=0)
    
    # Plot the polar heatmap with updated labels
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection="polar")
    
    # Plot the MSE values
    c = ax.pcolormesh(grid_theta, grid_r, grid_values.T, cmap="viridis")
    plt.colorbar(c, ax=ax, pad=0.1, label="MSE")
    
    # Replace radial labels with hidden dimensions
    ax.set_yticks(np.linspace(0, 1, len(hidden_sizes)))
    ax.set_yticklabels(hidden_sizes)
    ax.set_ylabel("Hidden Dimensions", labelpad=20)
    
    # Replace angular labels with epochs
    ax.set_xticks(epoch_angles)
    ax.set_xticklabels(epoch_list)
    ax.set_xlabel("Number of Epochs", labelpad=20)
    
    # Add title
    ax.set_title(f"Circular Heatmap of MSE (Hidden Dimensions vs. Epochs)", va='bottom')
    
    # Show the updated plot
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 59 FAILED, continuing', flush=True)

print('=== CELL 60 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    # Example data for circular heatmap (replace with actual MSE data)
    hidden_sizes = [64, 128, 256, 612, 1224]
    epoch_list = [100, 200, 400, 800, 1500]
    mse_values = [
        [0.018, 0.018, 0.017, 0.017, 0.016],
        [0.019, 0.017, 0.017, 0.017, 0.017],
        [0.019, 0.019, 0.018, 0.018, 0.019],
        [0.023, 0.021, 0.021, 0.022, 0.021],
        [0.028, 0.027, 0.026, 0.026, 0.026]
    ]
    
    # Transform the data into polar coordinates
    hidden_sizes_normalized = np.linspace(0, 1, len(hidden_sizes))  # Normalize hidden sizes (radii)
    epoch_angles = np.linspace(0, 2 * np.pi, len(epoch_list), endpoint=False)  # Angles for epochs
    
    # Create a grid of points
    r, theta = np.meshgrid(hidden_sizes_normalized, epoch_angles)
    points = np.array([(r_, t_) for r_, t_ in zip(r.flatten(), theta.flatten())])
    values = np.array(mse_values).flatten()
    
    # Interpolate to create a smooth polar heatmap
    grid_r = np.linspace(0, 1, 200)
    grid_theta = np.linspace(0, 2 * np.pi, 200)
    grid_rr, grid_tt = np.meshgrid(grid_r, grid_theta)
    grid_values = griddata(points, values, (grid_rr, grid_tt), method='cubic', fill_value=0)
    
    # Plot the polar heatmap
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection="polar")
    c = ax.pcolormesh(grid_theta, grid_r, grid_values.T, cmap="viridis")
    plt.colorbar(c, ax=ax, pad=0.1, label="MSE")
    
    # Add title and labels
    ax.set_title("Circular Heatmap of MSE (Hidden Size vs. Epochs)", va='bottom')
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 60 FAILED, continuing', flush=True)

print('=== CELL 61 ===', flush=True)
try:
    
    
    # =============== 12) Evaluate Predictions on a Uniform Grid ===============
    # Example: last 100 points for overlay
    # 1) Torch version for model inference
    X_plot_torch      = X.to(device)  # shape: (100,) or (100,1)
    Y_plot_true_torch = Y.to(device)  # unscaled ground truth
    
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
        filename="subplot_predictions2.pdf"
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
    
    plot_box_errors(error_dict, color_map, filename="box_errors2.pdf")
    
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
    plot_error_curves(X_plot_np, Y_plot_true_np, overlay_preds, color_map, filename="error_curves2.pdf")
    
    print("\n=== Done! ===")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 61 FAILED, continuing', flush=True)

print('=== CELL 62 ===', flush=True)
try:
    import torch
    import pickle
    import os
    # === Custom utility imports (as in your snippet) ===
    
    # Directory to store model checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    # Directory to store logs
    os.makedirs("logs", exist_ok=True)
    from utils2 import train_and_evaluate_model,get_model_colors
    
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    input_size=1
    hidden_size=128
    output_siz=1
    
    model_results = {}
    
    # List of models
    
    
    # After training each model in your loop:
    for model_name, model in models.items():
        # Train the model (already done in your code)
        loss_history, val_loss_history, test_data, test_predictions, test_targets = train_model(
            model, 
            train_loader, 
            val_loader, 
            test_loader, 
            model_name,     
            epochs=200,
            lr=0.005,
            scheduler_step_size=2000,
            scheduler_gamma=0.5
            )
    
        
        # Save the logs to model_results
        model_results[model_name] = {
            'loss_history': loss_history, 
            'val_loss_history': val_loss_history, 
            'test_data': test_data, 
            'test_predictions': test_predictions, 
            'test_targets': test_targets
        }
    
        # 1) SAVE MODEL WEIGHTS
        # Example checkpoint path: "checkpoints/CauchyNet.pth"
       # ckpt_path = f"checkpoints/{model_name}_best_missingdata.pth"
       # torch.save(model.state_dict(), ckpt_path)
       # print(f"Saved model weights to {ckpt_path}")
    
    # 2) SAVE TRAINING LOGS (MODEL_RESULTS)
    #with open("logs/training_logs.pkl", "wb") as f:
    #    pickle.dump(model_results, f)
    
    #print("Saved training logs to logs/training_logs.pkl")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 62 FAILED, continuing', flush=True)

print('=== CELL 63 ===', flush=True)
try:
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
    
    # 6) Prepare your model dictionary (classes must be defined/imported)
    models_dict = {
        "CauchyNet":   CauchyNet,
        "SIREN":       SIREN,
        "NBeats":      NBeats,
        "FNN":         FNN_ReLU,
        "RBF":         RBFNetwork,
        "Transformer": MinimalTransformer
    }
    
    # 7) Create a color map for consistent plotting
    model_names = list(models_dict.keys())
    color_map   = get_model_colors(model_names, emphasize_model="CauchyNet")
    
    # 8) Training configuration
    input_size   = 1
    hidden_size  = 128
    output_size  = 1
    lr           = 0.01
    epochs       = 200
    num_repeats  = 10
    
    # 9) Dictionaries to log intervals and results
    interval_logs = {}
    results_dict  = {}
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 63 FAILED, continuing', flush=True)

print('=== CELL 64 ===', flush=True)
try:
    model_results = {}
    
    # 10) Train multiple runs per model & collect logs
    for model_name, constructor in models_dict.items():
        print(f"\n=== Training {model_name} ===")
        runs_mse, runs_mae, runs_time = [], [], []
        train_runs, val_runs         = [], []
        Num_Params = None
    
        for _ in range(num_repeats):
            (model, 
             train_losses, 
             val_losses,
             test_mse, 
             test_mae, 
             preds_unscaled,
             truths_unscaled,
             training_time,
             Num_Params)  =  train_and_evaluate_model(
                model_constructor  = constructor,
                input_size         = input_size,
                hidden_size        = hidden_size,
                output_size        = output_size,
                train_loader       = train_loader,
                val_loader         = val_loader,
                test_loader        = test_loader,
                lr                 = lr,
                epochs             = epochs,
                device             = device,
                scaler             = scaler  # pass the scaler for consistent transforms
            )
    
            runs_mse.append(test_mse)
            runs_mae.append(test_mae)
            runs_time.append(training_time)
            train_runs.append(train_losses)
            val_runs.append(val_losses)
    
            model_results[model_name] = {
            'loss_history': train_losses, 
            'val_loss_history': val_losses, 
            'test_predictions': preds_unscaled, 
            'test_targets': truths_unscaled
        }
    
        # Mean ± Std across the repeats
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
    
    # 11) Plot training curves (with confidence intervals)
    plot_training_curves_with_confidence(
        interval_logs,
        model_names,
        color_map,
        filename="training_curves1.pdf"
    )
    
    # 12) Final run for each model (for final overlays or predictions)
    print("\n=== Final single run for each model to produce final predictions ===")
    final_models = {}
    for model_name, constructor in models_dict.items():
        model, *_ = train_and_evaluate_model(
            model_constructor=constructor,
            input_size       =input_size,
            hidden_size      =hidden_size,
            output_size      =output_size,
            train_loader     =train_loader,
            val_loader       =val_loader,
            test_loader      =test_loader,
            lr               =lr,
            epochs           =epochs,
            device           =device,
            scaler           =scaler  # pass the same scaler
        )
        final_models[model_name] = model
    
        # 12a) Save the trained model weights to disk
        # # e.g. "CauchyNet_trained.pth"
        model_path = f"{model_name}_trained_10.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model weights for {model_name} to {model_path}")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 64 FAILED, continuing', flush=True)

print('=== CELL 65 ===', flush=True)
try:
    import torch
    import numpy as np
    import os
    
    # Suppose you have the following saved model checkpoints
    # in the same directory or a "checkpoints/" folder:
    # "CauchyNet_trained.pth", "SIREN_trained.pth", "NBeats_trained.pth", "FNN_trained.pth", etc.
    # We'll do a dictionary to match checkpoint paths to model classes:
    checkpoint_paths = {
        "CauchyNet_ReLU":   "CauchyNet_ReLU_trained_10.pth",
        "CauchyNet_NoImag":       "CauchyNet_NoImag_trained_10.pth",
        "CauchyNet_RealPar":      "CauchyNet_RealPar_trained_10.pth",
        "CauchyNet_CReLU":         "CauchyNet_CReLU_trained_10.pth",
        "CauchyNet0":         "CauchyNet0_trained_10.pth",
        "CauchyNet1": "CauchyNet1_trained_10.pth"
    }
    
    # 1) Create fresh instances, load their state_dicts
    loaded_models = {}
    for model_name, ckpt_path in checkpoint_paths.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: checkpoint not found => {ckpt_path}")
            continue
    
        # Create a fresh instance with the same architecture
        # (same input_size, hidden_size, output_size, etc.)
        constructor = models_dict[model_name]  # from your original dictionary
        model = constructor(input_size=1, hidden_size=128, output_size=1)  # or however you init
    
        # Load the saved weights
        print(f"Loading {model_name} from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    
        loaded_models[model_name] = model
    
    # 2) Evaluate predictions on a uniform grid for final overlay
    #    We re-fit the same scaler on original Y to ensure consistent inverses
    # X_plot = X_mapped
    # Y_plot_true = Y_mapped   # unscaled
    
    
    
    X_plot = torch.tensor(X_mapped, dtype=torch.float32)
    Y_plot_true = torch.tensor(Y_mapped, dtype=torch.float32)
    scaler.fit(Y_np)  # re-fit on original Y
    
    overlay_preds = {}
    for model_name, model in loaded_models.items():
        model.eval()
        with torch.no_grad():
            out = model(X_plot.unsqueeze(-1).float())
            if isinstance(out, tuple):
                y_real, _ = out
                preds_scaled = y_real.cpu().numpy().flatten()
            else:
                preds_scaled = out.cpu().numpy().flatten()
    
        preds_2d       = preds_scaled.reshape(-1, 1)
        preds_unscaled = scaler.inverse_transform(preds_2d).flatten()
        overlay_preds[model_name] = preds_unscaled
    
    # (Optional) Subplot predictions
    subplot_predictions(
        X_plot.cpu().numpy(),
        Y_plot_true,
        overlay_preds,
        color_map,
        filename="subplot_predictions_from_saved_models.pdf"
    )
    
    # 3) Box plot of errors on test set (unscaled)
    error_dict = {}
    for mname, model in loaded_models.items():
        model.eval()
        abs_errors = []
        with torch.no_grad():
            for x_batch, y_batch_scaled in test_loader:
                x_batch, y_batch_scaled = x_batch.to(device), y_batch_scaled.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_scaled = out.cpu().numpy().flatten()
    
                truths_scaled = y_batch_scaled.cpu().numpy().flatten()
                # Inverse transform both
                preds_unscaled  = scaler.inverse_transform(
                                    preds_scaled.reshape(-1,1)
                                  ).flatten()
                truths_unscaled = scaler.inverse_transform(
                                    truths_scaled.reshape(-1,1)
                                  ).flatten()
    
                abs_errors.extend(np.abs(preds_unscaled - truths_unscaled))
        error_dict[mname] = np.array(abs_errors)
    
    plot_box_errors(
        error_dict,
        color_map,
        filename="box_errors_from_saved_models.pdf"
    )
    
    # 4) (Optional) Print summary if you have 'results_dict' saved
    #    If you also saved results_dict to a file, load it similarly and print it again.
    print("\n=== Summary from loaded models ===")
    for mname in loaded_models.keys():
        stats = results_dict.get(mname, None)  # If 'results_dict' is in memory
        if stats is not None:
            print(f"{mname:12s} => "
                  f"MSE={stats['MSE_mean']:.6f}±{stats['MSE_std']:.4f}, "
                  f"MAE={stats['MAE_mean']:.4f}±{stats['MAE_std']:.4f}, "
                  f"Time={stats['Time_mean']:.2f}±{stats['Time_std']:.2f}s, "
                  f"#Params={stats['Num_Params']}")
        else:
            print(f"{mname:12s} => No stats found in results_dict")
    
    # 5) (Optional) Plot error curves
    plot_error_curves(
        X_plot.cpu().numpy(),
        Y_plot_true,
        overlay_preds,
        color_map,
        filename="error_curves_from_saved_models.pdf"
    )
    
    print("\n=== Done with loaded models! ===")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 65 FAILED, continuing', flush=True)

print('=== CELL 66 ===', flush=True)
try:
    # # 2) Prepare arrays for boxplot
    # #    We'll compute abs_errors_all = list of np arrays [abs_errs_model1, abs_errs_model2, ...]
    # model_names    = []
    # abs_errors_all = []
    
    # for model_name, logs_dict in loaded_results.items():
    #     # Typically, you stored:
    #     #  'test_data'        -> shape [N, ...]
    #     #  'test_predictions' -> shape [N, ]
    #     #  'test_targets'     -> shape [N, ]
    #     preds  = logs_dict["test_predictions"]
    #     truths = logs_dict["test_targets"]
    
    #     # Ensure these are numpy arrays
    #     # (they might already be if you used .cpu().numpy() in your train_model function)
    #     preds_arr  = np.array(preds)
    #     truths_arr = np.array(truths)
    
    #     # Compute absolute error
    #     abs_errs = np.abs(preds_arr - truths_arr)
    
    #     model_names.append(model_name)
    #     abs_errors_all.append(abs_errs)
    
    
    # 3) Box plot of errors on test set (unscaled)
    error_dict = {}
    for mname, model in loaded_models.items():
        model.eval()
        abs_errors = []
        with torch.no_grad():
            for x_batch, y_batch_scaled in test_loader:
                x_batch, y_batch_scaled = x_batch.to(device), y_batch_scaled.to(device)
                out = model(x_batch)
                if isinstance(out, tuple):
                    y_real, _ = out
                    preds_scaled = y_real.cpu().numpy().flatten()
                else:
                    preds_scaled = out.cpu().numpy().flatten()
    
                truths_scaled = y_batch_scaled.cpu().numpy().flatten()
                # Inverse transform both
                preds_unscaled  = scaler.inverse_transform(
                                    preds_scaled.reshape(-1,1)
                                  ).flatten()
                truths_unscaled = scaler.inverse_transform(
                                    truths_scaled.reshape(-1,1)
                                  ).flatten()
    
                abs_errors.extend(np.abs(preds_unscaled - truths_unscaled))
        error_dict[mname] = np.array(abs_errors)
    
    
    # 3) Optional color map 
    color_map = {
        "CauchyNet": "#F2C649",
        "SIREN":     "#5DA5DA",
        "FNN":       "#60BD68",
        "NBeats":    "#FAA43A",
        # Add RBF/Transformer if needed
    }
    
    # 4) Call the boxplot function
    from utils4 import plot_boxplot_with_stats_table  # or wherever your function is located
    
    # 1) Create a list of arrays in the same order as model_names
    abs_errors_all = [error_dict[m] for m in model_names]
    
    # 2) Call the plot function
    plot_boxplot_with_stats_table(
        model_names=model_names,
        abs_errors_all=abs_errors_all,  # now a list, not a dict
        color_map=color_map,
        title="Absolute Error Distribution on Test Set",
        filename="boxplot_imputation_results.png"
    )
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 66 FAILED, continuing', flush=True)

print('=== CELL 67 ===', flush=True)
try:
    import math
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    
    
    
    ##############################################################################
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
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 67 FAILED, continuing', flush=True)

print('=== CELL 68 ===', flush=True)
try:
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
        epochs=10000,
        lr=0.005,
        scheduler_step_size=2000,
        scheduler_gamma=0.5
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
    
        return loss_history, val_loss_history, test_data, test_predictions, test_targets
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 68 FAILED, continuing', flush=True)

print('=== CELL 69 ===', flush=True)
try:
    import math
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    # 1) Sample data
    #train_X, train_Y, val_X, val_Y, test_X, test_Y = sample_data(data_size=1000)
    
    # 2) Build DataLoaders
    train_ds = TensorDataset(train_X, train_Y)
    val_ds   = TensorDataset(val_X,   val_Y)
    test_ds  = TensorDataset(test_X,  test_Y)
    
    batch_size = 64
    train_loader= DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    # 3) Build the model
    net = CauchyNet0(input_size=1, hidden_size=800, output_size=1)
    model_name = "CauchyNet_FixedXi"
    
    # 4) Train
    epochs = 10000
    lr = 1e-3
    train_loss, val_loss, test_data, test_preds, test_targs = train_model(
        net, train_loader, val_loader, test_loader, model_name=model_name, epochs=epochs, lr=lr
    )
    
    # 5) Prepare to plot the function and dense predictions
    X_dense = torch.linspace(-2, 2, 300)
    with torch.no_grad():
        out_dense = net(X_dense.unsqueeze(-1))
        if isinstance(out_dense, tuple):
            y_real_dense, y_imag_dense = out_dense
            preds_dense = y_real_dense.numpy().flatten()
        else:
            preds_dense = out_dense.numpy().flatten()
    
    Y_dense_true = f(X_dense)
    
    # 6) First figure: entire function + train/val/test + dense predictions
    plt.figure(figsize=(9, 5))
    plt.plot(X_dense.numpy(), Y_dense_true.numpy(), 'k-', lw=2, alpha=0.4, label="f(x) original")
    
    plt.scatter(train_X.numpy(), train_Y.numpy(), color='blue',   marker='o', s=50, alpha=0.8, label="Train data")
    plt.scatter(val_X.numpy(),   val_Y.numpy(),   color='green',  marker='s', s=50, alpha=0.8, label="Val data")
    plt.scatter(test_X.numpy(),  test_Y.numpy(),  color='orange', marker='^', s=50, alpha=0.8, label="Test data")
    
    plt.plot(X_dense.numpy(), preds_dense, 'r--', lw=2, label="Cauchy Pred (Dense)")
    plt.title(f"Function + Train/Val/Test + Dense Predictions\n{model_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 7) Second figure: focuses on test data (orange) and discrete test predictions (red)
    
    test_data  = np.array(test_data)
    test_preds = np.array(test_preds)
    idx_sort = np.argsort(test_data) 
    test_data_sorted  = test_data[idx_sort]
    test_preds_sorted = test_preds[idx_sort]
    
    plt.figure(figsize=(9, 5))
    plt.plot(X_dense.numpy(), Y_dense_true.numpy(), 'k-', lw=2, alpha=0.4, label="f(x) original")
    
    # Test data in orange triangles
    plt.scatter(test_X.numpy(), test_Y.numpy(), color='orange', marker='^', s=60, alpha=0.8, label="Test data")
    
    # Test predictions in red dashed line
    plt.plot(test_data_sorted, test_preds_sorted, 'r--', lw=2, label="Test Predictions (Discrete)")
    
    plt.title(f"Test Predictions vs. Test Data\n({model_name})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 8) Third figure: training & validation loss (log-scale)
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss,   label="Val Loss")
    plt.yscale("log")
    plt.title(f"{model_name} Train & Val Loss (Log Scale)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- Create a new figure focusing ONLY on the test domain and test predictions ---
    plt.figure(figsize=(9, 5))
    
    # 1) Plot the continuous “true” function across [-2,2], using your dense mesh:
    plt.plot(X_dense.numpy(), Y_dense_true.numpy(),
             'k-', lw=2, alpha=0.4,
             label="f(x) original")
    
    # 2) Plot the actual test points as orange triangles
    test_X_np = test_X.cpu().numpy().flatten()  # ensure shape [Ntest]
    test_Y_np = test_Y.cpu().numpy().flatten()
    plt.scatter(test_X_np, test_Y_np,
                color='orange', marker='^', s=60, alpha=0.8,
                label="Test data")
    
    # 3) Sort your test_X before drawing a line of predicted values
    #    This avoids weird “zig-zag” lines if test_X is unsorted.
    idx_sort = np.argsort(test_X_np)
    test_X_sorted  = test_X_np[idx_sort]          # shape [Ntest]
    test_preds_np  = test_preds                   # from your “train_model” return
    test_preds_sorted = test_preds_np[idx_sort]   # shape [Ntest]
    
    # 4) Now draw the model’s test predictions in red
    plt.plot(test_X_sorted, test_preds_sorted,
             'r--', lw=2,
             label="Test Predictions (Discrete)")
    
    plt.title(f"Test Predictions vs. Test Data\n({model_name})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 69 FAILED, continuing', flush=True)

print('=== CELL 70 ===', flush=True)
try:
    # 4) Train
    epochs = 10000
    lr = 1e-4
    # train_loss, val_loss, test_data, test_preds, test_targs = train_model(
    #     net, train_loader, val_loader, test_loader, model_name=model_name, epochs=epochs, lr=lr
    # )
    
    loss_history, val_loss_history, test_data, test_predictions, test_targets = train_model(
            net, train_loader, val_loader, test_loader, model_name
        )
    
    
    # 5) Prepare to plot the function and dense predictions
    X_dense = torch.linspace(-2, 2, 300)
    with torch.no_grad():
        out_dense = net(X_dense.unsqueeze(-1))
        if isinstance(out_dense, tuple):
            y_real_dense, y_imag_dense = out_dense
            preds_dense = y_real_dense.numpy().flatten()
        else:
            preds_dense = out_dense.numpy().flatten()
    
    Y_dense_true = f(X_dense)
    
    # 6) First figure: entire function + train/val/test + dense predictions
    plt.figure(figsize=(9, 5))
    plt.plot(X_dense.numpy(), Y_dense_true.numpy(), 'k-', lw=2, alpha=0.4, label="f(x) original")
    
    plt.scatter(train_X.numpy(), train_Y.numpy(), color='blue',   marker='o', s=50, alpha=0.8, label="Train data")
    plt.scatter(val_X.numpy(),   val_Y.numpy(),   color='green',  marker='s', s=50, alpha=0.8, label="Val data")
    plt.scatter(test_X.numpy(),  test_Y.numpy(),  color='orange', marker='^', s=50, alpha=0.8, label="Test data")
    
    plt.plot(X_dense.numpy(), preds_dense, 'r--', lw=2, label="Cauchy Pred (Dense)")
    plt.title(f"Function + Train/Val/Test + Dense Predictions\n{model_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 7) Second figure: focuses on test data (orange) and discrete test predictions (red)
    # Convert test_data, test_preds to numpy arrays first:
    
    # 1) Make sure test_data, test_preds are NumPy arrays:
    import matplotlib.pyplot as plt
    
    # 1) Evaluate model on test_X, same as “dense” predictions:
    test_X_cpu = test_X.cpu()  # shape [Ntest]
    with torch.no_grad():
        out_test = net(test_X_cpu.unsqueeze(-1).float())
        if isinstance(out_test, tuple):
            # If your CauchyNet returns (y_real, y_imag)
            y_real_test, _ = out_test
            preds_test = y_real_test.numpy().flatten()
        else:
            preds_test = out_test.numpy().flatten()
    
    # 2) Convert test_X and test_Y to numpy arrays [Ntest]
    test_X_np = test_X_cpu.numpy().flatten()
    test_Y_np = test_Y.cpu().numpy().flatten()
    
    # 3) Sort by X so lines or connectivity will be monotonic
    idx_sort = np.argsort(test_X_np)
    test_X_sorted     = test_X_np[idx_sort]
    preds_test_sorted = preds_test[idx_sort]
    
    # 4) Then plot them:
    plt.figure(figsize=(9,5))
    plt.plot(X_dense.numpy(), Y_dense_true.numpy(), 'k-', lw=2, alpha=0.4, label="f(x) original")
    
    # Plot test data in, say, red dots:
    plt.scatter(test_X.numpy(), test_Y.numpy(), color='red', marker='o', s=60,
                alpha=0.8, label="Test data")
    
    # Plot sorted predictions in orange dashed line:
    plt.plot(test_data_sorted, test_preds_sorted, 'orange', linestyle='--', lw=2,
             label="Test Predictions")
    
    plt.title(f"Test Predictions vs. Test Data\n({model_name})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
    plt.figure(figsize=(9, 5))
    plt.plot(X_dense.numpy(), Y_dense_true.numpy(), 'k-', lw=2, alpha=0.4, label="f(x) original")
    
    # Test data in orange triangles
    plt.scatter(test_X.numpy(), test_Y.numpy(), color='orange', marker='^', s=60, alpha=0.8, label="Test data")
    
    # Test predictions in red dashed line
    plt.plot(test_data_sorted, test_preds_sorted, 'r--', lw=2, label="Test Predictions (Discrete)")
    
    plt.title(f"Test Predictions vs. Test Data\n({model_name})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 8) Third figure: training & validation loss (log-scale)
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss,   label="Val Loss")
    plt.yscale("log")
    plt.title(f"{model_name} Train & Val Loss (Log Scale)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- Create a new figure focusing ONLY on the test domain and test predictions ---
    plt.figure(figsize=(9, 5))
    
    # 1) Plot the continuous “true” function across [-2,2], using your dense mesh:
    plt.plot(X_dense.numpy(), Y_dense_true.numpy(),
             'k-', lw=2, alpha=0.4,
             label="f(x) original")
    
    # 2) Plot the actual test points as orange triangles
    test_X_np = test_X.cpu().numpy().flatten()  # ensure shape [Ntest]
    test_Y_np = test_Y.cpu().numpy().flatten()
    plt.scatter(test_X_np, test_Y_np,
                color='orange', marker='^', s=60, alpha=0.8,
                label="Test data")
    
    # 3) Sort your test_X before drawing a line of predicted values
    #    This avoids weird “zig-zag” lines if test_X is unsorted.
    
    # 4) Now draw the model’s test predictions in red
    plt.plot(test_X_sorted, test_preds_sorted,
             'r--', lw=2,
             label="Test Predictions (Discrete)")
    
    plt.title(f"Test Predictions vs. Test Data\n({model_name})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 70 FAILED, continuing', flush=True)

print('=== CELL 71 ===', flush=True)
try:
    import math
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def plot_two_subfigures_sharedY(
        X_dense, Y_dense_true,
        train_X, train_Y,
        val_X,   val_Y,
        test_X,  test_Y,
        test_X_sorted, preds_test_sorted,
        model_name="CauchyNet"
    ):
        """
        Creates a figure with two side-by-side subplots (sharing one y-axis):
          (a) Data splits + original function.
          (b) Test data & model predictions (with the same Y range).
        
        Args:
          X_dense, Y_dense_true : 1D arrays/tensors for the continuous domain + f(x).
          train_X, train_Y, val_X, val_Y, test_X, test_Y : data splits (1D).
          test_X_sorted, preds_test_sorted : test X and model predictions, sorted by X.
          model_name (str) : label/title for subplot (b).
        """
    
        # --- 1) Convert everything to NumPy if needed ---
        def to_np(x):
            return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    
        Xd = to_np(X_dense)
        Yd = to_np(Y_dense_true)
    
        trX = to_np(train_X);  trY = to_np(train_Y)
        vlX = to_np(val_X);    vlY = to_np(val_Y)
        teX = to_np(test_X);   teY = to_np(test_Y)
    
        tX_sorted = np.array(test_X_sorted)
        pX_sorted = np.array(preds_test_sorted)
    
        # --- 2) Styling & fonts ---
        sns.set_style("whitegrid")
        sns.set_context("talk")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 12,       # Base font size
            "axes.labelsize": 11,  # Axis labels
            "axes.titlesize": 12,  # Subplot title
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        })
    
        # --- 3) Set up subplots with sharey=True so they have one vertical axis ---
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2,
            figsize=(10, 4.5),
            sharey=True
        )
    
    
        ax1.scatter(
            trX, trY,
            color='blue', marker='*', s=5, alpha=0.8,
            label="Train"
        )
        ax1.scatter(
            vlX, vlY,
            color='green', marker='s', s=5, alpha=0.8,
            label="Val"
        )
        ax1.scatter(
            teX, teY,
            color='red', marker='^', s=10, alpha=0.8,
            label="Test"
        )
    
        #ax1.set_title("(a) Data Splits + Original Function")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
    
        # ----------------------------------------------------
        # (b) Test data & model predictions
    
        # Re-plot the same sets for reference:
        ax2.scatter(
            trX, trY,
            color='blue', marker='o', s=5, alpha=0.6,
            label="Train"
        )
        ax2.scatter(
            vlX, vlY,
            color='green', marker='s', s=5, alpha=0.8,
            label="Val"
        )
        ax2.scatter(
            teX, teY,
            color='red', marker='^', s=10, alpha=0.5,
            label="Test"
        )
        ax2.scatter(
            tX_sorted, pX_sorted,
            color='orange', marker='*', s=15, alpha=0.9,
            label="Test Predictions"
        )
    
        #ax2.set_title(f"(b) Test Predictions: {model_name}")
        ax2.set_xlabel("X")
        ax2.set_ylabel("")  # sharey => only left axis has Y label
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
    
        # --- 4) Tight layout so it looks clean ---
        fig.tight_layout()
        fig.suptitle(f"Data Splits & Test Predictions ({model_name})", fontsize=12)
    
    
        # --- 5) Save or show ---
        fig.savefig("missingloss.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    
    # ------------------------------------------------------------------------
    # Example usage:
    plot_two_subfigures_sharedY(
        X_dense, Y_dense_true,
        train_X, train_Y,
        val_X,   val_Y,
        test_X,  test_Y,
        test_X_sorted, preds_test_sorted,
        model_name="CauchyNet"
    )
    # ------------------------------------------------------------------------
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 71 FAILED, continuing', flush=True)

print('=== CELL 72 ===', flush=True)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    #def plot_test_results(model, test_loader, model_path, data_size, plot_real_function=True):
    def plot_test_results(model, test_loader,data_size, plot_real_function=True):
    
        # Load the best model
    #    model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
    
        test_predictions = []
        test_targets = []
        test_data = []
    
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
                test_data.extend(inputs.cpu().numpy())
    
        # Convert lists to numpy arrays for plotting
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        test_data = np.array(test_data)
    
        # Calculate MSE and MAE
        test_mse = mean_squared_error(test_targets, test_predictions)
        test_mae = mean_absolute_error(test_targets, test_predictions)
        print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
    
        # Plotting
        plt.figure(figsize=(10, 5))
        if plot_real_function:
            X_full = np.linspace(-np.pi, np.pi, data_size)
            Y_full = X_full**2 - X_full**3 + np.sin(X_full)
            plt.plot(X_full, Y_full, label='Real Function', alpha=0.5)
    
        plt.scatter(test_data, test_predictions, color='red', label='Test Predictions', s=3)
        plt.title('Function Graph with Test Predictions')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 72 FAILED, continuing', flush=True)

print('=== CELL 73 ===', flush=True)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split, TensorDataset
    
    # Define the FNN model
    class FNN(nn.Module):
        def __init__(self):
            super(FNN, self).__init__()
            self.fc1 = nn.Linear(1, 400)  # Adjust the size as needed
            #self.fc2 = nn.Linear(400, 400)  # Hidden layer
            self.fc3 = nn.Linear(400, 1)   # Output layer
    
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
        """
        SIREN: Implicit Neural Representation with Periodic (sine) Activation.
        
        This version is adapted to match the typical (x -> y) shape:
          - If x is shape [batch_size], we unsqueeze to [batch_size, 1].
          - Output shape => [batch_size, output_size].
        
        Example usage for 1D inputs/outputs:
            model = SIREN(input_size=1, hidden_size=64, output_size=1)
            out   = model(x)  # x in shape [B,1] or [B]
        """
        def __init__(self, input_size, hidden_size, output_size, w0_initial=30.0, w0_hidden=30.0):
            """
            Args:
                input_size  (int): Dimensionality of input (e.g. 1 for scalar x).
                hidden_size (int): Number of hidden neurons in the single hidden layer.
                output_size (int): Dimensionality of output (e.g. 1 for scalar y).
                w0_initial (float): Frequency scale for the first (input) layer.
                w0_hidden  (float): Frequency scale for subsequent layers 
                                    (if you had more layers).
            """
            super().__init__()
            self.input_size  = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            
            self.linear_in   = nn.Linear(input_size,  hidden_size)
            self.linear_out  = nn.Linear(hidden_size, output_size)
            
            self.w0_initial  = w0_initial
            self.w0_hidden   = w0_hidden
            
            # Initialize weights (SIREN style)
            self._init_weights()
    
        def _init_weights(self):
            """
            Custom initialization following the SIREN paper:
             - First layer scaled by w0_initial
             - Output layer uses typical uniform init but scaled by w0_hidden (or so).
            """
            with torch.no_grad():
                # 1) For linear_in: uniform in [-1/fan_in, 1/fan_in], then scale by w0_initial
                fan_in = self.linear_in.in_features  # typically input_size
                bound  = 1.0 / fan_in
                nn.init.uniform_(self.linear_in.weight, -bound, bound)
                nn.init.zeros_(self.linear_in.bias)
                self.linear_in.weight *= self.w0_initial
    
                # 2) For linear_out: standard uniform init, scaled by w0_hidden
                #    A common approach: uniform in [-sqrt(6/fan_in), sqrt(6/fan_in)]
                fan_in_out = self.linear_out.in_features
                out_bound  = (6.0 / fan_in_out) ** 0.5
                nn.init.uniform_(self.linear_out.weight, -out_bound / self.w0_hidden, out_bound / self.w0_hidden)
                nn.init.zeros_(self.linear_out.bias)
    
        def sine(self, x: torch.Tensor, w0: float) -> torch.Tensor:
            """
            Sine activation with frequency scaling: sin(w0 * x).
            """
            return torch.sin(w0 * x)
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass:
               1) Possibly reshape input to [B, input_size].
               2) First layer => sine(...) with frequency=1 for the hidden layer
                  (the weight matrix was already scaled by w0_initial).
               3) Linear out => shape [B, output_size].
            """
            # If x is [batch_size], reshape to [batch_size, input_size].
            if x.dim() == 1:
                x = x.unsqueeze(-1)  # shape => [B,1] if input_size=1
    
            # For the first linear: we previously scaled the matrix by w0_initial
            # so we pass w0=1.0 inside the sine. 
            h = self.sine(self.linear_in(x), w0=1.0)
            out = self.linear_out(h)  # shape => [B, output_size]
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
    
    class MinimalTransformer(nn.Module):
        """
        Optimized minimal Transformer for scalar input-output mappings.
        """
        def __init__(self, input_size, hidden_size, output_size, nhead=2, num_layers=1, dropout=0.1):
            super().__init__()
            self.embedding = nn.Linear(input_size, hidden_size)
            self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_size))  # Positional encoding
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 2, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc_out = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            # Ensure x has the shape (batch_size, input_dim)
            if x.dim() == 1:
                x = x.unsqueeze(-1)
    
            # Embedding and positional encoding
            x = self.embedding(x)                          # Shape: (batch_size, hidden_size)
            x = x.unsqueeze(1) + self.positional_encoding  # Shape: (batch_size, seq_len=1, hidden_size)
            x = self.dropout(x)                            # Apply dropout
    
            # Transformer encoding
            x = self.encoder(x)                            # Shape: (batch_size, seq_len=1, hidden_size)
            x = x.squeeze(1)                               # Remove sequence dimension: (batch_size, hidden_size)
    
            # Fully connected output layer
            out = self.fc_out(x)                           # Shape: (batch_size, output_size)
            return out
    
    class MinimalInformer(nn.Module):
        """
        Optimized Informer-like model for scalar input-output mappings.
        """
        def __init__(self, input_size, hidden_size, output_size, nhead=2, num_layers=1, dropout=0.1):
            super().__init__()
            self.embedding = nn.Linear(input_size, hidden_size)
            self.gate = nn.Linear(hidden_size, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 2, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc_out = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            # Ensure x has the shape (batch_size, input_dim)
            if x.dim() == 1:
                x = x.unsqueeze(-1)
    
            # Embedding and gating
            emb = self.embedding(x)                        # Shape: (batch_size, hidden_size)
            gate = torch.sigmoid(self.gate(emb)) * emb     # Apply gating
            gate = gate.unsqueeze(1)                       # Shape: (batch_size, seq_len=1, hidden_size)
            gate = self.dropout(gate)                      # Apply dropout
    
            # Transformer encoding
            enc = self.encoder(gate)                       # Shape: (batch_size, seq_len=1, hidden_size)
            enc = enc.squeeze(1)                           # Remove sequence dimension: (batch_size, hidden_size)
    
            # Fully connected output layer
            out = self.fc_out(enc)                         # Shape: (batch_size, output_size)
            return out
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 73 FAILED, continuing', flush=True)

print('=== CELL 74 ===', flush=True)
try:
    import torch
    import pickle
    import os
    
    # Directory to store model checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    # Directory to store logs
    os.makedirs("logs", exist_ok=True)
    
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    input_size=1
    hidden_size=1000
    output_siz=1
    cauchynet_model = CauchyNet(input_size=1, hidden_size=1000, output_size=1)
    fnn_model = FNN()  # Assuming FNN is a class that defines your model
    NBeats_model = NBeats(input_size=1, hidden_size=1000, output_size=1)  # 
    SIREN_model = SIREN(input_size=1, hidden_size=1000, output_size=1)
    Transformer_model =MinimalInformer(input_size=1, hidden_size=1000, output_size=1)
    
    model_results = {}
    
    # List of models
    models = {'CauchyNet': cauchynet_model,
              'SIREN': SIREN_model, 
              'NBeats': NBeats_model, 
              'FNN': fnn_model, 
              'Transformer':Transformer_model
             }
    
    # After training each model in your loop:
    for model_name, model in models.items():
        # Train the model (already done in your code)
        loss_history, val_loss_history, test_data, test_predictions, test_targets = train_model(
            model, train_loader, val_loader, test_loader, model_name,     epochs=10,
        lr=0.005,
        scheduler_step_size=2000,
        scheduler_gamma=0.5
        )
    
        
        # Save the logs to model_results
        model_results[model_name] = {
            'loss_history': loss_history, 
            'val_loss_history': val_loss_history, 
            'test_data': test_data, 
            'test_predictions': test_predictions, 
            'test_targets': test_targets
        }
    
        # 1) SAVE MODEL WEIGHTS
        # Example checkpoint path: "checkpoints/CauchyNet.pth"
       # ckpt_path = f"checkpoints/{model_name}_best_missingdata.pth"
       # torch.save(model.state_dict(), ckpt_path)
       # print(f"Saved model weights to {ckpt_path}")
    
    # 2) SAVE TRAINING LOGS (MODEL_RESULTS)
    #with open("logs/training_logs.pkl", "wb") as f:
    #    pickle.dump(model_results, f)
    
    #print("Saved training logs to logs/training_logs.pkl")
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 74 FAILED, continuing', flush=True)

print('=== CELL 75 ===', flush=True)
try:
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    input_size=1
    hidden_size=1000
    output_siz=1
    cauchynet_model = CauchyNet(input_size=1, hidden_size=1000, output_size=1)
    fnn_model = FNN()  # Assuming FNN is a class that defines your model
    NBeats_model = NBeats(input_size=1, hidden_size=1000, output_size=1)  # 
    SIREN_model = SIREN(input_size=1, hidden_size=1000, output_size=1)
    
    model_results = {}
    
    # List of models
    models = {'CauchyNet': cauchynet_model,'SIREN': SIREN_model, 'NBeats': NBeats_model, 'FNN': fnn_model}
    
    for model_name, model in models.items():
        loss_history, val_loss_history, test_data, test_predictions, test_targets = train_model(
            model, train_loader, val_loader, test_loader, model_name
        )
        model_results[model_name] = {
            'loss_history': loss_history, 
            'val_loss_history': val_loss_history, 
            'test_data': test_data, 
            'test_predictions': test_predictions, 
            'test_targets': test_targets
        }
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 75 FAILED, continuing', flush=True)

print('=== CELL 76 ===', flush=True)
try:
    model_results = {}
    
    # List of models
    models = {'CauchyNet': cauchynet_model,'SIREN': SIREN_model, 'NBeats': NBeats_model, 'FNN': fnn_model}
    
    for model_name, model in models.items():
        loss_history, val_loss_history, test_data, test_predictions, test_targets = train_model(
            model, train_loader, val_loader, test_loader, model_name
        )
        model_results[model_name] = {
            'loss_history': loss_history, 
            'val_loss_history': val_loss_history, 
            'test_data': test_data, 
            'test_predictions': test_predictions, 
            'test_targets': test_targets
        }
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 76 FAILED, continuing', flush=True)

print('=== CELL 77 ===', flush=True)
try:
    # Create a fresh instance of the same model architecture
    cauchynet_model_loaded = CauchyNet(input_size=1, hidden_size=1000, output_size=1)
    
    # Load the state dictionary from the file
    state_dict = torch.load("checkpoints/CauchyNet_best_missingdata.pth", map_location=torch.device('cpu'))
    cauchynet_model_loaded.load_state_dict(state_dict)
    
    cauchynet_model_loaded.eval()  # Set to eval mode if you just want to do inference
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 77 FAILED, continuing', flush=True)

print('=== CELL 78 ===', flush=True)
try:
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    
    
    # 1) Convert your PyTorch tensors to NumPy (if not already NumPy):
    train_x_np = train_X.cpu().numpy() if hasattr(train_X, 'cpu') else train_X
    train_y_np = train_Y.cpu().numpy() if hasattr(train_Y, 'cpu') else train_Y
    val_x_np   = val_X.cpu().numpy()   if hasattr(val_X, 'cpu')   else val_X
    val_y_np   = val_Y.cpu().numpy()   if hasattr(val_Y, 'cpu')   else val_Y
    test_x_np  = test_X.cpu().numpy()  if hasattr(test_X, 'cpu')  else test_X
    test_y_np  = test_Y.cpu().numpy()  if hasattr(test_Y, 'cpu')  else test_Y
    
    
    ##############################################################################
    # A) Plot the training, validation, and test sets
    ##############################################################################
    
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    
    # Training points (blue)
    plt.scatter(train_x_np, train_y_np,
                color='blue', edgecolor='k', s=40, alpha=0.8,
                label='Train Points')
    
    # Validation points (green)
    plt.scatter(val_x_np, val_y_np,
                color='green', edgecolor='k', s=40, alpha=0.8,
                label='Val Points')
    
    # Test points (red)
    plt.scatter(test_x_np, test_y_np,
                color='red', edgecolor='k', s=40, alpha=0.8,
                label='Test Points')
    
    plt.title("Train, Validation, and Test Sets", fontsize=13)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
    ##############################################################################
    # B) Plot the “dense” function and the predicted points for each model
    ##############################################################################
    
    X_dense =torch.linspace(-2, 2, 400)
    
    Y_dense_true = f(X_dense)
    X_dense_np = X_dense.cpu().numpy() if hasattr(X_dense, 'cpu') else X_dense
    Y_dense_np = Y_dense_true.cpu().numpy() if hasattr(Y_dense_true, 'cpu') else Y_dense_true
    
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    
    # Plot the original function as a gray line
    plt.plot(X_dense_np, Y_dense_np,
             color='gray', lw=2, alpha=0.7,
             label="f(x) original")
    
    
    colors = ['orange', 'royalblue', 'limegreen', 'purple', 'red']
    for i, (model_name, results) in enumerate(model_results.items()):
        # Retrieve test data & test predictions from your dictionary
        test_data_this_model  = np.array(results['test_data'])
        test_preds_this_model = np.array(results['test_predictions'])
    
        # (Optional) Sort them by X to avoid zigzags
        idx_sort = np.argsort(test_data_this_model)
        test_data_sorted  = test_data_this_model[idx_sort]
        test_preds_sorted = test_preds_this_model[idx_sort]
    
        plt.scatter(test_data_sorted, test_preds_sorted,
                 color=colors[i % len(colors)],  linestyle='--',  edgecolor='k', s=40, alpha=0.8,
                 label=f"{model_name} Predictions")
    
    # (Optional) highlight training & test sets again
    # plt.scatter(train_x_np, train_y_np, color='blue', marker='o', s=40, alpha=0.7, label='Train Points')
    # plt.scatter(test_x_np, test_y_np, color='red',  marker='^', s=40, alpha=0.7, label='Test Points')
    
    plt.title("Model Predictions vs. Original Function", fontsize=13)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 78 FAILED, continuing', flush=True)

print('=== CELL 79 ===', flush=True)
try:
    #plt.figure(figsize=(8,5))
    
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
    plt.plot(Xd_np, Yd_np, color='gray', lw=2.5, label="f(x) original")
    
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
        c = color_cycle[i % len(color_cycle)]
        
        # Scatter plot: use marker style, remove line style
        plt.scatter(x_sorted, y_sorted,
                    color=c, 
                    s=8,             # marker size
                    alpha=0.7,        # transparency
                    #edgecolor='black',# black edge for good contrast
                    label=f"{model_name} predictions")
    
    plt.title("Comparison of Test Predictions (Scatter)", fontsize=12)
    plt.xlabel("X")
    plt.ylabel("Predicted Y")
    plt.legend()
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 79 FAILED, continuing', flush=True)

print('=== CELL 80 ===', flush=True)
try:
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
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 80 FAILED, continuing', flush=True)

print('=== CELL 81 ===', flush=True)
try:
    plot_data_sets(train_X, train_Y, val_X, val_Y, test_X, test_Y)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 81 FAILED, continuing', flush=True)

print('=== CELL 82 ===', flush=True)
try:
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
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 82 FAILED, continuing', flush=True)

print('=== CELL 83 ===', flush=True)
try:
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
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 83 FAILED, continuing', flush=True)

print('=== CELL 84 ===', flush=True)
try:
    import pickle
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    
    # 1) Load your saved logs from "training_logs.pkl"
    with open("logs/training_logs.pkl", "rb") as f:
        loaded_results = pickle.load(f)
    
    # 2) Prepare arrays for boxplot
    #    We'll compute abs_errors_all = list of np arrays [abs_errs_model1, abs_errs_model2, ...]
    model_names    = []
    abs_errors_all = []
    
    for model_name, logs_dict in loaded_results.items():
        # Typically, you stored:
        #  'test_data'        -> shape [N, ...]
        #  'test_predictions' -> shape [N, ]
        #  'test_targets'     -> shape [N, ]
        preds  = logs_dict["test_predictions"]
        truths = logs_dict["test_targets"]
    
        # Ensure these are numpy arrays
        # (they might already be if you used .cpu().numpy() in your train_model function)
        preds_arr  = np.array(preds)
        truths_arr = np.array(truths)
    
        # Compute absolute error
        abs_errs = np.abs(preds_arr - truths_arr)
    
        model_names.append(model_name)
        abs_errors_all.append(abs_errs)
    
    # 3) Optional color map 
    color_map = {
        "CauchyNet": "#F2C649",
        "SIREN":     "#5DA5DA",
        "FNN":       "#60BD68",
        "NBeats":    "#FAA43A",
        # Add RBF/Transformer if needed
    }
    
    # 4) Call the boxplot function
    from utils4 import plot_boxplot_with_stats_table  # or wherever your function is located
    
    plot_boxplot_with_stats_table(
        model_names=model_names,
        abs_errors_all=abs_errors_all,
        color_map=color_map,
        title="Absolute Error Distribution on Test Set",
        filename="boxplot_imputation_results.png"
    )
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 84 FAILED, continuing', flush=True)
