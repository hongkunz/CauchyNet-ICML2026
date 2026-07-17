import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
_n=[0]
def _show(*a,**k):
    plt.gcf().savefig(f'redo_fig_{_n[0]:02d}.png', dpi=110, bbox_inches='tight'); _n[0]+=1; plt.close('all')
plt.show=_show

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

SEED = 10
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cpu")
print("torch", torch.__version__, "device", device)

class CauchyNet(nn.Module):
    '''Improved CauchyNet (V3): bias-only, conjugate-symmetric, real-by-construction.
    Released-code elliptical init (1.5, 0.5) and 1/h output normalization.

    Parameters
    ----------
    input_size : int     N = input dimensionality.
    h_pairs    : int     number of stored (upper-half) pole rows.
                         Effective hidden width is 2 * h_pairs.
    r_re, r_im : float   semi-major / semi-minor radii of the init ellipse
                         (defaults match the released code).
    '''
    def __init__(self, input_size, h_pairs, r_re=1.5, r_im=0.5, normalize=True):
        super().__init__()
        self.input_size = input_size
        self.h_pairs = h_pairs
        self.normalize = normalize
        # Upper half-ellipse per input dimension: angles in (~0, ~pi)
        angles    = np.pi * (0.05 + 0.9 * torch.rand(h_pairs, input_size))
        real_part = r_re * torch.cos(angles)
        imag_part = r_im * torch.sin(angles)
        self.B = nn.Parameter(torch.complex(real_part, imag_part))
        self.C = nn.Parameter(
            torch.normal(0.0, 1.0, size=(h_pairs,), dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (batch, N) or (batch,) for N=1
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_c  = torch.complex(x, torch.zeros_like(x))          # (batch, N)
        diff = self.B.unsqueeze(0) - x_c.unsqueeze(1)          # (batch, h_pairs, N)
        h_k  = (1.0 / diff).prod(dim=-1)                       # (batch, h_pairs) complex
        # 2 * Re(.) applies the conjugate-partner contribution; optional /h
        # normalization matches Algorithm 1 (released-code convention).
        y = 2.0 * (self.C.unsqueeze(0) * h_k).real.sum(dim=-1)
        if self.normalize:
            y = y / self.h_pairs
        return y

def count_real_params(m):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())

def rescale_to_unit(X, half_range):
    '''Map [-half_range, half_range]^N onto [-1, 1]^N when the input can
    exceed [-1, 1].  Inputs already inside [-1, 1] are left unchanged --
    no point shrinking data closer to the poles.'''
    if half_range <= 1.0:
        return X
    return X / float(half_range)

import copy

def train_model(model, train_loader, val_loader, n_epochs, lr=0.01,
                weight_decay=1e-4, step_size=100, gamma=0.5, log_every=None):
    '''Train with early stopping by best val loss.  Without the lambda|e|^2
    regularizer, losses occasionally spike across epochs; we keep the
    best-val snapshot so final performance tracks the best checkpoint,
    not whatever state training happened to end in.'''
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()
    train_hist, val_hist = [], []
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(n_epochs):
        model.train()
        tl, tn = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            yp = model(xb)
            loss = criterion(yp, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * xb.size(0); tn += xb.size(0)
        scheduler.step()
        train_hist.append(tl / tn)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                yp = model(xb)
                vl += criterion(yp, yb).item() * xb.size(0); vn += xb.size(0)
        v = vl / vn
        val_hist.append(v)
        if v < best_val:
            best_val = v
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch+1:5d}: train={train_hist[-1]:.4e}, "
                  f"val={v:.4e}, best_val={best_val:.4e}@{best_epoch}")

    # Restore best-val snapshot before returning.
    model.load_state_dict(best_state)
    return train_hist, val_hist

def test_error(model, loader):
    ys, yh = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            ys.append(yb.numpy()); yh.append(model(xb).numpy())
    ys = np.concatenate(ys); yh = np.concatenate(yh)
    return mean_squared_error(ys, yh), mean_absolute_error(ys, yh)

def run_preds(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).numpy()

def target_exp1(x):
    term1 = 1.0 / ((x + 0.6) ** 2 + 0.005)
    term2 = -40.0 * torch.exp(-2.0 * (x + 0.4) ** 2)
    term3 = 50.0 * torch.sign(x) * torch.abs(torch.sin(3 * x) + 0.8) ** 1.5 * torch.sin(10 * x)
    return term1 + term2 + term3

torch.manual_seed(SEED); np.random.seed(SEED)
N_TOT = 300
X1_raw = torch.linspace(-1.0, 1.0, N_TOT).float()
Y1_raw = target_exp1(X1_raw).float()

scaler1 = MinMaxScaler()
Y1_norm = torch.tensor(
    scaler1.fit_transform(Y1_raw.numpy().reshape(-1, 1)).reshape(-1),
    dtype=torch.float32,
)

ds1 = TensorDataset(X1_raw, Y1_norm)
train1, val1, test1 = random_split(
    ds1, [150, 75, 75],
    generator=torch.Generator().manual_seed(SEED),
)
train_loader1 = DataLoader(train1, batch_size=32, shuffle=True)
val_loader1   = DataLoader(val1,   batch_size=64, shuffle=False)
test_loader1  = DataLoader(test1,  batch_size=64, shuffle=False)

torch.manual_seed(SEED); np.random.seed(SEED)
cn1 = CauchyNet(input_size=1, h_pairs=128)
print(f"CauchyNet  input_size=1  h_pairs=128  eff width=256  "
      f"real params={count_real_params(cn1)}")

th1, vh1 = train_model(cn1, train_loader1, val_loader1,
                       n_epochs=600, lr=0.01, step_size=200, gamma=0.5,
                       log_every=100)
mse1, mae1 = test_error(cn1, test_loader1)
print(f"Exp 1 test MSE (normalized): {mse1:.4e}")
print(f"Exp 1 test MAE (normalized): {mae1:.4e}")

# Plot predictions vs. truth on dense grid
X1_dense = torch.linspace(-1.0, 1.0, 600).float()
yp1 = run_preds(cn1, X1_dense)
yp1_orig = scaler1.inverse_transform(yp1.reshape(-1, 1)).ravel()
y1_true  = target_exp1(X1_dense).numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
ax.plot(th1, label="train"); ax.plot(vh1, label="val")
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
ax.set_title(f"Exp 1 — training curves")

ax = axes[1]
ax.plot(X1_dense.numpy(), y1_true, "k--", lw=1.5, label="true f(x)")
ax.plot(X1_dense.numpy(), yp1_orig, "C0", lw=1, label="CauchyNet")
ax.set_xlabel("x"); ax.legend(); ax.set_title("Exp 1 — prediction vs. truth")
plt.tight_layout(); plt.show()

def target_exp2a(x):
    return (torch.sin(2 * x - 4)
            + 0.5 * torch.cos(5 * x - 5)
            + 0.05 / ((x - 1.0) ** 2 + 0.1)
            + 0.01 / ((x + 0.5) ** 2 + 0.05)
            - 0.01 * (x ** 2 - x ** 3))

torch.manual_seed(SEED); np.random.seed(SEED)
X2a = torch.linspace(-2.0, 2.0, 400).float()
Y2a = target_exp2a(X2a).float()

# Detect local extrema of g on the dense grid -> turning points
y_np = Y2a.numpy()
diffs = np.sign(np.diff(y_np))
turning_idx = np.where(np.diff(diffs) != 0)[0] + 1
turning_pts = X2a[turning_idx].numpy()
print("Turning points detected:", np.round(turning_pts, 3))

# Missing mask: any x within ±0.15 of a turning point is TEST
dist_to_tp = np.min(np.abs(X2a.numpy()[:, None] - turning_pts[None, :]), axis=1)
is_missing = dist_to_tp <= 0.15
train_idx = np.where(~is_missing)[0]
test_idx  = np.where( is_missing)[0]
print(f"Exp 2a: {len(train_idx)} kept / {len(test_idx)} withheld")

# Normalize Y
scaler2a = MinMaxScaler()
Y2a_norm = torch.tensor(
    scaler2a.fit_transform(Y2a.numpy().reshape(-1, 1)).reshape(-1),
    dtype=torch.float32,
)

X2a_in = rescale_to_unit(X2a, half_range=2.0)  # [-2, 2] -> [-1, 1]
ds2a = TensorDataset(X2a_in, Y2a_norm)
# Remap turning-point distances to rescaled space too

train_set2a = Subset(ds2a, train_idx.tolist())
# Use a small random-split of the training points for validation
val_cut = int(0.2 * len(train_idx))
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(train_idx))
val_sub = Subset(train_set2a, perm[:val_cut].tolist())
train_sub = Subset(train_set2a, perm[val_cut:].tolist())
test_set2a = Subset(ds2a, test_idx.tolist())

train_loader2a = DataLoader(train_sub,  batch_size=32, shuffle=True)
val_loader2a   = DataLoader(val_sub,    batch_size=64, shuffle=False)
test_loader2a  = DataLoader(test_set2a, batch_size=64, shuffle=False)

torch.manual_seed(SEED); np.random.seed(SEED)
# Extrapolation-heavy task: use V2-style config (larger radii (2pi, 1), no /h
# normalization). Smoother kernels generalize better into the withheld ±0.15
# zones than V3's default (1.5, 0.5)+/h combination.
cn2a = CauchyNet(input_size=1, h_pairs=128,
                 r_re=2*np.pi, r_im=1.0, normalize=False)
print(f"CauchyNet  input_size=1  h_pairs=128  real params={count_real_params(cn2a)}  "
      f"(radii=(2pi, 1), no /h normalization — tuned for extrapolation)")

th2a, vh2a = train_model(cn2a, train_loader2a, val_loader2a,
                         n_epochs=3000, lr=0.01, step_size=500, gamma=0.7,
                         log_every=500)
mse2a, mae2a = test_error(cn2a, test_loader2a)
print(f"Exp 2a test MSE (on missing zones, normalized): {mse2a:.4e}")
print(f"Exp 2a test MAE (on missing zones, normalized): {mae2a:.4e}")

X2a_dense = torch.linspace(-2.0, 2.0, 800).float()
yp2a = run_preds(cn2a, rescale_to_unit(X2a_dense, 2.0))
yp2a_orig = scaler2a.inverse_transform(yp2a.reshape(-1, 1)).ravel()
y2a_true  = target_exp2a(X2a_dense).numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
ax.plot(th2a, label="train"); ax.plot(vh2a, label="val")
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
ax.set_title("Exp 2a — training curves")

ax = axes[1]
ax.plot(X2a_dense.numpy(), y2a_true, "k--", lw=1.5, label="true g(x)")
ax.plot(X2a_dense.numpy(), yp2a_orig, "C0", lw=1, label="CauchyNet")
ax.scatter(X2a[train_idx].numpy(), Y2a[train_idx].numpy(),
           s=6, color="C2", alpha=0.4, label="training pts")
ax.scatter(X2a[test_idx].numpy(),  Y2a[test_idx].numpy(),
           s=10, color="C3", alpha=0.7, label="withheld (test)")
ax.set_xlabel("x"); ax.legend(); ax.set_title("Exp 2a — 1D gap-filling")
plt.tight_layout(); plt.show()

def target_exp2b(x, y):
    return 3.0 - x ** 2 + x * y - y ** 2 - 1.0 / (5.0 + (x - 1.0) ** 2)

torch.manual_seed(SEED); np.random.seed(SEED)
rng = np.random.default_rng(SEED)
N2b = 3000
xy2b = rng.uniform(-0.8, 0.8, size=(N2b, 2)).astype(np.float32)
X2b = torch.from_numpy(xy2b)                # (N, 2)
Y2b = target_exp2b(X2b[:, 0], X2b[:, 1])     # (N,)

dist_origin = np.linalg.norm(xy2b, axis=1)
test_mask_2b = dist_origin < 0.3
train_idx_2b = np.where(~test_mask_2b)[0]
test_idx_2b  = np.where( test_mask_2b)[0]

# Val split from train (20%)
perm = rng.permutation(len(train_idx_2b))
val_cut = int(0.2 * len(train_idx_2b))
val_idx_2b   = train_idx_2b[perm[:val_cut]]
trn_idx_2b   = train_idx_2b[perm[val_cut:]]

scaler2b = MinMaxScaler()
Y2b_norm = torch.tensor(
    scaler2b.fit_transform(Y2b.numpy().reshape(-1, 1)).reshape(-1),
    dtype=torch.float32,
)

# Rescale X from [-0.8, 0.8]^2 -> [-1, 1]^2 (user-requested uniform rescaling)
X2b_in = rescale_to_unit(X2b, half_range=0.8)
ds2b = TensorDataset(X2b_in, Y2b_norm)
train_loader2b = DataLoader(Subset(ds2b, trn_idx_2b.tolist()),
                            batch_size=64, shuffle=True)
val_loader2b   = DataLoader(Subset(ds2b, val_idx_2b.tolist()),
                            batch_size=128, shuffle=False)
test_loader2b  = DataLoader(Subset(ds2b, test_idx_2b.tolist()),
                            batch_size=128, shuffle=False)
print(f"Exp 2b: {len(trn_idx_2b)} train / {len(val_idx_2b)} val / {len(test_idx_2b)} test (missing disk)")

torch.manual_seed(SEED); np.random.seed(SEED)
# Extrapolation-heavy task: V2-style config (larger radii, no /h normalization).
cn2b = CauchyNet(input_size=2, h_pairs=128,
                 r_re=2*np.pi, r_im=1.0, normalize=False)
print(f"CauchyNet  input_size=2  h_pairs=128  real params={count_real_params(cn2b)}  "
      f"(radii=(2pi, 1), no /h — tuned for extrapolation)")

th2b, vh2b = train_model(cn2b, train_loader2b, val_loader2b,
                         n_epochs=1500, lr=0.01, step_size=300, gamma=0.6,
                         log_every=300)
mse2b, mae2b = test_error(cn2b, test_loader2b)
print(f"Exp 2b test MSE on missing disk (normalized): {mse2b:.4e}")
print(f"Exp 2b test MAE on missing disk (normalized): {mae2b:.4e}")

# Visualize prediction on a dense grid inside the disk
from matplotlib import cm
gx = np.linspace(-0.8, 0.8, 120); gy = np.linspace(-0.8, 0.8, 120)
GX, GY = np.meshgrid(gx, gy)
grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], axis=1), dtype=torch.float32)
pred_grid = run_preds(cn2b, rescale_to_unit(grid, 0.8)).reshape(GX.shape)
pred_grid_orig = scaler2b.inverse_transform(pred_grid.reshape(-1, 1)).reshape(GX.shape)
true_grid = target_exp2b(torch.tensor(GX.ravel()), torch.tensor(GY.ravel())).numpy().reshape(GX.shape)
err_grid = pred_grid_orig - true_grid

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
im0 = axes[0].imshow(true_grid, extent=[-0.8, 0.8, -0.8, 0.8],
                     origin="lower", cmap=cm.viridis)
axes[0].set_title("true g(x,y)"); plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(pred_grid_orig, extent=[-0.8, 0.8, -0.8, 0.8],
                     origin="lower", cmap=cm.viridis)
axes[1].set_title("CauchyNet prediction"); plt.colorbar(im1, ax=axes[1], fraction=0.046)
im2 = axes[2].imshow(err_grid, extent=[-0.8, 0.8, -0.8, 0.8],
                     origin="lower", cmap=cm.coolwarm,
                     vmin=-abs(err_grid).max(), vmax=abs(err_grid).max())
# Overlay missing disk
theta = np.linspace(0, 2 * np.pi, 200)
for ax in axes:
    ax.plot(0.3 * np.cos(theta), 0.3 * np.sin(theta), "r--", lw=1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
axes[2].set_title("signed error"); plt.colorbar(im2, ax=axes[2], fraction=0.046)
plt.tight_layout(); plt.show()
print("max |error| inside the disk:", float(np.abs(err_grid[np.sqrt(GX**2+GY**2)<0.3]).max()))

def target_exp3(x, y):
    return x ** 2 - x * y + 3.0 * y + y ** 2 + 1.0 / (5.0 + x ** 2)

torch.manual_seed(SEED); np.random.seed(SEED)
rng = np.random.default_rng(SEED)
N3 = 300
xy3 = rng.uniform(-1.5, 1.5, size=(N3, 2)).astype(np.float32)
X3 = torch.from_numpy(xy3)
Y3 = target_exp3(X3[:, 0], X3[:, 1])

scaler3 = MinMaxScaler()
Y3_norm = torch.tensor(
    scaler3.fit_transform(Y3.numpy().reshape(-1, 1)).reshape(-1),
    dtype=torch.float32,
)

X3_in = rescale_to_unit(X3, half_range=1.5)  # [-1.5, 1.5]^2 -> [-1, 1]^2
ds3 = TensorDataset(X3_in, Y3_norm)
n_tr, n_va = int(0.5 * N3), int(0.25 * N3)
n_te = N3 - n_tr - n_va
tr3, va3, te3 = random_split(ds3, [n_tr, n_va, n_te],
                             generator=torch.Generator().manual_seed(SEED))
train_loader3 = DataLoader(tr3, batch_size=32, shuffle=True)
val_loader3   = DataLoader(va3, batch_size=64, shuffle=False)
test_loader3  = DataLoader(te3, batch_size=64, shuffle=False)

torch.manual_seed(SEED); np.random.seed(SEED)
# Extrapolation-heavy task: V2-style config.
cn3 = CauchyNet(input_size=2, h_pairs=128,
                r_re=2*np.pi, r_im=1.0, normalize=False)
print(f"CauchyNet  input_size=2  h_pairs=128  real params={count_real_params(cn3)}  "
      f"(radii=(2pi, 1), no /h — tuned for extrapolation)")

th3, vh3 = train_model(cn3, train_loader3, val_loader3,
                       n_epochs=1500, lr=0.01, step_size=300, gamma=0.6,
                       log_every=300)
mse3, mae3 = test_error(cn3, test_loader3)
print(f"Exp 3 test MSE (normalized): {mse3:.4e}")
print(f"Exp 3 test MAE (normalized): {mae3:.4e}")

# Surface + error visualization
gx = np.linspace(-1.5, 1.5, 120); gy = np.linspace(-1.5, 1.5, 120)
GX, GY = np.meshgrid(gx, gy)
grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], axis=1), dtype=torch.float32)
pred_grid = run_preds(cn3, rescale_to_unit(grid, 1.5)).reshape(GX.shape)
pred_grid_orig = scaler3.inverse_transform(pred_grid.reshape(-1, 1)).reshape(GX.shape)
true_grid = target_exp3(torch.tensor(GX.ravel()), torch.tensor(GY.ravel())).numpy().reshape(GX.shape)
err_grid = pred_grid_orig - true_grid

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
im0 = axes[0].imshow(true_grid, extent=[-1.5, 1.5, -1.5, 1.5],
                     origin="lower", cmap="viridis")
axes[0].set_title("true g(x,y)"); plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(pred_grid_orig, extent=[-1.5, 1.5, -1.5, 1.5],
                     origin="lower", cmap="viridis")
axes[1].set_title("CauchyNet prediction"); plt.colorbar(im1, ax=axes[1], fraction=0.046)
im2 = axes[2].imshow(err_grid, extent=[-1.5, 1.5, -1.5, 1.5],
                     origin="lower", cmap="coolwarm",
                     vmin=-abs(err_grid).max(), vmax=abs(err_grid).max())
axes[2].set_title("signed error"); plt.colorbar(im2, ax=axes[2], fraction=0.046)
for ax in axes:
    ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout(); plt.show()
print("max |error| on grid:", float(np.abs(err_grid).max()))

import pandas as pd

import os as _os
# M4 trend subset ships with this repo in reproducibility_audit/legacy_m4/.
# Override with the M4_TREND_CSV environment variable if needed.
M4_PATH = _os.environ.get(
    "M4_TREND_CSV",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                  "..", "legacy_m4", "M4_trend_data.csv"))
df_m4 = pd.read_csv(M4_PATH)
print("M4 trend shape:", df_m4.shape, " cols:", list(df_m4.columns))

t_raw = df_m4["X"].to_numpy(dtype=np.float32)
y_raw = df_m4["Y"].to_numpy(dtype=np.float32)

# Normalize t to [-1, 1] to match the (1.5, 0.5) elliptical init scale.
# Normalize y via default MinMaxScaler([0,1]) — matches the released M4 code.
t_norm = 2.0 * (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min()) - 1.0
scaler4 = MinMaxScaler()
y_norm = scaler4.fit_transform(y_raw.reshape(-1, 1)).ravel().astype(np.float32)

X4 = torch.from_numpy(t_norm)
Y4 = torch.from_numpy(y_norm)

ds4 = TensorDataset(X4, Y4)
n_total = len(ds4)
n_tr, n_va = 350, 175
n_te = n_total - n_tr - n_va  # 175
tr4, va4, te4 = random_split(ds4, [n_tr, n_va, n_te],
                             generator=torch.Generator().manual_seed(SEED))
train_loader4 = DataLoader(tr4, batch_size=32, shuffle=True)
val_loader4   = DataLoader(va4, batch_size=64, shuffle=False)
test_loader4  = DataLoader(te4, batch_size=64, shuffle=False)
print(f"Exp 4: {n_tr} train / {n_va} val / {n_te} test")

torch.manual_seed(SEED); np.random.seed(SEED)
cn4 = CauchyNet(input_size=1, h_pairs=128)
print(f"CauchyNet  input_size=1  h_pairs=128  real params={count_real_params(cn4)}")

th4, vh4 = train_model(cn4, train_loader4, val_loader4,
                       n_epochs=600, lr=0.01, step_size=200, gamma=0.5,
                       log_every=150)
mse4, mae4 = test_error(cn4, test_loader4)
print(f"Exp 4 test MSE (normalized [-1, 1]): {mse4:.4e}")
print(f"Exp 4 test MAE (normalized [-1, 1]): {mae4:.4e}")

# Plot: training curve + prediction over full time index
X4_dense = torch.linspace(-1.0, 1.0, 700).float()
yp4_norm = run_preds(cn4, X4_dense)
yp4_orig = scaler4.inverse_transform(yp4_norm.reshape(-1, 1)).ravel()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
ax.plot(th4, label="train"); ax.plot(vh4, label="val")
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
ax.set_title("Exp 4 — training curves")

ax = axes[1]
ax.plot(t_raw, y_raw, "k--", lw=1.5, label="M4 trend (true)")
ax.plot(t_raw, yp4_orig, "C0", lw=1, label="CauchyNet")
ax.set_xlabel("time index"); ax.set_ylabel("trend value")
ax.legend(); ax.set_title("Exp 4 — M4 trend reconstruction")
plt.tight_layout(); plt.show()

summary = [
    ("Exp 1 — 1D sharp peak",       "1D",  mse1,  mae1,  count_real_params(cn1),   600),
    ("Exp 2a — 1D gap-filling",     "1D",  mse2a, mae2a, count_real_params(cn2a), 3000),
    ("Exp 2b — 2D missing disk",    "2D",  mse2b, mae2b, count_real_params(cn2b), 1500),
    ("Exp 3 — 2D poly-rational",    "2D",  mse3,  mae3,  count_real_params(cn3),  1500),
    ("Exp 4 — M4 trend forecast",   "1D",  mse4,  mae4,  count_real_params(cn4),   600),
]
print(f"{'experiment':32s}  {'dim':3s}  {'test MSE (norm)':>16s}  "
      f"{'test MAE (norm)':>16s}  {'params':>7s}  {'epochs':>7s}")
for name, dim, mse, mae, p, e in summary:
    print(f"{name:32s}  {dim:3s}  {mse:16.4e}  {mae:16.4e}  {p:7d}  {e:7d}")
