"""
Independent reproduction of supp Fig. 25 (left panel of the parameter-
sensitivity figure): heatmap of test MSE for a trainable-pole CauchyNet
across hidden dimensions {32, 64, 128, 256, 612, 1224} and dataset sizes
{100, 300, 600, 1200} on the paper's 1D mixed target.

Paper claim: larger models reach MSE as low as 1e-6; MSE decreases
monotonically-ish with hidden size (shipped values span 1.3e-3 .. 1.5e-6).

Protocol (legacy trainable-pole setting): MinMaxScaler on targets,
Adam lr=0.01, weight decay 1e-5, 200 epochs, batch 32, imag penalty 0.1.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HIDDENS = [32, 64, 128, 256, 612, 1224]
NDATA = [100, 300, 600, 1200]
EPOCHS, LR, WD, IMAG = 200, 1e-2, 1e-5, 0.1


def target(x):
    return (np.sin(2 * x) + 0.5 * np.sin(5 * x) + 0.3 * np.cos(3 * x)
            - 0.01 * x ** 2 + 0.01 * x ** 3)


class CauchyNet1D(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lam = nn.Parameter(torch.normal(0., 0.1, size=(h, 1), dtype=torch.cfloat))
        ang = 2 * math.pi * torch.rand(h)
        self.xi = nn.Parameter(torch.complex(2 * math.pi * torch.cos(ang), torch.sin(ang)))

    def forward(self, x):
        xc = torch.complex(x, torch.zeros_like(x))
        act = 1.0 / (self.xi.unsqueeze(0) - xc + 1e-8)
        out = act @ self.lam
        return out.real, out.imag


def one_cell(h, n, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    x = np.linspace(-2, 2, n).astype(np.float32)
    y = target(x).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min())
    idx = np.random.permutation(n)
    ntr, nva = int(0.6 * n), int(0.15 * n)
    xtr = torch.tensor(x[idx[:ntr]]).unsqueeze(1); ytr = torch.tensor(y[idx[:ntr]]).unsqueeze(1)
    xva = torch.tensor(x[idx[ntr:ntr+nva]]).unsqueeze(1); yva = torch.tensor(y[idx[ntr:ntr+nva]]).unsqueeze(1)
    xte = torch.tensor(x[idx[ntr+nva:]]).unsqueeze(1); yte = torch.tensor(y[idx[ntr+nva:]]).unsqueeze(1)

    m = CauchyNet1D(h)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    crit = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    best_val, best_state = float("inf"), None
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in dl:
            opt.zero_grad()
            yr, yi = m(xb)
            loss = crit(yr, yb) + IMAG * crit(yi, torch.zeros_like(yi))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(crit(m(xva)[0], yva))
        if v < best_val:
            best_val = v
            best_state = {k: t.clone() for k, t in m.state_dict().items()}
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        yr, _ = m(xte)
        return float(crit(yr, yte))


def main():
    grid = np.zeros((len(HIDDENS), len(NDATA)))
    for i, h in enumerate(HIDDENS):
        for j, n in enumerate(NDATA):
            grid[i, j] = one_cell(h, n)
            print(f"h={h:<5} n={n:<5} test MSE = {grid[i,j]:.2e}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="magma_r",
                   norm=mcolors.PowerNorm(gamma=0.2, vmin=grid.min(), vmax=grid.max()),
                   aspect="auto")
    for i in range(len(HIDDENS)):
        for j in range(len(NDATA)):
            ax.text(j, i, f"{grid[i,j]:.2e}", ha="center", va="center",
                    color="white" if grid[i,j] < np.median(grid) else "black", fontsize=10)
    ax.set_xticks(range(len(NDATA)), NDATA); ax.set_yticks(range(len(HIDDENS)), HIDDENS)
    ax.set_xlabel("Data Size"); ax.set_ylabel("Hidden Size")
    ax.set_title("Heatmap of test MSE for CauchyNet (independent rerun)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("repro_fig25.png", dpi=130, bbox_inches="tight")
    print("saved repro_fig25.png")


if __name__ == "__main__":
    main()
