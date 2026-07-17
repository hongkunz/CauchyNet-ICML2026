"""
Independent reproduction of supp Fig. 24: imaginary-penalty sensitivity.

Trainable-pole CauchyNet variants Cauchy(lambda), lambda in {0.1, 0.3, 0.5, 1, 1.5},
trained 200 epochs on the paper 1D mixed target; box plot of test-set
absolute errors pooled over 5 runs per variant.

Paper claim: a positive penalty helps this trainable-pole ablation,
'especially lambda = 1'.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDAS = [0.1, 0.3, 0.5, 1.0, 1.5]
RUNS, EPOCHS, H, LR, WD = 5, 200, 128, 1e-2, 1e-5


def target(x):
    return (np.sin(2 * x) + 0.5 * np.sin(5 * x) + 0.3 * np.cos(3 * x)
            - 0.01 * x ** 2 + 0.01 * x ** 3)


class CauchyNet1D(nn.Module):
    def __init__(self, h=H):
        super().__init__()
        self.lam = nn.Parameter(torch.normal(0., 0.1, size=(h, 1), dtype=torch.cfloat))
        ang = 2 * math.pi * torch.rand(h)
        self.xi = nn.Parameter(torch.complex(2 * math.pi * torch.cos(ang), torch.sin(ang)))

    def forward(self, x):
        xc = torch.complex(x, torch.zeros_like(x))
        act = 1.0 / (self.xi.unsqueeze(0) - xc + 1e-8)
        out = act @ self.lam
        return out.real, out.imag


def run_variant(lmbda, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = 300
    x = np.linspace(-2, 2, n).astype(np.float32)
    y = target(x).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min())
    idx = np.random.permutation(n)
    ntr, nva = int(0.6 * n), int(0.15 * n)
    xt = torch.tensor(x[idx[:ntr]]).unsqueeze(1);       yt = torch.tensor(y[idx[:ntr]]).unsqueeze(1)
    xv = torch.tensor(x[idx[ntr:ntr+nva]]).unsqueeze(1); yv = torch.tensor(y[idx[ntr:ntr+nva]]).unsqueeze(1)
    xe = torch.tensor(x[idx[ntr+nva:]]).unsqueeze(1);   ye = torch.tensor(y[idx[ntr+nva:]]).unsqueeze(1)

    m = CauchyNet1D()
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    crit = nn.MSELoss()
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt, yt),
                                     batch_size=32, shuffle=True)
    best_v, best_state = float("inf"), None
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in dl:
            opt.zero_grad()
            yr, yi = m(xb)
            loss = crit(yr, yb) + lmbda * crit(yi, torch.zeros_like(yi))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(crit(m(xv)[0], yv))
        if v < best_v:
            best_v, best_state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return np.abs((m(xe)[0] - ye).numpy().flatten())


def main():
    errs = {}
    for l in LAMBDAS:
        pooled = np.concatenate([run_variant(l, s) for s in range(RUNS)])
        errs[l] = pooled
        print(f"Cauchy({l}): MAE={pooled.mean():.6f}  MSE={np.mean(pooled**2):.2e}  "
              f"median={np.median(pooled):.6f}", flush=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot([errs[l] for l in LAMBDAS],
               tick_labels=[f"Cauchy({l:g})" for l in LAMBDAS], showfliers=True)
    ax.set_ylabel("Absolute Error")
    ax.set_title("Distribution of Absolute Errors (Test Set) — independent rerun")
    fig.tight_layout()
    fig.savefig("repro_fig24.png", dpi=130, bbox_inches="tight")
    print("saved repro_fig24.png")


if __name__ == "__main__":
    main()
