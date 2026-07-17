"""
Independent reproduction of main-paper Figure 1.

Caption: (Left) True function (dashed), CauchyNet (orange), width-matched
ReLU FFN (blue) on a rational-spike target with a steep peak near x = 0.5.
(Right) Training and validation loss (log scale) over 500 epochs,
median +/- std across 10 independent runs.

Target (README minimal example / peak height ~400 at x=0.5, domain [-1,1]):
    f(x) = 4 / ((x - 0.5)^2 + 0.01) + sin(3x)

Written from scratch; trainable-pole CauchyNet (paper Sec. 4.1 legacy setting)
vs a width-matched single-hidden-layer ReLU FFN.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = 64          # hidden width, both models
EPOCHS = 500
LR = 1e-2
RUNS = 10


def target(x):
    return 4.0 / ((x - 0.5) ** 2 + 0.01) + torch.sin(3 * x)


class CauchyNet1D(nn.Module):
    """Trainable poles xi in C^h, complex coefficients lambda; y = Re(sum lam/(xi-x))."""
    def __init__(self, h=H):
        super().__init__()
        self.lam = nn.Parameter(torch.normal(0., 0.1, size=(h, 1), dtype=torch.cfloat))
        ang = 2 * math.pi * torch.rand(h)
        self.xi = nn.Parameter(torch.complex(2 * math.pi * torch.cos(ang), torch.sin(ang)))

    def forward(self, x):
        xc = torch.complex(x, torch.zeros_like(x))          # (B,1)
        act = 1.0 / (self.xi.unsqueeze(0) - xc + 1e-8)      # (B,h)
        out = act @ self.lam                                # (B,1) complex
        return out.real, out.imag


class FFN(nn.Module):
    def __init__(self, h=H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)


def run_once(model, xtr, ytr, xva, yva, imag_pen=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.MSELoss()
    tr_curve, va_curve = [], []
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(xtr)
        if isinstance(out, tuple):
            yr, yi = out
            loss = crit(yr, ytr) + imag_pen * crit(yi, torch.zeros_like(yi))
        else:
            loss = crit(out, ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr_curve.append(loss.item())
        model.eval()
        with torch.no_grad():
            o = model(xva); o = o[0] if isinstance(o, tuple) else o
            va_curve.append(crit(o, yva).item())
    return np.array(tr_curve), np.array(va_curve)


def main():
    torch.manual_seed(0); np.random.seed(0)
    x = torch.linspace(-1, 1, 200).unsqueeze(1)
    y_raw = target(x)
    # scale to [0,1] for training stability, keep raw for the left panels
    ymin, ymax = y_raw.min(), y_raw.max()
    y = (y_raw - ymin) / (ymax - ymin)
    idx = torch.randperm(200)
    tr, va = idx[:150], idx[150:]
    xtr, ytr, xva, yva = x[tr], y[tr], x[va], y[va]

    curves = {"CauchyNet": {"tr": [], "va": []}, "FFN": {"tr": [], "va": []}}
    preds = {}
    for name, ctor in [("CauchyNet", CauchyNet1D), ("FFN", FFN)]:
        for s in range(RUNS):
            torch.manual_seed(s); np.random.seed(s)
            m = ctor()
            t, v = run_once(m, xtr, ytr, xva, yva)
            curves[name]["tr"].append(t); curves[name]["va"].append(v)
            if s == 0:
                m.eval()
                with torch.no_grad():
                    o = m(x); o = o[0] if isinstance(o, tuple) else o
                preds[name] = (o.flatten() * (ymax - ymin) + ymin).numpy()
        print(f"{name}: final val MSE median over {RUNS} runs = "
              f"{np.median([v[-1] for v in curves[name]['va']]):.2e}")

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6])
    xn = x.flatten().numpy(); yn = y_raw.flatten().numpy()
    for i, name, color in [(0, "CauchyNet", "tab:orange"), (1, "FFN", "tab:blue")]:
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(xn, yn, "k--", label="True")
        ax.plot(xn, preds[name], color=color, label=name)
        ax.set_title(name); ax.legend()
    ax = fig.add_subplot(gs[:, 1])
    ep = np.arange(1, EPOCHS + 1)
    for name, color in [("CauchyNet", "tab:orange"), ("FFN", "tab:blue")]:
        tr_m = np.median(np.stack(curves[name]["tr"]), axis=0)
        va_m = np.median(np.stack(curves[name]["va"]), axis=0)
        ax.plot(ep, tr_m, color=color, label=name)
        ax.plot(ep, va_m, color=color, alpha=0.45, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Training & Validation (Median over 10 runs)"); ax.legend()
    fig.tight_layout()
    fig.savefig("repro_fig1.png", dpi=130, bbox_inches="tight")
    print("saved repro_fig1.png")


if __name__ == "__main__":
    main()
