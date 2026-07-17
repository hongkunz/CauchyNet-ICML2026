"""
Independent reproduction of supp Fig. 19 (legacy M4 trend case study).

Protocol from supp Experiment 4: 700-observation M4 trend subset
(M4_trend_data.csv, columns X, Y), trend normalized to [-1, 1],
350/175/175 train/val/test split, 200 epochs, and the 7-model comparison
from the shipped figure: CauchyNet, SIREN, NBeats, LSTM, Transformer,
RBF, Informer (configs per supp 'Per-model configurations', h=128).

Paper's qualitative claim: faster convergence and lower absolute-error
spread for CauchyNet on this case study.

Outputs: repro_fig19_losses.png (median +/- std over runs, log scale),
         repro_fig19_boxplot.png (test absolute errors), stats to stdout.
"""
import math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H, EPOCHS, RUNS, LR, WD, BATCH = 128, 200, 5, 1e-2, 1e-5, 32


# ---------------------------------------------------------------- models
class CauchyNet(nn.Module):
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


class SIREN(nn.Module):
    def __init__(self, h=H, w0=30.0):
        super().__init__()
        self.fc1, self.fc2, self.w0 = nn.Linear(1, h), nn.Linear(h, 1), w0
        with torch.no_grad():
            self.fc1.weight.uniform_(-1, 1)
            b = (6 / h) ** 0.5 / w0
            self.fc2.weight.uniform_(-b, b)

    def forward(self, x):
        return self.fc2(torch.sin(self.w0 * self.fc1(x)))


class NBeats1(nn.Module):          # single generic block per supp param table
    def __init__(self, h=H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)


class RBF(nn.Module):
    def __init__(self, h=H):
        super().__init__()
        self.c = nn.Parameter(torch.linspace(-1, 1, h))
        self.logw = nn.Parameter(torch.zeros(h))
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        z = torch.exp(-((x - self.c) ** 2) * torch.exp(self.logw))
        return self.out(z)


class LSTMReg(nn.Module):
    def __init__(self, h=H):
        super().__init__()
        self.l = nn.LSTM(1, h, 1, batch_first=True)
        self.fc = nn.Linear(h, 1)

    def forward(self, x):
        o, _ = self.l(x.unsqueeze(1))
        return self.fc(o[:, -1])


class TransformerReg(nn.Module):
    def __init__(self, h=H, nhead=2, ff=2*H):
        super().__init__()
        self.embed = nn.Linear(1, h)
        self.enc = nn.TransformerEncoderLayer(d_model=h, nhead=nhead,
                                              dim_feedforward=ff, batch_first=True)
        self.fc = nn.Linear(h, 1)

    def forward(self, x):
        z = self.embed(x).unsqueeze(1)
        return self.fc(self.enc(z)[:, -1])


def make_informer():
    return TransformerReg(nhead=1, ff=H)


MODELS = {
    "CauchyNet":   lambda: CauchyNet(),
    "SIREN":       lambda: SIREN(),
    "NBeats":      lambda: NBeats1(),
    "LSTM":        lambda: LSTMReg(),
    "Transformer": lambda: TransformerReg(),
    "RBF":         lambda: RBF(),
    "Informer":    make_informer,
}
COLORS = {"CauchyNet": "tab:orange", "SIREN": "tab:blue", "NBeats": "tab:green",
          "LSTM": "tab:red", "Transformer": "tab:purple", "RBF": "tab:gray",
          "Informer": "tab:olive"}


# ---------------------------------------------------------------- data
def load_data(seed):
    df = pd.read_csv("M4_trend_data.csv")
    y = df["Y"].to_numpy(dtype=np.float32)[:700]
    y = 2 * (y - y.min()) / (y.max() - y.min()) - 1          # trend -> [-1, 1]
    x = np.linspace(-1, 1, len(y)).astype(np.float32)        # time index -> [-1, 1]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    tr, va, te = idx[:350], idx[350:525], idx[525:]
    t = lambda a: torch.tensor(a).unsqueeze(1)
    return (t(x[tr]), t(y[tr]), t(x[va]), t(y[va]), t(x[te]), t(y[te]))


def train_one(model, data):
    xt, yt, xv, yv, xe, ye = data
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.MSELoss()
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt, yt),
                                     batch_size=BATCH, shuffle=True)
    tr_c, va_c = [], []
    best_v, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        tot, nb = 0., 0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                yr, yi = out
                loss = crit(yr, yb) + 0.1 * crit(yi, torch.zeros_like(yi))
            else:
                loss = crit(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        tr_c.append(tot / nb)
        model.eval()
        with torch.no_grad():
            o = model(xv); o = o[0] if isinstance(o, tuple) else o
            v = float(crit(o, yv))
        va_c.append(v)
        if v < best_v:
            best_v, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        o = model(xe); o = o[0] if isinstance(o, tuple) else o
        errs = np.abs((o - ye).numpy().flatten())
    return np.array(tr_c), np.array(va_c), errs


def main():
    curves, errors = {}, {}
    for name, ctor in MODELS.items():
        trs, vas, errs = [], [], []
        t0 = time.time()
        for s in range(RUNS):
            torch.manual_seed(s); np.random.seed(s)
            tr, va, e = train_one(ctor(), load_data(seed=0))
            trs.append(tr); vas.append(va); errs.append(e)
        curves[name] = (np.stack(trs), np.stack(vas))
        errors[name] = np.concatenate(errs)
        n_params = sum(p.numel() * (2 if p.is_complex() else 1)
                       for p in ctor().parameters())
        print(f"{name:<12} final val MSE median={np.median([v[-1] for v in vas]):.2e}  "
              f"test MAE={errors[name].mean():.4f}  params={n_params}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # loss curves (median +/- std)
    fig, ax = plt.subplots(figsize=(10, 7))
    ep = np.arange(1, EPOCHS + 1)
    for name in MODELS:
        tr, va = curves[name]
        med = np.median(tr, 0)
        ax.plot(ep, med, color=COLORS[name], label=name)
        ax.fill_between(ep, np.maximum(med - tr.std(0), 1e-12), med + tr.std(0),
                        color=COLORS[name], alpha=0.15)
        ax.plot(ep, np.median(va, 0), color=COLORS[name], ls="--", alpha=0.5)
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title(f"M4 trend — Training & Validation (Median ± Std, {RUNS} runs)")
    ax.legend()
    fig.tight_layout(); fig.savefig("repro_fig19_losses.png", dpi=130, bbox_inches="tight")

    # absolute-error boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(MODELS)
    ax.boxplot([errors[n] for n in names], tick_labels=names, showfliers=False)
    ax.set_ylabel("Absolute Error")
    ax.set_title("M4 trend — test absolute errors (independent rerun)")
    fig.tight_layout(); fig.savefig("repro_fig19_boxplot.png", dpi=130, bbox_inches="tight")
    print("saved repro_fig19_losses.png, repro_fig19_boxplot.png")


if __name__ == "__main__":
    main()
