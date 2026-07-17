"""
INDEPENDENT reproduction of the CauchyNet headline result (paper Table 1,
"Final gap-filling configuration and test errors").

This is written from scratch based only on the PAPER's description of the
model, the target function, the train/val/test split, and the reported
configuration, written independently of the authors' shared.py / best_config code.
Goal: check whether the reported per-model MAEs are recoverable.

Paper Table 1 targets (10 seeds, held-out turning-point regions):
    CauchyNet  MAE 0.0202   median 0.0124   max 0.1618   params 128
    SIREN      MAE 0.0872   median 0.0724   max 0.3349   params 193
    FNN        MAE 0.2032   median 0.1962   max 0.5093   params 193
    N-BEATS    MAE 0.1940   median 0.1860   max 0.8401   params 204

Reported config: h=64, fixed elliptical poles (r_re,r_im)=(2.5,0.4),
lambda_imag=0, lr=5e-2, n_train=200, epochs=3000, Adam, batch=32, 10 seeds.
"""
import math, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = "cpu"


# ---------------------------------------------------------------- target
def f(x):
    # f(x)= sin(2(x-2)) + 0.5 cos(5(x-1)) + 0.05/((x-1)^2+0.1)
    #       + 0.01/((x+0.5)^2+0.05) - 0.01 x^2 + 0.01 x^3   on [-2,2]
    return (torch.sin(2 * (x - 2)) + 0.5 * torch.cos(5 * (x - 1))
            + 0.05 / ((x - 1) ** 2 + 0.1)
            + 0.01 / ((x + 0.5) ** 2 + 0.05)
            - 0.01 * x ** 2 + 0.01 * x ** 3)


def df(x):
    return (2 * torch.cos(2 * (x - 2)) - 2.5 * torch.sin(5 * (x - 1))
            + 0.05 * (-2 * (x - 1)) / (((x - 1) ** 2 + 0.1) ** 2)
            + 0.01 * (-2 * (x + 0.5)) / (((x + 0.5) ** 2 + 0.05) ** 2)
            - 0.02 * x + 0.03 * x ** 2)


def build_data(seed=10):
    """Train/val away from turning points; test near turning points (the gaps)."""
    torch.manual_seed(seed); np.random.seed(seed)
    grid = torch.linspace(-2, 2, 2000)
    with torch.no_grad():
        turning = grid[torch.abs(df(grid)) < 0.15]

    def sample(n, near):
        out = []
        while len(out) < n:
            s = torch.clamp(torch.normal(0., 1., size=(1,)) * 2., -2, 2)
            if bool(torch.any(torch.abs(turning - s) < 0.15)) == near:
                out.append(s)
        return torch.cat(out).unsqueeze(-1)

    tX = sample(200, near=False); vX = sample(50, near=False); teX = sample(100, near=True)
    return tX, f(tX), vX, f(vX), teX, f(teX)


# ---------------------------------------------------------------- models
class CauchyNet(nn.Module):
    """y = (1/h) * Re sum_k lambda_k / (xi_k - x). Poles fixed on an ellipse."""
    def __init__(self, h=64, r_re=2.5, r_im=0.4, seed=0, eps=1e-8):
        super().__init__()
        self.h, self.eps = h, eps
        self.lam = nn.Parameter(torch.randn(h, 1, dtype=torch.cfloat))
        g = torch.Generator(); g.manual_seed(seed)
        ang = 2 * math.pi * torch.rand(h, generator=g)
        self.register_buffer("xi", torch.complex(r_re * torch.cos(ang), r_im * torch.sin(ang)))

    def forward(self, x):
        xc = torch.complex(x.view(-1), torch.zeros(x.shape[0]))
        act = 1.0 / (self.xi.unsqueeze(0) - xc.unsqueeze(1) + self.eps)  # (B,h)
        out = torch.matmul(act, self.lam) / self.h                       # (B,1) complex
        return out.real, out.imag


class SIREN(nn.Module):
    def __init__(self, h=64, w0=30.0):
        super().__init__()
        self.lin_in = nn.Linear(1, h); self.lin_out = nn.Linear(h, 1); self.w0 = w0
        with torch.no_grad():
            nn.init.uniform_(self.lin_in.weight, -1.0, 1.0); nn.init.zeros_(self.lin_in.bias)
            self.lin_in.weight *= w0
            b = (6.0 / h) ** 0.5
            nn.init.uniform_(self.lin_out.weight, -b / w0, b / w0); nn.init.zeros_(self.lin_out.bias)

    def forward(self, x):
        return self.lin_out(torch.sin(self.lin_in(x)))


class FNN(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)


class NBeatsBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = nn.Linear(1, h); self.fc2 = nn.Linear(h, h)
        self.back = nn.Linear(h, 1); self.fore = nn.Linear(h, 1)

    def forward(self, x):
        z = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        return self.back(z), self.fore(z)


class NBeats(nn.Module):
    def __init__(self, h=6, blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(h) for _ in range(blocks)])

    def forward(self, x):
        res, fore = x, 0
        for b in self.blocks:
            bc, fc = b(res); res = res - bc; fore = fore + fc
        return fore


def n_real_params(m):
    return (2 * sum(p.numel() for p in m.parameters() if p.is_complex())
            + sum(p.numel() for p in m.parameters() if not p.is_complex()))


# ---------------------------------------------------------------- train
def train_eval(model, tl, vl, teX, teY, epochs, lr, imag_pen=0.0):
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.StepLR(opt, step_size=epochs, gamma=0.5)
    crit = nn.MSELoss()
    best_v, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xb, yb in tl:
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                yr, yi = out
                loss = crit(yr, yb) + imag_pen * crit(yi, torch.zeros_like(yi))
            else:
                loss = crit(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            v = 0.0; n = 0
            for xb, yb in vl:
                o = model(xb); o = o[0] if isinstance(o, tuple) else o
                v += crit(o, yb).item(); n += 1
            v /= n
            if v < best_v:
                best_v = v
                best_state = {k: val.clone() for k, val in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        o = model(teX); o = o[0] if isinstance(o, tuple) else o
        err = np.abs(o.numpy().flatten() - teY.numpy().flatten())
    return err, n_real_params(model)


def run(name, factory, tl, vl, teX, teY, n_seeds, epochs, lr):
    all_err, params = [], None
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        err, params = train_eval(factory(s), tl, vl, teX, teY, epochs, lr)
        all_err.append(err)
        print(f"  {name} seed{s}: MAE {err.mean():.4f}")
    e = np.concatenate(all_err)
    return {"mae_mean": float(e.mean()), "mae_median": float(np.median(e)),
            "mae_max": float(e.max()), "params": int(params)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3000)
    args = ap.parse_args()

    tX, tY, vX, vY, teX, teY = build_data(seed=10)
    print(f"Train {len(tX)}  Val {len(vX)}  Test {len(teX)}  "
          f"| seeds={args.seeds} epochs={args.epochs}")
    tl = DataLoader(TensorDataset(tX, tY), batch_size=32, shuffle=True)
    vl = DataLoader(TensorDataset(vX, vY), batch_size=32, shuffle=False)

    LR, H = 5e-2, 64
    t0 = time.time()
    res = {}
    res["CauchyNet"] = run("CauchyNet", lambda s: CauchyNet(h=H, seed=s), tl, vl, teX, teY, args.seeds, args.epochs, LR)
    res["SIREN"]     = run("SIREN",     lambda s: SIREN(h=H),             tl, vl, teX, teY, args.seeds, args.epochs, LR)
    res["FNN"]       = run("FNN",       lambda s: FNN(h=H),               tl, vl, teX, teY, args.seeds, args.epochs, LR)
    res["N-BEATS"]   = run("N-BEATS",   lambda s: NBeats(h=6, blocks=3),  tl, vl, teX, teY, args.seeds, args.epochs, LR)

    paper = {"CauchyNet": (0.0202, 0.0124, 0.1618, 128),
             "SIREN": (0.0872, 0.0724, 0.3349, 193),
             "FNN": (0.2032, 0.1962, 0.5093, 193),
             "N-BEATS": (0.1940, 0.1860, 0.8401, 204)}
    print("\n" + "=" * 78)
    print(f"{'Model':<11}{'MAE(mine)':>10}{'MAE(paper)':>12}{'med(mine)':>11}"
          f"{'med(paper)':>12}{'params':>8}{'paper':>7}")
    print("=" * 78)
    for k in res:
        r, p = res[k], paper[k]
        print(f"{k:<11}{r['mae_mean']:>10.4f}{p[0]:>12.4f}{r['mae_median']:>11.4f}"
              f"{p[1]:>12.4f}{r['params']:>8}{p[3]:>7}")
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
    with open("repro_results.json", "w") as fp:
        json.dump(res, fp, indent=2)


if __name__ == "__main__":
    main()
