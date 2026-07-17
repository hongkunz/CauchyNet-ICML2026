"""Compare rerun experiment JSONs (fresh, this machine) against the shipped
results in the release package. Reports the key headline metric per experiment."""
import json, os

# NEW = fresh rerun results (this audit); OLD = shipped results in the release
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, "results")
OLD = os.path.join(HERE, "..", "..", "experiments", "results")


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def walk_min_mse(obj):
    """Collect (path, value) for any dict key that looks like a headline number."""
    out = []
    def rec(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float)) and any(
                        t in k.lower() for t in ("mse_mean", "mae_mean", "ratio", "mean_mse")):
                    out.append((f"{path}/{k}", v))
                else:
                    rec(v, f"{path}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o[:6]):
                rec(v, f"{path}[{i}]")
    rec(obj, "")
    return dict(out)


FILES = [
    "exp1_epsilon_results.json", "exp2_overhead_results.json",
    "exp3_piecewise_results.json", "exp4_param_matched_results.json",
    "exp5_convergence_results.json", "exp6_multilayer_results.json",
    "exp7_uci_results.json", "exp8_rational_results.json",
    "exp9_delta_sweep_results.json", "exp10_sample_efficiency_results.json",
    "exp11_fixed_pole_ridge_results.json",
]

for fn in FILES:
    new, old = load(NEW, fn), load(OLD, fn)
    print(f"\n=== {fn} ===")
    if new is None:
        print("  [rerun missing]"); continue
    if old is None:
        print("  [shipped missing]"); continue
    n, o = walk_min_mse(new), walk_min_mse(old)
    keys = sorted(set(n) & set(o))
    if not keys:
        print("  (no common headline keys; compare manually)")
        continue
    worst = 0.0
    for k in keys[:14]:
        a, b = o[k], n[k]
        rel = abs(a - b) / max(abs(a), 1e-12)
        worst = max(worst, rel)
        flag = "  " if rel < 0.35 else "!!"
        print(f"  {flag} {k:<58s} shipped={a:<12.4g} rerun={b:<12.4g} ({rel*100:5.1f}%)")
    if len(keys) > 14:
        print(f"  ... {len(keys)-14} more keys compared silently")
        for k in keys[14:]:
            rel = abs(o[k] - n[k]) / max(abs(o[k]), 1e-12)
            worst = max(worst, rel)
    print(f"  -> max relative deviation across {len(keys)} keys: {worst*100:.1f}%")
