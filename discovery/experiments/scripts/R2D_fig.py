import pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import gammaln
OUT = "discovery/experiments/outputs/R2D"
OH = pickle.load(open(f"{OUT}/orders_h.pkl", "rb"))
def rec(a, f, c):
    with np.errstate(all="ignore"): return (a - f) / (c - f)
def lc(n, k): return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
def pr(N, m, B):
    B = int(min(max(B, 0), N))
    if m == 0 or B <= 0: return 0.0
    return 1.0 if B > N - m else 1 - np.exp(lc(N - m, B) - lc(N, B))
grid = np.unique(np.round(np.logspace(0, np.log10(1500), 40)).astype(int))
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, s in zip(axs, ["test_trained", "test_id", "test_external"]):
    R = pickle.load(open(f"{OUT}/rich_{s}.pkl", "rb")); sd = sorted(R)
    N = []; m = []; ph = []; m1 = []
    for x in sd:
        d = R[x]; ok = rec(d["acc"], d["acc_f"], d["acc_c"]) >= 0.95; o = OH[s]["order"][x]; pos = np.flatnonzero(ok[o])
        N.append(len(ok)); m.append(ok.sum()); ph.append(pos[0] + 1 if len(pos) else np.inf); m1.append(d["samples"][0]["success"])
    N = np.array(N); m = np.array(m); ph = np.array(ph); m1 = np.array(m1, float)
    rr = [np.mean([pr(n, k, B) for n, k in zip(N, m)]) for B in grid]
    rh = [np.mean(ph <= B) for B in grid]
    cr = [np.mean(np.where(m1 > 0, 1, [pr(n, k, B - 1) for n, k in zip(N, m)])) for B in grid]
    ch = [np.mean(np.where(m1 > 0, 1, ph <= B - 1)) for B in grid]
    ax.plot(grid, rr, "C0--", label="rich, random order"); ax.plot(grid, rh, "C0-", label="rich, heuristic order")
    ax.plot(grid, cr, "C3--", label="cascade(k=1)+rich, random"); ax.plot(grid, ch, "C3-", label="cascade(k=1)+rich, heuristic")
    ax.axhline(m1.mean(), color="gray", ls=":", label="scale_rl sample 0"); ax.axvline(25, color="k", lw=.5)
    ax.set_xscale("log"); ax.set_title(f"{s} (n={len(N)})"); ax.set_xlabel("forward-pass budget B")
axs[0].set_ylabel("repair rate (faulty nets)"); axs[0].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("discovery/reports/figures/R2D_budget_curves.png", dpi=130)
