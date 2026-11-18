"""R1A step 5: diagnostics for abstention jump, abstain-cascade cost, verifier-rule sensitivity, curve figures."""
import pickle, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "discovery/experiments/scripts")
import R1A_abstain as A
from sklearn.metrics import roc_auc_score
OUT = A.OUT; SPL = A.SPL
rows, res = pickle.load(open(f"{OUT}/abstain_res.pkl", "rb"))
sp = np.array([r["split"] for r in rows]); y = np.array([r["y"] for r in rows])
col = lambda c: np.array([r[c] for r in rows], float)
print("--- what drives the jump? (per-split LR OOF AUC)")
extra_sets = {"answer+agreement": A.SETS["answer"] + A.SETS["agreement"], "structural+agreement": A.SETS["structural"] + A.SETS["agreement"],
              "all_weightonly": A.SETS["all_weightonly"], "all minus noop/valid/n_invalid": [c for c in A.SETS["all_weightonly"] if c not in ("noop", "valid", "n_invalid")],
              "all minus n_ops/n_*/flip": [c for c in A.SETS["all_weightonly"] if c not in ("n_ops", "n_scale", "n_swap", "n_set", "n_add", "n_bias", "n_row", "n_col", "flip")]}
for nm, cols in extra_sets.items():
    X = np.array([[r[c] for c in cols] for r in rows], float); out = []
    for si, s in enumerate(SPL):
        m = sp == si; p = A.oof(X[m], y[m], "lr", reps=3); out.append(f"{s} {roc_auc_score(y[m], p):.3f}")
    print(f"  {nm:40s}", " | ".join(out))
print("--- noop / invalid / n_ops rates and success")
for si, s in enumerate(SPL):
    m = sp == si
    print(f" {s}: noop {col('noop')[m].mean():.3f} (succ {y[m][col('noop')[m]==1].mean() if (col('noop')[m]==1).any() else float('nan'):.3f}); invalid {1-col('valid')[m].mean():.3f}; n_ops dist {np.bincount(col('n_ops')[m].astype(int))}; "
          f"succ by n_ops: " + ", ".join(f"{k}: {y[m][col('n_ops')[m]==k].mean():.2f}(n={int((col('n_ops')[m]==k).sum())})" for k in range(4) if (col('n_ops')[m]==k).any()))
    # agreement x single-op
    for lab, mm in [("flip&agree8", (col('flip')[m]==1)&(col('agree')[m]==1)), ("flip&agree<1", (col('flip')[m]==1)&(col('agree')[m]<1)), ("nonflip&agree8", (col('flip')[m]==0)&(col('agree')[m]==1)), ("nonflip&agree<1", (col('flip')[m]==0)&(col('agree')[m]<1))]:
        print(f"    {lab}: n={mm.sum()} succ={y[m][mm].mean() if mm.sum() else float('nan'):.3f}")

print("\n--- abstain-then-cascade: accept top-coverage answers unverified (cost 0); the rest -> verify (1) then heuristic-order search; per split, per-split LR all_weightonly OOF score")
pr = pickle.load(open(f"{OUT}/curves.pkl", "rb"))[1]
for si, s in enumerate(SPL):
    m = sp == si; sc = res[("per_split", "all_weightonly", "lr", s)][-1]; E = pr[s]; N = E["N"]; rh = E["r_h"]; yy = y[m]
    base_cost_es = 1 + np.where(yy == 1, 0, np.where(np.isinf(rh), N, rh) + 1)      # always-verify cascade (heur early stop)
    o = np.argsort(-sc); line = []
    for cov in (0.0, 0.3, 0.5, 0.7, 0.9):
        acc = np.zeros(len(yy), bool); acc[o[: int(round(cov * len(yy)))]] = True
        succ = np.where(acc, yy == 1, (yy == 1) | np.isfinite(rh))  # accepted ones keep the model answer (unverified)
        cost = np.where(acc, 0, base_cost_es)
        line.append(f"cov {cov:.1f}: repair {succ.mean():.3f} evals {cost.mean():.1f}")
    print(s, " | ".join(line), f" | search-alone repair {E['search_ok'].mean():.3f} evals(early-stop heur) {(np.where(np.isinf(rh),N,rh)+1).mean():.1f}")

print("\n--- verifier-rule sensitivity (cascade k=1 -> search), first answer sample accepted if rule holds (acc computed on scoring inputs)")
for s in SPL:
    C = pickle.load(open(f"{OUT}/cands_{s}.pkl", "rb")); S = pickle.load(open(f"{OUT}/samples_{s}.pkl", "rb")); seeds = sorted(C)
    cur = pickle.load(open(f"{OUT}/curves.pkl", "rb"))[1][s]; srch = cur["search_ok"]; N = cur["N"]
    rules = {}
    for nm in ["recovery>=0.95 (published)", "acc_after==1.0", "any accuracy gain", "acc_after>=acc_clean"]:
        succ = []; cost = []
        for i, sd in enumerate(seeds):
            d = C[sd]; x = S[sd]["samples"][0]; rec = x["recovery"]
            aa = d["acc_f"] + rec * (d["acc_c"] - d["acc_f"]) if rec is not None else None
            acc = {"recovery>=0.95 (published)": rec is not None and rec >= .95, "acc_after==1.0": aa is not None and aa >= 1 - 1e-9, "any accuracy gain": rec is not None and rec > 1e-9,
                   "acc_after>=acc_clean": aa is not None and aa >= d["acc_c"] - 1e-9}[nm]
            truth = bool(x["success"])
            ok = truth if acc else bool(srch[i])
            succ.append(ok); cost.append(1 if acc else 1 + N[i] + 1)
        succ = np.array(succ); rules[nm] = (succ.mean(), np.mean(cost))
        print(f" {s:13s} {nm:28s} cascade repair {succ.mean():.3f} (search alone {srch.mean():.3f}) mean evals(full scan) {np.mean(cost):.1f}")

# ---------- curve figures
curves, R = pickle.load(open(f"{OUT}/curves.pkl", "rb"))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
sty = {"model_bestk": ("model best-of-k (k=min(B,8))", "#d62728", "-"), "search_random": ("search, random order", "#7f7f7f", "-"), "search_heur": ("search, weight-heuristic order", "#1f77b4", "-"),
       "cascade1_heur": ("cascade k=1 -> heuristic search", "#2ca02c", "-"), "cascade1_random": ("cascade k=1 -> random-order search", "#2ca02c", "--"),
       "cascade8_heur": ("cascade k=8 -> heuristic search", "#9467bd", "-"), "random_edits": ("random valid edits (random_policy)", "k", ":")}
for i, s in enumerate(SPL):
    Bs = np.array([r["B"] for r in curves[s]])
    for k, (lab, c, ls) in sty.items():
        v = np.array([r[k] for r in curves[s]], float); ax[i].plot(Bs, v, label=lab, color=c, ls=ls, lw=1.7)
    ax[i].axhline(R[s]["search_ok"].mean(), color="#7f7f7f", lw=.6, ls=":")
    ax[i].set_xscale("log"); ax[i].set_xlabel("forward-pass evaluations per network (budget B)"); ax[i].set_ylabel("share of faulty networks repaired"); ax[i].set_title(f"{s} (n={len(R[s]['N'])})"); ax[i].grid(alpha=.3); ax[i].set_ylim(0, 1)
ax[0].legend(fontsize=7, loc="lower right"); plt.tight_layout(); plt.savefig("discovery/reports/figures/R1A_budget_curves.png", dpi=130)
