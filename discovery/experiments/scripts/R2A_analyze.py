import sys, pickle, json, collections
import numpy as np
sys.path.insert(0, "discovery/experiments/scripts")
import R2A_feat as F
import R2A_main as M
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = M.OUT
logf = open(f"{OUT}/analyze.log", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
ev = pickle.load(open(f"{OUT}/eval.pkl", "rb")); models = pickle.load(open(f"{OUT}/models.pkl", "rb"))
rows_val = M.load_sp("val_trained")
rng = np.random.default_rng(0)

# -------- overlap on test_trained (primary arm)
d = ev["test_trained"]; fa = d["faulty"]; a = d["arms"]["trained_only"]["succ"]; l = d["llm"]
l = (l >= 0.5).astype(float)
def phi_kappa(x, y):
    n = len(x); n11 = ((x == 0) & (y == 0)).sum(); n10 = ((x == 0) & (y == 1)).sum(); n01 = ((x == 1) & (y == 0)).sum(); n00 = ((x == 1) & (y == 1)).sum()
    po = (n00 + n11) / n; pe = ((x == 1).mean() * (y == 1).mean() + (x == 0).mean() * (y == 0).mean())
    return dict(both_fail=int(n11), ranker_only_fail=int(n10), llm_only_fail=int(n01), both_ok=int(n00), phi=float(np.corrcoef(x, y)[0, 1]), kappa=float((po - pe) / (1 - pe)))
for nm, msk in [("all", np.ones(len(a), bool)), ("faulty", fa), ("clean", ~fa)]:
    log("overlap", nm, phi_kappa(a[msk], l[msk]))
log("union ranker-or-scale_rl success overall", float(np.maximum(a, l).mean()), "faulty", float(np.maximum(a, l)[fa].mean()), "oracle best-single-of-two:", max(a.mean(), l.mean()))
log("ranker failures also outside candidate space (no candidate repairs):", int(((a == 0) & (d["oracle"] == 0)).sum()), "of", int((a == 0).sum()))
log("\nBy fault (test_trained, faulty): n, ranker, scale_rl, oracle ceiling")
tab = collections.defaultdict(list)
for i, f in enumerate(d["fault"]):
    if f: tab[f].append(i)
for f, ix in sorted(tab.items()):
    log(f"  {f:16s} n={len(ix):3d} ranker {a[ix].mean():.2f} scale_rl {l[ix].mean():.2f} ceiling {d['oracle'][ix].mean():.2f}")
log("By depth (faulty):")
dep = np.array(d["depth"])
for k in (1, 2):
    ix = np.flatnonzero(fa & (dep == k)); log(f"  depth {k} n={len(ix)} ranker {a[ix].mean():.2f} scale_rl {l[ix].mean():.2f}")
log("By family (faulty):")
fam = np.array(d["family"])
for f in sorted(set(fam)):
    ix = np.flatnonzero(fa & (fam == f)); log(f"  {f:10s} n={len(ix):3d} ranker {a[ix].mean():.2f} scale_rl {l[ix].mean():.2f}")
# error-correlation across arms on all splits
log("\nPer-split phi(fail_ranker, fail_scale_rl) faulty-only, trained_only arm")
for s, dd in ev.items():
    m = dd["faulty"]; x = dd["arms"]["trained_only"]["succ"][m]; y = (dd["llm"][m] >= .5).astype(float)
    log(f"  {s}: {phi_kappa(x, y)}")

# -------- permutation importance (primary arm), val_trained top-1 repair (faulty)
m, am, tau, p = models["trained_only"]
base = M.top1_hit(m, rows_val)
imp = {}
Xall = np.vstack([r["X"] for r in rows_val if not r["clean"]])
for j, nme in enumerate(F.FEAT_NAMES):
    drops = []
    for rep in range(2):
        perm = rng.permutation(len(Xall)); col = Xall[perm, j]
        h, o = [], 0
        for r in rows_val:
            if r["clean"] or len(r["y"]) == 0: continue
            n = len(r["y"]); X2 = r["X"].copy(); X2[:, j] = col[o:o + n]; o += n
            h.append(r["y"][int(np.argmax(m.predict_proba(X2)[:, 1]))])
        drops.append(base - float(np.mean(h)))
    imp[nme] = float(np.mean(drops))
srt = sorted(imp.items(), key=lambda kv: -kv[1])
log(f"\nRanker permutation importance on val_trained (base top-1 repair {base:.3f}; drop in top-1 repair when column permuted across candidates)")
for k, v in srt[:25]: log(f"  {k:28s} {v:+.4f}")
groups = {"spec": F.SPEC_NAMES, "op descriptor": F.OP_NAMES, "neuron feats A": ["nb_" + n for n in F.NetInfo.NF_NAMES], "neuron feats B (swap)": ["nb2_" + n for n in F.NetInfo.NF_NAMES], "candidate/entry/pair": F.CAND_NAMES}
for g, names in groups.items():
    ix = [F.FEAT_NAMES.index(n) for n in names]
    drops = []
    for rep in range(3):
        perm = rng.permutation(len(Xall)); o = 0; h = []
        for r in rows_val:
            if r["clean"] or len(r["y"]) == 0: continue
            n = len(r["y"]); X2 = r["X"].copy(); X2[:, ix] = Xall[perm[o:o + n]][:, ix]; o += n
            h.append(r["y"][int(np.argmax(m.predict_proba(X2)[:, 1]))])
        drops.append(base - float(np.mean(h)))
    log(f"  GROUP {g:24s} drop {np.mean(drops):+.4f}")
json.dump(imp, open(f"{OUT}/importance.json", "w"), indent=1)
# abstain head importance (AUC drop on val_trained)
sc = [M.scores(m, r) for r in rows_val]; Xv, yv = M.abstain_xy(rows_val, sc); b = roc_auc_score(yv, am.predict_proba(Xv)[:, 1])
names = F.SPEC_NAMES + ["max_gain_z", "n_gain_out", "min_cluster_cons", "n_cons_neg", "n_dead", "n_always_on", "max_bias_rel", "max_entry_z", "n_dup_entries", "frac_cn_small", "m0_abs", "top1", "top2", "top3", "gap12", "ent", "sum_p", "n_p_gt_0.1", "n_cand"]
ai = []
for j, nme in enumerate(names):
    ds = []
    for rep in range(10):
        X2 = Xv.copy(); X2[:, j] = rng.permutation(X2[:, j]); ds.append(b - roc_auc_score(yv, am.predict_proba(X2)[:, 1]))
    ai.append((nme, float(np.mean(ds))))
ai.sort(key=lambda kv: -kv[1])
log(f"\nAbstain head permutation importance (val_trained AUC {b:.3f}):")
for k, v in ai[:12]: log(f"  {k:20s} {v:+.4f}")

# -------- figures
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
splits = list(ev.keys()); w = 0.2
for i, arm in enumerate(["trained_only", "compiled_only", "mixed"]):
    ax[0].bar(np.arange(len(splits)) + i * w, [ev[s]["arms"][arm]["succ"].mean() for s in splits], w, label=arm)
ax[0].bar(np.arange(len(splits)) + 3 * w, [ev[s]["llm"].mean() for s in splits], w, label="scale_rl", color="k")
ax[0].set_xticks(np.arange(len(splits)) + 1.5 * w); ax[0].set_xticklabels(splits, rotation=20); ax[0].set_ylabel("overall success"); ax[0].legend(fontsize=8)
top = srt[:15][::-1]; ax[1].barh([k for k, _ in top], [v for _, v in top]); ax[1].set_xlabel("drop in top-1 repair (permutation)"); ax[1].set_title("ranker feature importance (val_trained)")
plt.tight_layout(); plt.savefig("discovery/reports/figures/R2A_summary.png", dpi=130)
