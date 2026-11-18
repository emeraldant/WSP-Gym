import sys, json, numpy as np, pandas as pd
from weightgym.env import load_split
from weightgym.edits import apply_edit, validate_edit
OUT = "discovery/experiments/outputs/R0Q2/"
POL = sys.argv[1]
rng = np.random.default_rng(1)
d = pd.read_csv(OUT + "enum.csv")
def boot(x, f=np.mean, B=2000):
    x = np.asarray(x, float); 
    if len(x)==0: return (np.nan,)*3
    bs = [f(x[rng.integers(len(x), size=len(x))]) for _ in range(B)]
    return f(x), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def fm(t): return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"
# model rollouts
recs = []
for split in ["test_id", "test_trained"]:
    _, eps = load_split(f"data/{split}.jsonl"); E = {e.seed: e for e in eps}
    for ln in list(open(f"runs/final/{POL}/{split}.jsonl"))[1:]:
        r = json.loads(ln); e = E[r["seed"]]
        restores = False
        if r["valid"] and e.fault and r["ops"]:
            try:
                n2 = apply_edit(e.net, r["ops"])
                restores = all(np.allclose(a, b, atol=1e-6) for a, b in zip(n2.Ws + n2.bs, e.clean.Ws + e.clean.bs))
            except Exception: pass
        recs.append(dict(split=split, seed=r["seed"], success=r["success"], valid=r["valid"], noop=r["noop"], n_ops=r["n_ops"], restores=restores, recovery=r["recovery"]))
m = pd.DataFrame(recs)
d = d.merge(m, on=["split", "seed"])
d.to_csv(OUT + f"merged_{POL}.csv", index=False)
print("model rows merged", len(d), "model dup?", m.duplicated(["split","seed"]).sum())
d["n_rep_all"] = d.n_rep_base + d.n_rep_ext + (d.n_rep_ext_plus_inv - d.n_rep_ext)
fa = d[d.fault.notna()].copy()
lines = []
P = lambda *a: (print(*a), lines.append(" ".join(map(str, a))))
for split in ["test_id", "test_trained"]:
    f = fa[fa.split == split]
    P(f"\n=== {split}: n_faulty={len(f)} ===")
    P("model success", fm(boot(f.success)), " | planted inverse repairs (sanity)", fm(boot(f.inv_repairs)))
    P("inverse inside baseline candidate space:", fm(boot(f.inv_in_base)))
    for col in ["n_rep_base", "n_rep_base_plus_inv", "n_rep_all"]:
        P(f"{col}: mean {f[col].mean():.2f} median {f[col].median()} | frac==0 {np.mean(f[col]==0):.3f} frac==1 {np.mean(f[col]==1):.3f} frac>=2 {np.mean(f[col]>=2):.3f} frac>=5 {np.mean(f[col]>=5):.3f}")
    P("(b) planted inverse is the ONLY repair (base space + inverse):", fm(boot(f.n_rep_base_plus_inv == 1)), "| (ext space + inverse):", fm(boot(f.n_rep_all == 1)))
    P("   among faults with inverse inside baseline space (n=%d): only repair" % f.inv_in_base.sum(), fm(boot((f[f.inv_in_base].n_rep_base == 1))))
    P("   networks where baseline single-op search space contains NO repair:", fm(boot(f.n_rep_base == 0)))
    f = f.assign(mb=pd.cut(f.n_rep_base_plus_inv, [-1, 1, 2, 4, 9, 10**6], labels=["1", "2", "3-4", "5-9", "10+"]))
    t = f.groupby("mb", observed=True).agg(n=("success", "size"), model_success=("success", "mean"), restores_given_success=("restores", lambda s: np.nan)).drop(columns="restores_given_success")
    P("model success by multiplicity bin (base space + inverse):\n" + t.round(3).to_string())
    f2 = f.assign(mb=pd.cut(f.n_rep_all, [-1, 1, 2, 4, 9, 10**6], labels=["1", "2", "3-4", "5-9", "10+"]))
    P("model success by multiplicity bin (EXT space + inverse):\n" + f2.groupby("mb", observed=True).agg(n=("success", "size"), model_success=("success", "mean")).round(3).to_string())
    from scipy.stats import spearmanr
    P("spearman(log multiplicity, success) base+inv: %.3f p=%.3g" % spearmanr(np.log(f.n_rep_base_plus_inv.clip(lower=1)), f.success))
    P("spearman(multiplicity ext+inv, success): %.3f p=%.3g" % spearmanr(f.n_rep_all, f.success))
    s = f[f.success]
    P(f"among model successes (n={len(s)}): edit restores clean weights exactly (planted inverse effect) {fm(boot(s.restores))}; functionally-equivalent alternative {fm(boot(~s.restores))}")
    s1 = s[s.n_rep_base_plus_inv > 1]
    P(f"   among successes where >1 repair exists (n={len(s1)}): alt fraction {fm(boot(~s1.restores))}")
    s2 = s[s.n_rep_base_plus_inv == 1]
    P(f"   among successes where inverse is the only repair (n={len(s2)}): restores {fm(boot(s2.restores))}")
    P("by fault type:")
    t = f.groupby("fault").agg(n=("success", "size"), succ=("success", "mean"), mult_med=("n_rep_base_plus_inv", "median"), mult_mean=("n_rep_base_plus_inv", "mean"),
         inv_only=("n_rep_base_plus_inv", lambda s: (s == 1).mean()), inv_in_space=("inv_in_base", "mean"), ext_mult_med=("n_rep_all", "median"),
         alt_given_succ=("restores", lambda s: np.nan))
    t["alt_given_succ"] = f[f.success].groupby("fault").restores.apply(lambda x: 1 - x.mean())
    P(t.round(3).to_string())
    # logistic w/ fault fixed effects
    f = f.assign(lm=np.log(f.n_rep_base_plus_inv.clip(lower=1)), succ=f.success.astype(float))
    xr = f.lm - f.groupby("fault").lm.transform("mean"); yr = f.succ - f.groupby("fault").succ.transform("mean")
    obs = np.corrcoef(xr, yr)[0, 1]; perm = []
    for _ in range(2000):
        ys = f.groupby("fault").succ.transform(lambda v: rng.permutation(v.values))
        perm.append(np.corrcoef(xr, ys - f.groupby("fault").succ.transform("mean"))[0, 1])
    P("within-fault-type corr(log mult, success)=%.3f, permutation p=%.3f" % (obs, np.mean(np.abs(perm) >= abs(obs))))
    # failed networks: what did model do
    P("model failures: invalid %d, noop %d of %d" % ((~f.valid[~f.success]).sum(), f.noop[~f.success].sum(), (~f.success).sum()))
    # success by 'inverse is in space & unique' 
    P("success when inverse in space & unique-repair: %s ; inverse in space & multi-repair: %s" % (
        fm(boot(f[f.inv_in_base & (f.n_rep_base==1)].success)), fm(boot(f[f.inv_in_base & (f.n_rep_base>1)].success))))
cl = d[d.fault.isna() & (d.split == "test_id")]
P(f"\n=== clean networks test_id n={len(cl)} (candidates mean {cl.n_cand.mean():.0f}) ===")
for c, nm in [("n_same_base", "unchanged accuracy"), ("n_break_base", "accuracy drop >0.05"), ("n_break10_base", "drop >0.10"), ("n_better_base", "improves")]:
    fr = cl[c] / cl.n_cand
    P(f"frac of single-op edits with {nm}: mean per-net {fm(boot(fr))}, pooled {cl[c].sum()/cl.n_cand.sum():.3f}")
P("frac of nets where >=1 edit leaves accuracy unchanged:", fm(boot(cl.n_same_base > 0)))
P("ext space: unchanged frac %.3f ; break frac %.3f" % (cl.n_same_ext.sum()/cl.n_cand_ext.sum(), cl.n_break_ext.sum()/cl.n_cand_ext.sum()))
P("clean: model success (no-op)", fm(boot(cl.success)))
P("clean: model success vs slack (corr of success with frac-unchanged): ", np.corrcoef(cl.success, cl.n_same_base/cl.n_cand)[0,1])
cl = cl.assign(sl=cl.n_same_base / cl.n_cand)
P(cl.groupby("family").agg(n=("sl", "size"), unchanged=("sl", "mean"), succ=("success", "mean")).round(3).to_string())
open(OUT + f"summary_{POL}.txt", "w").write("\n".join(lines))
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for split, c in [("test_id", "C0"), ("test_trained", "C1")]:
    f = fa[fa.split == split]
    ax[0].hist(f.n_rep_base_plus_inv.clip(upper=30), bins=np.arange(0.5, 31.5, 1), alpha=.6, label=split, color=c)
ax[0].set_xlabel("# distinct single-op repairs (base space + planted inverse; clipped at 30)"); ax[0].set_ylabel("networks"); ax[0].legend()
for split, c in [("test_id", "C0"), ("test_trained", "C1")]:
    f = fa[fa.split == split]; f = f.assign(mb=pd.cut(f.n_rep_base_plus_inv, [-1, 1, 2, 4, 9, 10**6], labels=["1", "2", "3-4", "5-9", "10+"]))
    g = f.groupby("mb", observed=True).success.agg(["mean", "size"]); ax[1].plot(range(len(g)), g["mean"], "o-", color=c, label=split)
    ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels(g.index)
ax[1].set_xlabel("multiplicity bin"); ax[1].set_ylabel("final model success"); ax[1].legend()
t = fa.groupby("fault").agg(mult=("n_rep_base_plus_inv", "median"), succ=("success", "mean")).sort_values("succ")
ax[2].scatter(t.mult, t.succ); [ax[2].annotate(k, (r.mult, r.succ), fontsize=7) for k, r in t.iterrows()]
ax[2].set_xlabel("median multiplicity"); ax[2].set_ylabel("success")
plt.tight_layout(); plt.savefig(f"discovery/reports/figures/R0Q2_multiplicity_{POL}.png", dpi=120)
