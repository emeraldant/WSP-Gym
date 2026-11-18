import pandas as pd, numpy as np
d = pd.read_csv("discovery/experiments/outputs/R1B/merged.csv")
rng = np.random.default_rng(3)
def ci(x, B=5000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return f"{x.mean():.3f} [{np.percentile(m,2.5):.3f},{np.percentile(m,97.5):.3f}]"
d["loc"] = d.succ_M.astype(float); d["diff"] = d["loc"] - d.rand_p_M; d["diffO"] = d.succ_O.astype(float) - d.rand_p_O
pre = d[d.split.isin(["test_id", "test_trained", "test_faults"])]
print("PRIMARY pooled 3 splits n=%d: local %s random %s diff %s" % (len(pre), ci(pre["loc"]), ci(pre.rand_p_M), ci(pre["diff"])))
print("ALL 5 splits n=%d: local %s random %s diff %s" % (len(d), ci(d["loc"]), ci(d.rand_p_M), ci(d["diff"])))
print("oracle-loc ALL diff", ci(d.diffO), "oracle", ci(d.succ_O.astype(float)), "global", ci(d.global_any.astype(float)))
for f in ["spurious_direction", "scale_in_row"]:
    x = d[(d.split == "test_faults") & (d.fault == f)]
    print(f, len(x), "loc", ci(x["loc"]), "rand", ci(x.rand_p_M), "diff", ci(x["diff"]), "oracle", ci(x.succ_O.astype(float)), "global", ci(x.global_any.astype(float)), "base", ci(x.base_succ.astype(float)))
    e = x[x.n_M > 0]; print("   edit-type n", len(e), "loc", ci(e["loc"]), "rand", ci(e.rand_p_M), "diff", ci(e["diff"]))
    for h, g in e.groupby("hit_M"): print("   hit", h, len(g), ci(g["loc"]), ci(g.rand_p_M))
e = d[d.n_M > 0]
for h, g in e.groupby("hit_M"): print("ALL hit", h, len(g), "loc", ci(g["loc"]), "rand", ci(g.rand_p_M), "diff", ci(g["diff"]))
# seen vs unseen faults splits
print("seen-fault splits (id,trained,larger,families) diff", ci(d[d.split != "test_faults"]["diff"]), " unseen test_faults", ci(d[d.split == "test_faults"]["diff"]))
# local pool size effect: local frac of global
d["frac"] = d.n_M / d.N_global
print("mean local/global size frac (edit)", e.assign(fr=e.n_M/e.N_global).fr.mean())
# not-hit & hit share
print(d[d.n_M>0].groupby("split").hit_M.mean().round(3).to_dict())
# unit-level hit counts for noop: share failing
print(d.groupby("split").apply(lambda x: (x.cat!="edit").mean()).round(3).to_dict())
# held-out spurious
