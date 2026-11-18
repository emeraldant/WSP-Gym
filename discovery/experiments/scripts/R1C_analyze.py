import json, numpy as np, pandas as pd, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
OUT = "discovery/experiments/outputs/R1C"
rng = np.random.default_rng(0)
B = 5000
SPL = ["test_id", "test_larger", "test_families", "test_faults", "test_trained", "test_external"]

def roll(pol, sp):
    rows = [json.loads(l) for l in open(f"runs/final/{pol}/{sp}.jsonl")][1:]
    d = pd.DataFrame(rows).drop_duplicates("seed").set_index("seed")
    return d["success"].astype(float)

def load(sp):
    s = pd.read_csv(f"{OUT}/solver_{sp}.csv").set_index("seed")
    for pol in ["scale_rl", "scale_rl_greedy", "baseline_search", "baseline_noop"]:
        s[pol] = roll(pol, sp).reindex(s.index)
    s["solver"] = s.prompt_success.astype(float)
    s["solver_params"] = s.params_success.astype(float)
    s["clean"] = s.fault == "clean"
    return s

def boot_diff(a, b):
    n = len(a); idx = rng.integers(0, n, size=(B, n))
    d = a.values[idx].mean(1) - b.values[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def boot_ci(a):
    n = len(a); idx = rng.integers(0, n, size=(B, n)); m = a.values[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

D = {sp: load(sp) for sp in SPL}
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

P("## Headline table (solver = prompt-only alignment solver; scale_rl = final trained model, 1 sample/net)")
P("| split | n faulty | solver repair [95% CI] | scale_rl repair [95% CI] | diff (solver-model) [95% paired CI] | solver-only / model-only wins | n clean | solver clean-acc | scale_rl clean-acc | diff clean [CI] |")
P("|---|---|---|---|---|---|---|---|---|---|")
tab = {}
for sp in SPL:
    s = D[sp]; f = s[~s.clean]; c = s[s.clean]
    sc = boot_ci(f.solver); mc = boot_ci(f.scale_rl)
    d = boot_diff(f.solver, f.scale_rl)
    sw = int(((f.solver == 1) & (f.scale_rl == 0)).sum()); mw = int(((f.solver == 0) & (f.scale_rl == 1)).sum())
    dc = boot_diff(c.solver, c.scale_rl)
    tab[sp] = dict(solver=f.solver.mean(), model=f.scale_rl.mean(), diff=d, clean_s=c.solver.mean(), clean_m=c.scale_rl.mean())
    P(f"| {sp} | {len(f)} | {f.solver.mean():.3f} [{sc[0]:.3f},{sc[1]:.3f}] | {f.scale_rl.mean():.3f} [{mc[0]:.3f},{mc[1]:.3f}] | {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] | {sw}/{mw} | {len(c)} | {c.solver.mean():.3f} | {c.scale_rl.mean():.3f} | {dc[0]:+.3f} [{dc[1]:+.3f},{dc[2]:+.3f}] |")
P()
P("## Overall success (faulty + clean, the gym's headline metric)")
for sp in SPL:
    s = D[sp]; d = boot_diff(s.solver, s.scale_rl)
    P(f"- {sp}: solver {s.solver.mean():.3f}, scale_rl {s.scale_rl.mean():.3f}, noop {s.baseline_noop.mean():.3f}, search {s.baseline_search.mean():.3f}; diff {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
P()
P("## Leak check: ep.task.params canonical net vs prompt-parsed canonical net")
for sp in SPL:
    s = D[sp]
    P(f"- {sp}: identical per-network success in {(s.prompt_success == s.params_success).mean()*100:.1f}% of nets; identical ops {(s.prompt_ops == s.params_ops).mean()*100:.1f}%")
P()
P("## Trained / external: what does the solver do?")
for sp in ["test_trained", "test_external"]:
    s = D[sp]; f = s[~s.clean]
    P(f"- {sp}: solver emits [] on {(s.prompt_nops==0).mean()*100:.1f}% of nets (faulty {(f.prompt_nops==0).mean()*100:.1f}%); noop baseline repair on faulty {f.baseline_noop.mean():.3f} ; solver repair {f.solver.mean():.3f}; scale_rl {f.scale_rl.mean():.3f}")
P()
# per-fault, per-family, depth
def grp(sp, col):
    f = D[sp][~D[sp].clean]
    return f.groupby(col)[["solver", "scale_rl"]].agg(["mean", "count"])
for sp in ["test_id"]:
    f = D[sp][~D[sp].clean]
    P(f"## {sp}: per fault type (n, solver, scale_rl)")
    for k, g in f.groupby("fault"): P(f"- {k}: n={len(g)} solver {g.solver.mean():.3f} scale_rl {g.scale_rl.mean():.3f}")
    P(f"## {sp}: per depth")
    for k, g in f.groupby("layers"): P(f"- layers={k}: n={len(g)} solver {g.solver.mean():.3f} scale_rl {g.scale_rl.mean():.3f}")
    P(f"## {sp}: per family")
    for k, g in f.groupby("family"): P(f"- {k}: n={len(g)} solver {g.solver.mean():.3f} scale_rl {g.scale_rl.mean():.3f}")
    P(f"## {sp}: per variant (weights vs weights+failures; solver never sees failing inputs)")
    for k, g in f.groupby("variant"): P(f"- {k}: n={len(g)} solver {g.solver.mean():.3f} scale_rl {g.scale_rl.mean():.3f}")
    P("- solver repair on faulty when 3-layer vs 2-layer shown above")
P()
# per-fault table pooled over compiled splits (id, larger, families, faults)
comp = ["test_id", "test_larger", "test_families", "test_faults"]
allf = pd.concat([D[sp][~D[sp].clean].assign(split=sp) for sp in comp])
P("## Pooled compiled splits (id+larger+families+faults), per fault type")
pf = allf.groupby("fault")[["solver", "scale_rl", "baseline_search"]].mean()
pf["n"] = allf.groupby("fault").size()
P(pf.round(3).to_string())
P()
# correlation analysis
P("## Is scale_rl's per-fault profile correlated with the solver's?")
def prof_corr(df, keys, minn=8):
    g = df.groupby(keys)[["solver", "scale_rl", "baseline_search"]].agg("mean"); g["n"] = df.groupby(keys).size(); g = g[g.n >= minn]
    return g
res = {}
g1 = prof_corr(D["test_id"][~D["test_id"].clean], ["fault"], 5)
P(f"- test_id, {len(g1)} fault types: Pearson r(solver, scale_rl) = {pearsonr(g1.solver, g1.scale_rl)[0]:+.2f}, Spearman {spearmanr(g1.solver, g1.scale_rl)[0]:+.2f}; r(search, scale_rl) = {pearsonr(g1.baseline_search, g1.scale_rl)[0]:+.2f}")
g2 = prof_corr(allf, ["fault", "split"], 8)
P(f"- compiled splits, {len(g2)} (fault x split) cells (n>=8): r(solver, scale_rl) = {pearsonr(g2.solver, g2.scale_rl)[0]:+.2f}, Spearman {spearmanr(g2.solver, g2.scale_rl)[0]:+.2f}; r(search, scale_rl) = {pearsonr(g2.baseline_search, g2.scale_rl)[0]:+.2f}")
g3 = prof_corr(allf, ["fault", "family"], 8)
P(f"- compiled splits pooled, {len(g3)} (fault x family) cells (n>=8): r(solver, scale_rl) = {pearsonr(g3.solver, g3.scale_rl)[0]:+.2f}, Spearman {spearmanr(g3.solver, g3.scale_rl)[0]:+.2f}; r(search, scale_rl) = {pearsonr(g3.baseline_search, g3.scale_rl)[0]:+.2f}")
g4 = prof_corr(allf, ["fault", "family", "layers"], 8)
P(f"- (fault x family x depth) cells (n>=8, {len(g4)}): r(solver, scale_rl) = {pearsonr(g4.solver, g4.scale_rl)[0]:+.2f}")
# network level
def phi(a, b):
    a = a.values.astype(bool); b = b.values.astype(bool)
    n11 = (a & b).sum(); n10 = (a & ~b).sum(); n01 = (~a & b).sum(); n00 = (~a & ~b).sum()
    den = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return (n11 * n00 - n10 * n01) / den if den > 0 else np.nan
def phi_strat(df, a, b, key):
    ws, ps = [], []
    for k, g in df.groupby(key):
        p = phi(g[a], g[b])
        if not np.isnan(p): ws.append(len(g)); ps.append(p)
    return float(np.average(ps, weights=ws))
def boot_phi(df, a, b, key=None):
    vals = []
    for _ in range(1000):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        vals.append(phi(s[a], s[b]) if key is None else phi_strat(s, a, b, key))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
for nm, df in [("test_id", D["test_id"][~D["test_id"].clean]), ("compiled pooled", allf)]:
    for other in ["solver", "baseline_search", "scale_rl_greedy"]:
        raw = phi(df.scale_rl, df[other]); cr = boot_phi(df, "scale_rl", other)
        df = df.assign(strat=df.fault + "|" + df.family + "|" + df.layers.astype(str))
        st = phi_strat(df, "scale_rl", other, "strat"); sc = boot_phi(df, "scale_rl", other, "strat")
        P(f"- network-level phi, {nm}, scale_rl vs {other}: raw {raw:+.3f} [{cr[0]:+.3f},{cr[1]:+.3f}], within fault x family x depth strata {st:+.3f} [{sc[0]:+.3f},{sc[1]:+.3f}]")
# conditional success
f = D["test_id"][~D["test_id"].clean]
P(f"- test_id faulty: P(model ok | solver ok) = {f.scale_rl[f.solver==1].mean():.3f} (n={int((f.solver==1).sum())}); P(model ok | solver fails) = {f.scale_rl[f.solver==0].mean():.3f} (n={int((f.solver==0).sum())})")
P(f"- union solver OR model = {((f.solver==1)|(f.scale_rl==1)).mean():.3f}; model-only fails {(f.scale_rl==0).mean():.3f}; of the model's failures solver fixes {f.solver[f.scale_rl==0].mean():.3f}")
# test_faults per fault
f = D["test_faults"][~D["test_faults"].clean]
P("## test_faults per fault type")
for k, g in f.groupby("fault"): P(f"- {k}: n={len(g)} solver {g.solver.mean():.3f} scale_rl {g.scale_rl.mean():.3f}")
f = D["test_families"][~D["test_families"].clean]
P("## test_families per family")
for k, g in f.groupby("family"): P(f"- {k}: n={len(g)} solver {g.solver.mean():.3f} scale_rl {g.scale_rl.mean():.3f}")
open(f"{OUT}/analysis.md", "w").write("\n".join(lines))
# figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(SPL)); w = 0.38
ax[0].bar(x - w/2, [tab[s]["solver"] for s in SPL], w, label="alignment solver (prompt-only)", color="#2a6f97")
ax[0].bar(x + w/2, [tab[s]["model"] for s in SPL], w, label="scale_rl", color="#e07a1f")
ax[0].set_xticks(x); ax[0].set_xticklabels([s.replace("test_", "") for s in SPL]); ax[0].set_ylabel("repair rate (faulty nets)"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1)
cc = g2 if len(g2) else None
ax[1].scatter(g2.solver, g2.scale_rl, s=g2.n * 2, alpha=.6, color="#555")
ax[1].set_xlabel("solver repair rate"); ax[1].set_ylabel("scale_rl repair rate"); ax[1].set_title("(fault x split) cells, compiled splits")
ax[1].plot([0, 1], [0, 1], "k:", lw=.8)
plt.tight_layout(); plt.savefig("discovery/reports/figures/R1C_main.png", dpi=130)
