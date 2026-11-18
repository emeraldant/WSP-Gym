import json, glob, numpy as np, pandas as pd, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [json.loads(l) for f in glob.glob("discovery/experiments/outputs/R1B/res_test_id_*.jsonl") for l in open(f)]
d = pd.DataFrame(rows)
# need noop/invalid category from rollouts
cat = {}
for s in d.split.unique():
    for l in list(open(f"runs/final/scale_rl/{s}.jsonl"))[1:]:
        r = json.loads(l); cat[(s, r["seed"])] = "noop" if r["noop"] else ("invalid" if not r["valid"] else "edit")
d["cat"] = [cat[(s, k)] for s, k in zip(d.split, d.seed)]
import sys; sys.path.insert(0,"discovery/experiments/scripts")
from R1B_search import units_of_ops
from weightgym.env import load_split
nl = {}
for s_ in d.split.unique():
    for e in load_split(f"data/{s_}.jsonl")[1]: nl[(s_, e.seed)] = e.net.n_layers
um = {}
for s_ in d.split.unique():
    for l in list(open(f"runs/final/scale_rl/{s_}.jsonl"))[1:]:
        r = json.loads(l)
        if r["ops"]: um[(s_, r["seed"])] = units_of_ops(r["ops"], nl[(s_, r["seed"])])
d["hit_M"] = [bool({tuple(u) for u in p} & um.get((s_, k), set())) for p, s_, k in zip(d.planted, d.split, d.seed)]
rng = np.random.default_rng(1)
def boot(x, B=4000):
    x = np.asarray(x, float); n = len(x)
    if n == 0: return (np.nan, np.nan)
    m = x[rng.integers(0, n, (B, n))].mean(1); return tuple(np.percentile(m, [2.5, 97.5]))
def fmt(x): x = np.asarray(x, float); lo, hi = boot(x); return f"{x.mean():.3f} [{lo:.3f},{hi:.3f}]"
splits = ["test_id", "test_trained", "test_faults", "test_larger", "test_families"]
out = []
def P(s=""): print(s); out.append(s)
# primary: all failed faulty nets in split. local M (no-op -> empty set -> fail). random = expected hit prob
P("## Primary (all failed faulty nets; M = units from scale_rl's single answer; noop/invalid count as local fail)")
P("| split | n fail | local M | random equal-size (exp.) | diff (paired) | local only / rand-only discordance: P(local=1,rand~0) |  global any (upper bound) | baseline single-op search |")
for s in splits:
    x = d[d.split == s]
    a = x.succ_M.astype(float); b = x.rand_p_M
    diff = a - b
    P(f"| {s} | {len(x)} | {fmt(a)} | {fmt(b)} | {fmt(diff)} | {(a*(1-b)).mean():.3f} | {fmt(x.global_any.astype(float))} | {fmt(x.base_succ.astype(float))} |")
P("\n## Primary restricted to edit-type failures (non-empty indicated units)")
P("| split | n | local M | random | diff | n units (mean) | pool size local (mean) / global (mean) | ratio | oracle-localised (planted units) | rand for oracle size |")
for s in splits:
    x = d[(d.split == s) & (d.n_M > 0)]
    a = x.succ_M.astype(float); b = x.rand_p_M
    P(f"| {s} | {len(x)} | {fmt(a)} | {fmt(b)} | {fmt(a-b)} | - | {x.n_M.mean():.0f} / {x.N_global.mean():.0f} | {(x.N_global/x.n_M).mean():.1f}x | {fmt(x.succ_O.astype(float))} | {fmt(x.rand_p_O)} |")
P("\n## Failure category split (pooled and per split): local success M / V")
d["catx"] = np.where(d.cat == "edit", np.where(d.n_M > 0, "edit", "edit-no-unit"), d.cat)
for s in splits + ["ALL"]:
    x = d if s == "ALL" else d[d.split == s]
    g = x.groupby("catx").agg(n=("seed", "size"), locM=("succ_M", "mean"), randM=("rand_p_M", "mean"), locV=("succ_V", "mean"), randV=("rand_p_V", "mean"), orac=("succ_O", "mean"))
    P(f"\n{s}\n" + g.round(3).to_string())
P("\n## By fault type (all splits pooled; M)")
g = d.groupby(["fault"]).agg(n=("seed", "size"), frac_edit=("n_M", lambda v: (v > 0).mean()), locM=("succ_M", "mean"), randM=("rand_p_M", "mean"), locV=("succ_V", "mean"), randV=("rand_p_V", "mean"), oracle=("succ_O", "mean"), globalany=("global_any", "mean"))
P(g.round(3).to_string())
g = d.groupby(["split", "fault"]).agg(n=("seed", "size"), locM=("succ_M", "mean"), randM=("rand_p_M", "mean"), locV=("succ_V", "mean"), oracle=("succ_O", "mean"), globalany=("global_any", "mean"))
P("\nby split x fault\n" + g.round(3).to_string())
# does planted unit lie in indicated set
def hit(r, key):
    return None
P("\n## Does the answer touch a planted unit? (edit-type failures) and local success by that")
for s_ in splits:
    x = d[(d.split == s_) & (d.n_M > 0)]
    g = x.groupby("hit_M").agg(n=("seed","size"), locM=("succ_M","mean"), randM=("rand_p_M","mean"))
    P(f"{s_}:\n" + g.round(3).to_string())
x = d[d.n_M > 0]; P("ALL:\n" + x.groupby("hit_M").agg(n=("seed","size"), locM=("succ_M","mean"), randM=("rand_p_M","mean")).round(3).to_string())
P("\n## Evaluations needed")
x = d[d.succ_M]
P(f"local M successes n={len(x)}: pool size mean {x.n_M.mean():.0f} median {x.n_M.median():.0f}; expected evals to first repair (random order) mean {x.exp_evals_M.mean():.1f} median {x.exp_evals_M.median():.1f}")
P(f"matching global pool (same nets): size mean {x.N_global.mean():.0f}, expected evals to first repair in global random order mean {((x.N_global+1)/(x.m_global+1)).mean():.1f} median {((x.N_global+1)/(x.m_global+1)).median():.1f}")
xe = d[d.n_M > 0]
P(f"all edit-type fails: local pool / global pool mean ratio {(xe.n_M/xe.N_global).mean():.3f} (=> {1/(xe.n_M/xe.N_global).mean():.1f}x fewer in size-ratio-of-means terms: {xe.N_global.sum()/xe.n_M.sum():.1f}x)")
P("\n## Variant V (union of 9 samples: main + 8 verify, where available)  [test_id, test_trained only]")
for s in ["test_id", "test_trained"]:
    x = d[d.split == s]
    P(f"{s}: n={len(x)} locV={fmt(x.succ_V.astype(float))} randV={fmt(x.rand_p_V)} diff={fmt(x.succ_V.astype(float)-x.rand_p_V)}; mean pool V {x.n_V.mean():.0f} vs global {x.N_global.mean():.0f}; units nonempty {(x.n_V>0).mean():.2f}")
    y = x[x.catx.isin(["noop", "invalid"])]
    P(f"   noop/invalid-only (n={len(y)}): V nonempty {(y.n_V>0).mean():.2f}, locV {fmt(y.succ_V.astype(float))}, randV {fmt(y.rand_p_V)}")
P("\n## Held-out halves (selection by best acc on half A of inputs; repaired = recovery>=0.95; B = never used for selection)")
def hold(x, name):
    h = [r for r in x[f"hold_{name}"] if isinstance(r, dict)]
    hr = [r for r in x[f"hold_rand_{name}"] if isinstance(r, dict)]
    if not h: return "n/a"
    A = np.array([r["A"] for r in h], float); B = np.array([r["B"] for r in h], float)
    foundB = B[A == 1].mean() if A.sum() else np.nan
    ra = np.mean([r["A"] for r in hr]); rb = np.mean([r["AB"] for r in hr])
    return f"n={len(h)} local: repaired on A {A.mean():.3f}, on A&B {(A*B).mean():.3f}, P(B|A)={foundB:.3f} (n_A={int(A.sum())}); random equal-size: A {ra:.3f}, A&B {rb:.3f}"
for s in splits:
    x = d[(d.split == s)]
    P(f"- {s} M: {hold(x,'M')}")
P(f"- ALL M: {hold(d,'M')}")
P(f"- ALL O (oracle units): {hold(d,'O')}")
P("\n## Greedy depth<=3 secondary (local M/V, nets not repaired at depth 1 with non-empty unit set)")
for name in ["M", "V"]:
    g = d[f"greedy_{name}"].dropna()
    if len(g):
        gl = np.array([r["local"] for r in g], float); gr = np.array([r["rand"] for r in g], float)
        P(f"{name}: n={len(g)} extra repaired by greedy local {gl.mean():.3f} [{boot(gl)[0]:.3f},{boot(gl)[1]:.3f}], random-subset greedy {gr.mean():.3f}")
        for s in splits:
            gg = d[(d.split == s)][f"greedy_{name}"].dropna()
            if len(gg): P(f"   {s}: n={len(gg)} local {np.mean([r['local'] for r in gg]):.3f} rand {np.mean([r['rand'] for r in gg]):.3f}")
P("\n## Which op family repairs (local M successes; count of networks where family has >=1 repairing candidate)")
c = collections.Counter(); 
for r in d[d.succ_M].fam_M:
    for k in r: c[k] += 1
P(str(dict(c)))
P(f"\nsanity: envcheck all true: {all(v for v in d.envcheck_M.dropna())}; inverse_ok all: {d.inverse_ok.all()} (n={len(d)})")
# figure
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sp = splits; xs = np.arange(len(sp))
l = [d[d.split == s].succ_M.mean() for s in sp]; r = [d[d.split == s].rand_p_M.mean() for s in sp]; o = [d[d.split == s].succ_O.mean() for s in sp]; g_ = [d[d.split == s].global_any.mean() for s in sp]
for i, (v, n) in enumerate([(l, "model-guided local"), (r, "random equal-size"), (o, "oracle-localised"), (g_, "global (all)")]): ax[0].bar(xs + (i - 1.5) * 0.2, v, 0.2, label=n)
ax[0].set_xticks(xs); ax[0].set_xticklabels([s.replace("test_", "") for s in sp]); ax[0].set_ylabel("share of scale_rl failures repaired"); ax[0].legend(fontsize=7)
ft = d.groupby("fault").agg(l=("succ_M", "mean"), r=("rand_p_M", "mean"), o=("succ_O", "mean")); ft.plot.bar(ax=ax[1]); ax[1].set_title("by fault type (pooled)"); plt.tight_layout(); plt.savefig("discovery/reports/figures/R1B_local_vs_random.png", dpi=130)
open("discovery/experiments/outputs/R1B/analysis.txt", "w").write("\n".join(out))
d.drop(columns=["planted"]).to_csv("discovery/experiments/outputs/R1B/merged.csv", index=False)
