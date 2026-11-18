import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "discovery/experiments/scripts")
OUT = "discovery/experiments/outputs/R2B"
rng = np.random.default_rng(1)
# ---------------- (a) ----------------
a = pd.read_csv(f"{OUT}/a_results_all.csv")
S = list(a.setting.unique())
a["faulty"] = a.fault != "clean"
piv_s = a.pivot(index="seed", columns="setting", values="success").astype(float)
fault = a.pivot(index="seed", columns="setting", values="faulty").astype(bool)
same_fault = a.pivot(index="seed", columns="setting", values="fault")
isf = fault["default"].values
assert all((fault[s].values == isf).all() for s in S)
seeds = piv_s.index.values
def stat(s, mask):
    return piv_s[s].values[mask].mean()
B = 2000; idx = rng.integers(0, len(seeds), size=(B, len(seeds)))
lines = []
res_a = []
for s in S:
    f = piv_s[s].values[isf]; c = piv_s[s].values[~isf]
    # bootstrap ci for level (over episodes)
    bf = [piv_s[s].values[isf][rng.integers(0, isf.sum(), isf.sum())].mean() for _ in range(B)]
    row = dict(setting=s, n_faulty=int(isf.sum()), repair=f.mean(), repair_lo=np.percentile(bf, 2.5), repair_hi=np.percentile(bf, 97.5),
              n_clean=int((~isf).sum()), clean_acc=c.mean(), overall=piv_s[s].values.mean())
    if s != "default":
        d = piv_s[s].values - piv_s["default"].values
        bd = []; bo = []
        for b in range(B):
            ii = idx[b]; m = isf[ii]
            bd.append(d[ii][m].mean()); bo.append(d[ii].mean())
        row.update(d_repair=d[isf].mean(), d_lo=np.percentile(bd, 2.5), d_hi=np.percentile(bd, 97.5),
                   d_overall=d.mean(), do_lo=np.percentile(bo, 2.5), do_hi=np.percentile(bo, 97.5),
                   d_clean=d[~isf].mean(),
                   # discordant counts among faulty
                   lost=int(((d < 0) & isf).sum()), gained=int(((d > 0) & isf).sum()),
                   same_fault_kind=float((same_fault[s].values[isf] == same_fault["default"].values[isf]).mean()))
    res_a.append(row)
RA = pd.DataFrame(res_a); RA.to_csv(f"{OUT}/a_summary.csv", index=False)
print(RA.round(3).to_string())
# by family / fault for scale and noise (faulty only)
for s in ("scale_0.1_10", "noise_0.03", "combined"):
    d = a[(a.setting == s) & a.faulty]; d0 = a[(a.setting == "default") & a.faulty]
    print("\n", s, "repair by family (default -> setting)")
    print(pd.concat([d0.groupby("family").success.mean().rename("default"), d.groupby("family").success.mean().rename(s), d.groupby("family").size().rename("n")], axis=1).round(3).to_string())
    print("by fault"); print(pd.concat([d0.groupby("fault").success.mean().rename("default"), d.groupby("fault").success.mean().rename(s)], axis=1).round(3).to_string())
    print("by size"); print(d.groupby("size").success.mean().round(3).to_string(), "errs", int(d.err.sum()))
# manipulation check: realised # splits per setting (200 eps each)
from weightgym import nets
from weightgym.env import GymConfig, make_episode
from weightgym.splits import config_for
import R2B_a_run as R
cnt = {}
orig = nets.split_neuron
def counting(net, l, j, alpha):
    cnt["n"] += 1; cnt["alphas"].append(alpha); return orig(net, l, j, alpha)
nets.split_neuron = counting
print("\nmanipulation check (200 eps per setting): mean splits / mean n_weights")
for s in ("default", "psplit_1.0", "psplit_2.0"):
    cnt.update(n=0, alphas=[]); nw = []
    for i in range(200):
        d = config_for("test_id", 0).to_json(); d["size"] = i % 2; d.update(R.SETTINGS[s])
        nw.append(make_episode(R.SEED0 + i, GymConfig.from_json(d)).net.n_weights)
    print(s, cnt["n"] / 200, np.mean(nw), "alpha |a-.5| mean", np.mean(np.abs(np.array(cnt["alphas"]) - .5)))
nets.split_neuron = orig

# ---------------- (b) ----------------
b = pd.read_csv(f"{OUT}/b_replay.csv")
def load_rl(sp):
    rows = [json.loads(l) for l in open(f"runs/final/scale_rl/{sp}.jsonl")][1:]
    d = pd.DataFrame(rows)[["seed", "success"]]; return d
def logit_fit(X, y, ridge=1e-4, pen=None, iters=100):
    n, p = X.shape; beta = np.zeros(p)
    P = np.eye(p) * ridge
    if pen is not None: P = np.diag(pen)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu) + 1e-9
        g = X.T @ (y - mu) - P @ beta; H = X.T @ (X * W[:, None]) + P
        step = np.linalg.solve(H, g); beta += step
        if np.abs(step).max() < 1e-9: break
    eta = np.clip(X @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu) + 1e-9
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + P)
    ll = (y * np.log(mu + 1e-12) + (1 - y) * np.log(1 - mu + 1e-12)).sum()
    return beta, cov, ll
def design(d, factors, ctrl_mode):
    cols = {"const": np.ones(len(d))}; names = []
    # controls
    cols["depth3"] = (d.depth == 3).astype(float).values
    cols["size1"] = (d["size"] == 1).astype(float).values
    cols["variant_wf"] = (d.variant == "weights+failures").astype(float).values
    if ctrl_mode == "full":
        for f in sorted(d.fault.unique())[1:]: cols[f"fault={f}"] = (d.fault == f).astype(float).values
        for f in sorted(d.family.unique())[1:]: cols[f"family={f}"] = (d.family == f).astype(float).values
    elif ctrl_mode == "light":
        for f in sorted(d.fault.unique())[1:]: cols[f"fault={f}"] = (d.fault == f).astype(float).values
    nctrl = len(cols)
    for f in factors:
        v = d[f].astype(float).values; cols[f] = (v - v.mean()) / (v.std() + 1e-12)
    return np.column_stack(list(cols.values())), list(cols.keys()), nctrl
def regress(d, y, factors, ctrl_mode, label):
    X, names, nc = design(d, factors, ctrl_mode)
    beta, cov, ll = logit_fit(X, y)
    X0 = X[:, :nc]; b0, c0, ll0 = logit_fit(X0, y)
    from scipy.stats import chi2, norm
    lr = 2 * (ll - ll0); pv = chi2.sf(lr, len(factors))
    se = np.sqrt(np.diag(cov)); out = []
    print(f"\n[{label}] controls={ctrl_mode} n={len(y)} failures={int(y.sum())} params={X.shape[1]}  LR test of factor block: chi2({len(factors)})={lr:.2f} p={pv:.3g}")
    for i in range(nc, len(names)):
        z = beta[i] / se[i]
        r = dict(model=label, controls=ctrl_mode, factor=names[i], OR_per_SD=np.exp(beta[i]), lo=np.exp(beta[i] - 1.96 * se[i]), hi=np.exp(beta[i] + 1.96 * se[i]), p=2 * norm.sf(abs(z)),
                 n=len(y), n_fail=int(y.sum()), LR_block_p=pv)
        out.append(r); print("  %-18s OR/SD %.2f [%.2f, %.2f] p=%.3g" % (names[i], r["OR_per_SD"], r["lo"], r["hi"], r["p"]))
    return out
allres = []
rl = load_rl("test_id").rename(columns={"success": "rl_success"})
sv = pd.read_csv("discovery/experiments/outputs/R1C/solver_test_id.csv")[["seed", "prompt_success"]]
bi = b[b.split == "test_id"].merge(rl, on="seed").merge(sv, on="seed")
bi["noise_retry"] = (bi.noise_attempt > 0).astype(int)
bi["fail_rl"] = (~bi.rl_success.astype(bool)).astype(float); bi["fail_solver"] = (~bi.prompt_success.astype(bool)).astype(float)
bf = bi[bi.fault != "clean"].reset_index(drop=True)
F1 = ["logc_sd", "n_split", "n_dead", "n_cancel", "noise_rms", "disp"]
print("\ncorr among factors (test_id faulty):"); print(bf[F1 + ["noise_retry"]].corr().round(2).to_string())
print("univariate failure rates: scale_rl %.3f solver %.3f" % (bf.fail_rl.mean(), bf.fail_solver.mean()))
for mode in ("full", "light"):
    allres += regress(bf, bf.fail_rl.values, F1, mode, "scale_rl test_id faulty")
    allres += regress(bf, bf.fail_solver.values, F1, mode, "R1C solver test_id faulty")
# also all episodes incl clean (failure = wrong answer on clean too), full controls w/o fault dummies for clean handled by fault=clean level
ba = bi.copy()
allres += regress(ba, ba.fail_rl.values, F1, "full", "scale_rl test_id ALL (incl clean)")
allres += regress(ba, ba.fail_solver.values, F1, "full", "R1C solver test_id ALL (incl clean)")
# one-factor-at-a-time with full controls
for f in F1:
    allres += regress(bf, bf.fail_rl.values, [f], "full", "scale_rl test_id faulty single:" + f)
    allres += regress(bf, bf.fail_solver.values, [f], "full", "R1C solver test_id faulty single:" + f)
# quartile descriptives
for f in ("logc_sd", "n_split", "n_dead", "n_cancel", "noise_rms", "disp"):
    q = pd.qcut(bf[f], 4, duplicates="drop")
    print("\nfailure by quartile of", f); print(bf.groupby(q, observed=True)[["fail_rl", "fail_solver"]].agg(["mean", "size"]).round(3).to_string())
# trained
rt = load_rl("test_trained").rename(columns={"success": "rl_success"})
bt = b[b.split == "test_trained"].merge(rt, on="seed"); bt["fail_rl"] = (~bt.rl_success.astype(bool)).astype(float)
bt["restart_flag"] = (bt.train_restart > 0).astype(int)
btf = bt[bt.fault != "clean"].reset_index(drop=True)
F2 = ["extras_total", "restart_flag", "rownorm_logsd", "w_absmax"]
print("\ntest_trained failure rate faulty %.3f" % btf.fail_rl.mean())
for mode in ("full", "light"):
    allres += regress(btf, btf.fail_rl.values, F2, mode, "scale_rl test_trained faulty")
pd.DataFrame(allres).to_csv(f"{OUT}/b_regressions.csv", index=False)
bf.to_csv(f"{OUT}/b_test_id_joined.csv", index=False); btf.to_csv(f"{OUT}/b_test_trained_joined.csv", index=False)
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
R_ = RA.set_index("setting")
order = ["default", "scale_0.1_10", "noise_0.03", "psplit_2.0", "psplit_1.0", "psplit_2.0_extremealpha", "combined"]
x = np.arange(len(order))
ax[0].bar(x - .2, R_.loc[order, "repair"], .4, yerr=[R_.loc[order, "repair"] - R_.loc[order, "repair_lo"], R_.loc[order, "repair_hi"] - R_.loc[order, "repair"]], label="repair (faulty)")
ax[0].bar(x + .2, R_.loc[order, "clean_acc"], .4, label="clean acc"); ax[0].set_xticks(x); ax[0].set_xticklabels(order, rotation=40, ha="right", fontsize=8); ax[0].legend(); ax[0].set_title("R1C solver, fresh seeds")
r = pd.DataFrame(allres); r = r[(r.model.isin(["scale_rl test_id faulty", "R1C solver test_id faulty"])) & (r.controls == "full")]
for k, (m, g) in enumerate(r.groupby("model")):
    yy = np.arange(len(g)) + (k - .5) * .3
    ax[1].errorbar(g.OR_per_SD, yy, xerr=[g.OR_per_SD - g.lo, g.hi - g.OR_per_SD], fmt="o", label=m)
ax[1].set_yticks(np.arange(len(g))); ax[1].set_yticklabels(g.factor); ax[1].axvline(1, color="k", lw=.5); ax[1].set_xscale("log"); ax[1].legend(fontsize=7); ax[1].set_title("failure OR per SD (full controls)")
plt.tight_layout(); plt.savefig("discovery/reports/figures/R2B_summary.png", dpi=130)
