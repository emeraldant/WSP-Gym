import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ARM = sys.argv[1] if len(sys.argv) > 1 else "sft_rl"
R = "/home/user/wsp-gym/discovery"; OUT = f"{R}/experiments/outputs/R0Q1"
df = pd.read_pickle(f"{OUT}/features_{ARM}.pkl")
SPL = ["test_id", "test_larger", "test_families", "test_faults", "test_trained", "test_external"]
rng = np.random.default_rng(0)
log = open(f"{OUT}/analysis_{ARM}.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-h, c+h
df["log_nw"] = np.log(df.n_weights)
df["twin_hi"] = (df.twin > 0.99).astype(float); df.loc[df.twin.isna(), "twin_hi"] = np.nan
NUM = ["log_nw", "depth", "excess", "acc_drop", "f_relpos", "f_is_last_layer", "n_out", "n_in", "twin_hi", "act", "outnorm"]
CAT = ["family", "fault", "variant"]

def design(d, cols_num, cols_cat, ref=None):
    X = d[cols_num].copy()
    for c in cols_num:
        X[c + "_na"] = X[c].isna().astype(float) if X[c].isna().any() else 0.0
        X[c] = X[c].fillna(X[c].mean())
    X = X.loc[:, X.nunique() > 1]
    X = (X - X.mean()) / X.std().replace(0, 1)
    for c in cols_cat:
        dm = pd.get_dummies(d[c], prefix=c).astype(float)
        X = pd.concat([X, dm - dm.mean()], axis=1)  # centred dummies: coef = deviation from average
    return X

def fit_cv(X, y, seed=0):
    # CV over C via inner LogisticRegressionCV; outer repeated CV AUC
    aucs = []
    for rep in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        pr = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            m = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=4, penalty="l2", max_iter=2000, scoring="neg_log_loss").fit(X.iloc[tr], y[tr])
            pr[te] = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, pr))
    return float(np.mean(aucs)), float(np.std(aucs))

P(f"# ARM = {ARM}")
# ---------- univariate
def bins(d):
    d = d.copy()
    d["nw_bin"] = pd.qcut(d.n_weights, 3, duplicates="drop").astype(str)
    d["drop_bin"] = pd.qcut(d.acc_drop, 3, duplicates="drop").astype(str)
    d["excess_bin"] = pd.qcut(d.excess, 3, duplicates="drop").astype(str)
    d["depth_b"] = d.depth.astype(str); d["layer_b"] = d.f_layer.astype("Int64").astype(str)
    d["relpos_bin"] = pd.cut(d.f_relpos, [-.01, .34, .67, 1.0]).astype(str)
    d["twin_b"] = d.twin_hi.map({0.0: "no twin", 1.0: "has twin"}).fillna("n/a")
    d["mag_b"] = d.mag_logfactor.round(2).astype(str)
    return d
dfb = pd.concat([bins(df[df.split == s]) for s in SPL])
uni_rows = []
for s in SPL:
    d = dfb[dfb.split == s]
    for f in ["family", "fault", "variant", "depth_b", "nw_bin", "excess_bin", "drop_bin", "layer_b", "relpos_bin", "twin_b"]:
        for lv, g in d.groupby(f):
            k, n = int(g.fail_model.sum()), len(g); lo, hi = wilson(k, n)
            uni_rows.append(dict(split=s, feature=f, level=lv, n=n, fail_rate=k/n, lo=lo, hi=hi))
        ct = pd.crosstab(d[f], d.fail_model)
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi = stats.chi2_contingency(ct)[1]
            uni_rows.append(dict(split=s, feature=f, level="__chi2_p", n=len(d), fail_rate=chi, lo=np.nan, hi=np.nan))
uni = pd.DataFrame(uni_rows); uni.to_csv(f"{OUT}/univariate_{ARM}.csv", index=False)
P("\n## Overall failure rate on faulty nets (Wilson CI)")
for s in SPL:
    d = df[df.split == s]; k, n = d.fail_model.sum(), len(d); lo, hi = wilson(k, n)
    P(f"{s}: n={n} fail={k/n:.3f} [{lo:.3f},{hi:.3f}] search_fail={d.fail_search.mean():.3f} grad_fail={d.fail_grad.mean():.3f}")
P("\n## Univariate: chi2 p-values (feature heterogeneity of failure rate)")
pv = uni[uni.level == "__chi2_p"].pivot(index="feature", columns="split", values="fail_rate")[SPL]; P(pv.round(4).to_string())
for f in ["fault", "family", "variant"]:
    P(f"\n## failure rate by {f}"); t = uni[(uni.feature == f) & (uni.level != "__chi2_p")]
    t = t.assign(txt=t.apply(lambda r: f"{r.fail_rate:.2f} [{r.lo:.2f},{r.hi:.2f}] n={r.n}", axis=1)).pivot(index="level", columns="split", values="txt")
    P(t[[c for c in SPL if c in t.columns]].to_string())
for f in ["depth_b", "nw_bin", "excess_bin", "drop_bin", "layer_b", "relpos_bin", "twin_b"]:
    P(f"\n## failure rate by {f}"); t = uni[(uni.feature == f) & (uni.level != "__chi2_p")]
    t = t.assign(txt=t.apply(lambda r: f"{r.fail_rate:.2f} n={r.n}", axis=1)).pivot(index="level", columns="split", values="txt")
    P(t[[c for c in SPL if c in t.columns]].to_string())

# ---------- logistic
def boot_coefs(X, y, C, B=200):
    co = []
    for b in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        m = LogisticRegression(C=C, max_iter=2000).fit(X.iloc[i], y[i]); co.append(m.coef_[0])
    return np.array(co)
P("\n## Regularised logistic regression (L2, C by CV; standardised numerics, centred dummies)")
coef_all = {}
for s in SPL + ["POOLED"]:
    d = df if s == "POOLED" else df[df.split == s]
    d = d.reset_index(drop=True); y = d.fail_model.values
    cat = CAT + (["split"] if s == "POOLED" else [])
    X = design(d, NUM, cat)
    auc, sd = fit_cv(X, y)
    m = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, max_iter=2000, scoring="neg_log_loss").fit(X, y); C = float(m.C_[0])
    cs = boot_coefs(X, y, C, 150)
    tab = pd.DataFrame(dict(coef=m.coef_[0], lo=np.percentile(cs, 2.5, axis=0), hi=np.percentile(cs, 97.5, axis=0)), index=X.columns)
    tab["sig"] = (tab.lo > 0) | (tab.hi < 0); coef_all[s] = tab
    # group ablation
    ab = {}
    for g, cols in {"family": ["family"], "fault": ["fault"], "variant": ["variant"], "size(n_w,depth,excess,n_in,n_out)": None, "acc_drop": None, "position": None, "twin/act/outnorm": None}.items():
        if g in CAT: Xg = design(d, NUM, [c for c in cat if c != g])
        else:
            drop = {"size(n_w,depth,excess,n_in,n_out)": ["log_nw", "depth", "excess", "n_in", "n_out"], "acc_drop": ["acc_drop"], "position": ["f_relpos", "f_is_last_layer"], "twin/act/outnorm": ["twin_hi", "act", "outnorm"]}[g]
            Xg = design(d, [c for c in NUM if c not in drop], cat)
        a2, _ = fit_cv(Xg, y); ab[g] = a2
    P(f"\n### {s}: n={len(d)}, fail={y.mean():.3f}, C={C:.3g}, repeated-5CV AUC={auc:.3f}+-{sd:.3f}; AUC w/o group: " + ", ".join(f"-{g}: {v:.3f}" for g, v in ab.items()))
    t2 = tab.reindex(tab.coef.abs().sort_values(ascending=False).index).head(10)
    P(t2.round(2).to_string())
pd.concat(coef_all, names=["split"]).to_csv(f"{OUT}/logit_coefs_{ARM}.csv")
# within-fault magnitude analyses
P("\n## Magnitude within fault (Spearman of magnitude with failure; pooled across splits)")
for f, col in [("scale_out_col", "mag_logfactor"), ("scale_in_row", "mag_logfactor"), ("bias_shift", "mag_delta")]:
    d = df[(df.fault == f)]
    if len(d) > 10:
        r, p = stats.spearmanr(d[col], d.fail_model); P(f"{f} {col}: n={len(d)} rho={r:.3f} p={p:.3g}")
for f in ["scale_out_col", "scale_in_row"]:
    d = df[df.fault == f]
    if len(d): P(f, d.groupby(d.mag_logfactor.round(2)).fail_model.agg(["mean", "size"]).round(2).to_dict())
d = df[df.fault == "weight_corrupt"]; P("weight_corrupt by mode", d.groupby("wc_mode").fail_model.agg(["mean", "size"]).round(2).to_dict())
P("acc_drop & failure spearman by split:", {s: round(stats.spearmanr(df[df.split == s].acc_drop, df[df.split == s].fail_model)[0], 3) for s in SPL})
P("n_weights & failure spearman by split:", {s: round(stats.spearmanr(df[df.split == s].n_weights, df[df.split == s].fail_model)[0], 3) for s in SPL})
P("excess & failure spearman by split:", {s: round(stats.spearmanr(df[df.split == s].excess, df[df.split == s].fail_model)[0], 3) for s in SPL})

# ---------- overlap with baselines
def boot_ci(f, d, B=1000):
    v = []
    n = len(d)
    for _ in range(B):
        v.append(f(d.iloc[rng.integers(0, n, n)]))
    return np.nanpercentile(v, [2.5, 97.5])
def kap(d, a, b):
    if d[a].nunique() < 2 or d[b].nunique() < 2: return np.nan
    return cohen_kappa_score(d[a], d[b])
P("\n## Overlap of model failures with baseline failures (faulty nets, same seeds)")
ov = []
for other, oc in [("search", "fail_search"), ("gradient", "fail_grad")]:
    for s in SPL + ["POOLED"]:
        d = df if s == "POOLED" else df[df.split == s]
        a = d.fail_model; b = d[oc]
        both = int(((a == 1) & (b == 1)).sum()); onlyM = int(((a == 1) & (b == 0)).sum()); onlyB = int(((a == 0) & (b == 1)).sum()); neither = int(((a == 0) & (b == 0)).sum())
        k = kap(d, "fail_model", oc); kci = boot_ci(lambda x: kap(x, "fail_model", oc), d, 500)
        union = 1 - neither / len(d) * 0 - both / len(d)  # fixed by model OR other = 1 - both fail
        fixM = 1 - a.mean(); fixB = 1 - b.mean()
        p_b_given_m = b[a == 1].mean(); p_b_given_nm = b[a == 0].mean()
        ci_u = boot_ci(lambda x: 1 - ((x.fail_model == 1) & (x[oc] == 1)).mean(), d, 500)
        # expected union under independence
        ind = 1 - a.mean() * b.mean()
        ov.append(dict(other=other, split=s, n=len(d), model_fix=fixM, other_fix=fixB, both_fail=both, only_model_fails=onlyM, only_other_fails=onlyB, both_fix=neither,
                       P_other_fail_given_model_fail=p_b_given_m, P_other_fail_given_model_fix=p_b_given_nm, kappa=k, kappa_lo=kci[0], kappa_hi=kci[1],
                       union_fix=union, union_lo=ci_u[0], union_hi=ci_u[1], union_if_independent=ind))
ov = pd.DataFrame(ov); ov.to_csv(f"{OUT}/overlap_{ARM}.csv", index=False); P(ov.round(3).to_string())

# ---------- failure taxonomy
P("\n## What the model answers when it fails (faulty nets)")
fl = df[df.fail_model == 1]
tx = pd.crosstab(fl.split, fl.fail_cat).reindex(SPL); P(tx.to_string()); P((tx.div(tx.sum(1), axis=0)).round(3).to_string())
fl2 = fl.assign(cat=fl.fail_cat)
P("\n by fault (pooled over splits) share of each category among failures:"); P(pd.crosstab(fl2.fault, fl2.cat, normalize="index").round(2).to_string()); P(pd.crosstab(fl2.fault, fl2.cat).to_string())
P("\nOf ALL faulty nets per split: noop rate, right-unit rate (among failures + successes)")
P(df.groupby("split").apply(lambda g: pd.Series(dict(noop=g.noop.mean(), invalid=(~g.valid).mean(), n_ops_mean=g.n_ops.mean()))).reindex(SPL).round(3).to_string())
P("recovery among failed valid non-noop answers (median, share <0 = harmful):")
P(fl[(fl.fail_cat != "noop") & (fl.fail_cat != "invalid")].groupby("split").rec_model.agg(["median", "size", lambda x: (x < 0).mean(), lambda x: ((x > 0.5)).mean()]).reindex(SPL).round(3).to_string())
P("\nfail category by inverse op types:"); P(pd.crosstab(fl.inv_op_types, fl.fail_cat).to_string())
P("\nanswer op types among failures:"); P(pd.crosstab(fl.inv_op_types, fl.ans_op_types).to_string())
fl.groupby(["split","fault"]).fail_cat.value_counts().unstack().fillna(0).astype(int).to_csv(f"{OUT}/taxonomy_{ARM}.csv")

# ---------- figures
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
t = df.pivot_table(index="fault", columns="split", values="fail_model", aggfunc="mean")[SPL]
im = ax[0].imshow(t.values, cmap="viridis_r", vmin=0, vmax=1, aspect="auto"); ax[0].set_xticks(range(6)); ax[0].set_xticklabels([s[5:] for s in SPL], rotation=45); ax[0].set_yticks(range(len(t))); ax[0].set_yticklabels(t.index)
for i in range(t.shape[0]):
    for j in range(t.shape[1]):
        if not np.isnan(t.values[i, j]): ax[0].text(j, i, f"{t.values[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
ax[0].set_title(f"{ARM}: failure rate by fault type")
t = df.pivot_table(index="family", columns="split", values="fail_model", aggfunc="mean")[SPL]
ax[1].imshow(t.values, cmap="viridis_r", vmin=0, vmax=1, aspect="auto"); ax[1].set_xticks(range(6)); ax[1].set_xticklabels([s[5:] for s in SPL], rotation=45); ax[1].set_yticks(range(len(t))); ax[1].set_yticklabels(t.index)
for i in range(t.shape[0]):
    for j in range(t.shape[1]):
        if not np.isnan(t.values[i, j]): ax[1].text(j, i, f"{t.values[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
ax[1].set_title("failure rate by family")
tab = coef_all["POOLED"]; tab = tab[~tab.index.str.startswith(("family_", "fault_", "split_"))]
y = np.arange(len(tab)); ax[2].errorbar(tab.coef, y, xerr=[tab.coef - tab.lo, tab.hi - tab.coef], fmt="o"); ax[2].axvline(0, c="k", lw=.5); ax[2].set_yticks(y); ax[2].set_yticklabels(tab.index, fontsize=8); ax[2].set_title("pooled logit: numeric/variant coefs (std)")
plt.tight_layout(); plt.savefig(f"{R}/reports/figures/R0Q1_{ARM}_failure_anatomy.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(9, 4))
o = ov[(ov.split != "POOLED")]; w = 0.35
for i, (oth, c) in enumerate([("search", "C0"), ("gradient", "C1")]):
    q = o[o.other == oth]; ax.bar(np.arange(6) + i*w, q.kappa, w, yerr=[q.kappa - q.kappa_lo, q.kappa_hi - q.kappa], label=f"kappa(model fail, {oth} fail)", color=c)
ax.set_xticks(np.arange(6) + w/2); ax.set_xticklabels([s[5:] for s in SPL]); ax.axhline(0, c="k", lw=.5); ax.legend(); ax.set_title(f"{ARM}: failure agreement with baselines")
plt.tight_layout(); plt.savefig(f"{R}/reports/figures/R0Q1_{ARM}_overlap.png", dpi=110); plt.close()
