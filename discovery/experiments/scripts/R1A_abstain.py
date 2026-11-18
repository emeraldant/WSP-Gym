"""R1A step 4: weight-only abstention. Features without forward passes; CV classifier; risk-coverage."""
import pickle, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
OUT = "discovery/experiments/outputs/R1A"; SPL = ["test_id", "test_trained", "test_external"]
FAM = ["dnf", "parity", "threshold", "compare", "minmax", "arith", "argmax", "count", "equality", "mux", "interval"]

def key(x): return json.dumps(x["ops"], sort_keys=True) if x["valid"] else "INVALID"
def loc(x):
    if not x["valid"]: return "INVALID"
    return json.dumps(sorted({(o["tensor"], str(o.get("index", (o.get("i"), o.get("j"))))) for o in x["ops"]}))

def build(mode):
    rows = []
    for si, split in enumerate(SPL):
        C = pickle.load(open(f"{OUT}/cands_{split}.pkl", "rb")); S = pickle.load(open(f"{OUT}/samples_{split}.pkl", "rb"))
        for s in sorted(C):
            d = C[s]; sm = S[s]["samples"]
            keys = [key(x) for x in sm]; cnt = collections.Counter(keys)
            if mode == "plurality":
                top = cnt.most_common(1)[0][0]; a = keys.index(top)
            else: a = 0
            x = sm[a]; ak = keys[a]
            locs = [loc(y) for y in sm]; lc = collections.Counter(locs)
            ops = x["ops"] if x["valid"] else []
            w = d["widths"]
            r = dict(split=si, seed=s, y=int(x["success"]), family=d["family"], fault=d["fault"],
                     depth=d["depth"], w1=w[0], w2=w[1] if len(w) > 1 else 0, wtot=sum(w), n_weights=d["n_weights"], variant=int("failures" in d["variant"]),
                     valid=int(x["valid"]), noop=int(x["noop"]), n_ops=len(ops),
                     n_scale=sum(o["op"] == "scale" for o in ops), n_swap=sum(o["op"] == "swap" for o in ops), n_set=sum(o["op"] == "set" for o in ops), n_add=sum(o["op"] == "add" for o in ops),
                     n_bias=sum(o["tensor"].startswith("b") for o in ops), n_row=sum(o.get("axis") == "row" for o in ops), n_col=sum(o.get("axis") == "col" for o in ops),
                     flip=int(len(ops) == 1 and ops[0]["op"] == "scale" and ops[0].get("factor") == -1),
                     max_layer=max([int(o["tensor"][1:]) for o in ops], default=0), last_layer=int(any(int(o["tensor"][1:]) == d["depth"] + 1 for o in ops)),
                     agree=cnt[ak] / 8, n_distinct=len(cnt), top_share=cnt.most_common(1)[0][1] / 8, is_plural=int(cnt[ak] == cnt.most_common(1)[0][1]),
                     loc_agree=lc[locs[a]] / 8, n_invalid=sum(k == "INVALID" for k in keys) / 8,
                     succ8=int(any(y["success"] for y in sm)), N=len(d["acc"]), r_h=None)
            for f in FAM: r["fam_" + f] = int(d["family"] == f)
            rows.append(r)
    return rows

SETS = {
    "structural": ["depth", "w1", "w2", "wtot", "n_weights", "variant"],
    "answer": ["valid", "noop", "n_ops", "n_scale", "n_swap", "n_set", "n_add", "n_bias", "n_row", "n_col", "flip", "max_layer", "last_layer"],
    "agreement": ["agree", "n_distinct", "top_share", "is_plural", "loc_agree", "n_invalid"],
}
SETS["struct+answer"] = SETS["structural"] + SETS["answer"]
SETS["all_weightonly"] = SETS["structural"] + SETS["answer"] + SETS["agreement"]
SETS["all+family"] = SETS["all_weightonly"] + ["fam_" + f for f in FAM]
SETS["depth_only"] = ["depth"]
SETS["agree_only_logit"] = ["agree"]

def oof(X, y, model, reps=5, seed=0):
    p = np.zeros(len(y))
    for r in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)) if model == "lr" else HistGradientBoostingClassifier(max_iter=80, learning_rate=0.06, max_leaf_nodes=6, min_samples_leaf=15, random_state=0)
            m.fit(X[tr], y[tr]); p[te] += m.predict_proba(X[te])[:, 1]
    return p / reps

def rc_curve(score, fail):
    o = np.argsort(-score, kind="stable"); f = fail[o]; n = len(f)
    cov = np.arange(1, n + 1) / n; risk = np.cumsum(f) / np.arange(1, n + 1)
    return cov, risk
def aurc(score, fail): return rc_curve(score, fail)[1].mean()
def cov_at_risk(score, fail, tgt):
    cov, risk = rc_curve(score, fail); ok = np.flatnonzero(risk <= tgt)
    return cov[ok.max()] if len(ok) else 0.0
def bci(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return np.quantile(v, .025), np.quantile(v, .975)

def run(mode, fh):
    P = lambda *a: (print(*a), print(*a, file=fh))
    rows = build(mode); P(f"\n######## ANSWER = {mode}  (n={len(rows)})")
    sp = np.array([r["split"] for r in rows]); y = np.array([r["y"] for r in rows])
    X = {k: np.array([[r[c] for c in cols] for r in rows], float) for k, cols in SETS.items()}
    res = {}
    # agreement table
    P("\nSuccess rate by agreement (share of the 8 samples identical to the answer):")
    for si, split in enumerate(SPL):
        m = sp == si; a = np.array([r["agree"] for r in rows])[m]; yy = y[m]
        out = []
        for name, lo, hi in [("8/8", 1.0, 1.01), ("6-7/8", .75, .99), ("4-5/8", .5, .74), ("<=3/8", 0, .49)]:
            mm = (a >= lo) & (a < hi); out.append(f"{name}: n={mm.sum()} succ={yy[mm].mean() if mm.sum() else float('nan'):.3f}")
        P(f"  {split}: " + "; ".join(out) + f" | base succ {yy.mean():.3f}")
    # within-depth agreement AUC
    P("\nAUROC of raw agreement (higher agree = success), overall and within depth:")
    for si, split in enumerate(SPL):
        m = sp == si; a = np.array([r["agree"] for r in rows])[m]; d = np.array([r["depth"] for r in rows])[m]; yy = y[m]
        s = f"  {split}: overall {roc_auc_score(yy, a):.3f}"
        for dd in (1, 2):
            mm = d == dd
            if mm.sum() > 20 and len(set(yy[mm])) > 1: s += f" | depth{dd} n={mm.sum()} AUC {roc_auc_score(yy[mm], a[mm]):.3f}"
        # also family-stratified (mean within-family AUC weighted)
        fams = np.array([r["family"] for r in rows])[m]; num = den = 0
        for f in set(fams):
            mm = fams == f
            if mm.sum() >= 15 and 0 < yy[mm].sum() < mm.sum(): num += roc_auc_score(yy[mm], a[mm]) * mm.sum(); den += mm.sum()
        s += f" | family-weighted within-family AUC {num/den:.3f}"
        P(s)
    # classifiers: pooled-split fit (no split indicator) and per-split fit
    for fit in ("per_split", "pooled"):
        P(f"\n=== CV classifiers, fit={fit} (5x repeated stratified 5-fold OOF); AUROC [95% boot CI], AURC (mean risk over coverage; lower=better), cov@risk<=5%/10%")
        for name in ["depth_only", "structural", "answer", "agreement", "struct+answer", "all_weightonly", "all+family"]:
            for model in (["lr", "gbm"]):
                if name == "depth_only" and model == "gbm": continue
                if fit == "pooled":
                    sc = oof(X[name], y, model)
                else:
                    sc = np.zeros(len(y))
                    for si in range(3):
                        m = sp == si; sc[m] = oof(X[name][m], y[m], model)
                for si, split in enumerate(SPL):
                    m = sp == si; yy = y[m]; ss = sc[m]; fail = 1 - yy
                    auc = roc_auc_score(yy, ss); lo, hi = bci(lambda i: roc_auc_score(yy[i], ss[i]) if 0 < yy[i].sum() < len(i) else np.nan, m.sum(), 500)
                    ar = aurc(ss, fail); base = fail.mean()
                    c5 = cov_at_risk(ss, fail, .05); c10 = cov_at_risk(ss, fail, .10)
                    res[(fit, name, model, split)] = (auc, lo, hi, ar, c5, c10, ss)
                    P(f"  {fit:9s} {name:15s} {model:3s} {split:13s} AUC {auc:.3f} [{lo:.3f},{hi:.3f}] AURC {ar:.3f} (base risk {base:.3f}) cov@5% {c5:.2f} cov@10% {c10:.2f}")
    # raw agreement score
    P("\nRaw agreement as the ranking score (no model):")
    for si, split in enumerate(SPL):
        m = sp == si; a = np.array([r["agree"] + 0.001 * r["loc_agree"] for r in rows])[m]; fail = 1 - y[m]
        P(f"  {split}: AUC {roc_auc_score(y[m], a):.3f} AURC {aurc(a, fail):.3f} cov@5% {cov_at_risk(a, fail, .05):.2f} cov@10% {cov_at_risk(a, fail, .10):.2f}  base risk {fail.mean():.3f}")
    return rows, X, y, sp, res

if __name__ == "__main__":
    fh = open(f"{OUT}/abstain.log", "w")
    rows, X, y, sp, res = run("sample0", fh)
    rowsP, XP, yP, spP, resP = run("plurality", fh)
    pickle.dump((rows, res), open(f"{OUT}/abstain_res.pkl", "wb"))
    # ---- figure: risk-coverage
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for si, split in enumerate(SPL):
        m = sp == si; fail = 1 - y[m]; a = ax[si]
        for lab, (name, model, col) in {"depth only": ("depth_only", "lr", "#999999"), "structural (LR)": ("structural", "lr", "#1f77b4"), "answer feats (LR)": ("answer", "lr", "#2ca02c"),
                                        "agreement only (LR)": ("agreement", "lr", "#d62728"), "all weight-only (LR)": ("all_weightonly", "lr", "#9467bd"), "all weight-only (GBM)": ("all_weightonly", "gbm", "#8c564b")}.items():
            ss = res[("per_split", name, model, split)][-1]; c, r = rc_curve(ss, fail); a.plot(c, r, label=lab, color=col, lw=1.6)
        a.axhline(fail.mean(), color="k", ls=":", lw=1, label="no abstention (random order)")
        c, r = rc_curve(-fail + 0.0, fail); a.plot(c, r, color="k", lw=0.8, ls="--", label="oracle")
        a.set_title(f"{split} (n={m.sum()}, base failure {fail.mean():.2f})"); a.set_xlabel("coverage (share accepted without verification)"); a.set_ylabel("risk (failure rate among accepted)")
        a.set_ylim(0, min(0.7, fail.mean() * 1.6 + .05)); a.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig("discovery/reports/figures/R1A_risk_coverage.png", dpi=130)
