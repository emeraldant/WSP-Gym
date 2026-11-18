"""R3C / H10: benign-repair neuron localiser -> zero-shot backdoor localisation / detection."""
import sys, glob, json, pickle, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "discovery/experiments/scripts")
import R3C_feat as F
from weightgym.env import load_split, score_answer, Episode
from weightgym.audit_baselines import hidden_stats, _robust_z, zero_col, weight_rule_policy, run_all
OUT = "discovery/experiments/outputs/R3C"
rng = np.random.default_rng(0)

# ---------------- data
def load_train(src=None):
    rows = []
    for f in sorted(glob.glob(f"{OUT}/train_*.pkl")):
        rows += [r for r in pickle.load(open(f, "rb")) if src is None or r["src"] == src]
    return rows

def stack(rows, key="lab"):
    return np.vstack([r["X"] for r in rows]), np.concatenate([r[key] for r in rows])

def fit(X, y, ncol=None, seed=0):
    ncol = ncol or X.shape[1]
    m = HGB(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
    m.fit(X[:, :ncol], y)
    return m

def score_nets(model, tabs, ncol):
    return [model.predict_proba(t["X"][:, :ncol])[:, 1] for t in tabs]

def tables(eps):
    return [F.net_neuron_table(e.net, e.task) for e in eps]

def det_labels(ep):
    """compiled audit: detector neuron idx (hidden layer 0)."""
    return ep.fault_info["neuron"] if not ep.is_clean else None

_recov_cache = {}
def col_recovery(ep, j):
    k = (ep.seed, j)
    if k not in _recov_cache:
        r = score_answer(ep, json.dumps([zero_col(j)]))
        _recov_cache[k] = bool(r["success"])
    return _recov_cache[k]

def success_matrix(ep):
    return np.array([col_recovery(ep, j) for j in range(ep.net.Ws[0].shape[0])])

# ---------------- metrics
def boot(vals_fn, n, B=1000, seed=1):
    r = np.random.default_rng(seed)
    return np.array([vals_fn(r.integers(0, n, n)) for _ in range(B)])

def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

def auroc(bd_scores, cl_scores):
    y = np.r_[np.ones(len(bd_scores)), np.zeros(len(cl_scores))]
    return roc_auc_score(y, np.r_[bd_scores, cl_scores])

def stat_scores(ep):
    """stat_outlier internals: per-neuron max robust |z| over STAT_FEATURES among live neurons (0 for dead)."""
    st = hidden_stats(ep.net)
    live = st["cn"] > 0.1
    s = np.zeros(len(live))
    if live.sum() < 4: return s
    idx = np.flatnonzero(live)
    for name in ("gain", "bias", "col", "fanin"):
        z = np.abs(_robust_z(st[name][live]))
        s[idx] = np.maximum(s[idx], z)
    return s

def rule_scores(ep):
    net = ep.net
    W, b, Wn = net.Ws[0], net.bs[0], net.Ws[1]
    s = np.zeros(W.shape[0])
    for j in range(W.shape[0]):
        mag = np.abs(W[j]); strong = mag > 0.25 * max(mag.max(), 1e-9)
        peak = float(np.maximum(W[j], 0).sum() + b[j])
        if strong.sum() < 5 or peak <= 0 or peak - float(mag[strong].min()) > 0: continue
        s[j] = peak * float(np.abs(Wn[:, j]).max())
    return s

def evaluate(name, eps, tabs, score_lists, kind):
    """score_lists: dict method-> list of per-neuron score arrays. kind 'compiled'|'sgd'.
    Returns dict of metrics with CIs. top-1 hit: compiled = argmax == detector idx; also recovery-hit (zero col removes backdoor)
    sgd = recovery-hit only."""
    bd = [i for i, e in enumerate(eps) if not e.is_clean]
    cl = [i for i, e in enumerate(eps) if e.is_clean]
    succ = {i: success_matrix(eps[i]) for i in bd}
    out = {}
    chance_id = np.array([1.0 / len(score_lists[next(iter(score_lists))][i]) for i in bd])
    chance_rec = np.array([succ[i].mean() for i in bd])
    anyfix = np.array([succ[i].any() for i in bd], float)
    for m, sc in score_lists.items():
        top = np.array([int(np.argmax(sc[i])) for i in bd])
        hit_rec = np.array([succ[i][t] for i, t in zip(bd, top)], float)
        hit_id = np.array([t == det_labels(eps[i]) for i, t in zip(bd, top)], float) if kind == "compiled" else None
        # top-3 recovery
        top3 = np.array([succ[i][np.argsort(-sc[i])[:3]].any() for i in bd], float)
        mx = np.array([sc[i].max() for i in range(len(eps))])
        au = auroc(mx[bd], mx[cl])
        bi = boot(lambda ix: auroc(mx[np.array(bd)[ix]], mx[np.array(cl)[rng.integers(0, len(cl), len(cl))]]), len(bd), B=500)
        r = dict(n_bd=len(bd), n_cl=len(cl), top1_rec=hit_rec.mean(), top1_rec_ci=ci(boot(lambda ix: hit_rec[ix].mean(), len(bd))),
                 top3_rec=top3.mean(), auroc=au, auroc_ci=ci(bi), chance_rec=chance_rec.mean(), anyfix=anyfix.mean())
        if hit_id is not None:
            r.update(top1_id=hit_id.mean(), top1_id_ci=ci(boot(lambda ix: hit_id[ix].mean(), len(bd))), chance_id=chance_id.mean())
        out[m] = (r, hit_rec, hit_id)
    return out, bd, cl

if __name__ == "__main__":
    tr = load_train()
    print("train nets", len(tr), "neurons", sum(len(r["lab"]) for r in tr), "pos", sum(r["lab"].sum() for r in tr))
    # repair-domain sanity: held-out nets by seed hash
    bysrc = {s: [r for r in tr if r["src"] == s] for s in ("compiled", "trained")}
    NB, NA = F.N_BASE, len(F.NAMES)
    res = {}
    # in-domain sanity (80/20 by net)
    for s in bysrc:
        rs = bysrc[s]; k = int(0.8 * len(rs)); a, b = rs[:k], rs[k:]
        X, y = stack(a); m = fit(X, y, NB)
        top1 = np.mean([r["lab"][np.argmax(m.predict_proba(r["X"][:, :NB])[:, 1])] for r in b])
        ch = np.mean([r["lab"].mean() for r in b])
        print(f"repair held-out ({s}) top1 neuron hit {top1:.3f} chance {ch:.3f} n={len(b)}")
        res[f"repair_heldout_{s}"] = dict(top1=float(top1), chance=float(ch), n=len(b))
    # main models
    X, y = stack(tr)
    models = {"R2A-adapted (primary)": (fit(X, y, NB), NB), "+extra feats (secondary)": (fit(X, y, NA), NA),
              "R2A raw only (23)": (fit(X, y, len(F.BASE)), len(F.BASE))}
    for s in bysrc:
        Xs, ys = stack(bysrc[s]); models[f"primary feats, {s}-source only"] = (fit(Xs, ys, NB), NB)
    pickle.dump({k: v for k, v in models.items()}, open(f"{OUT}/models.pkl", "wb"))

    splits = {"test_audit": "compiled", "test_audit_sgd": "sgd"}
    ev = {}
    for sp, kind in splits.items():
        _, eps = load_split(f"data/{sp}.jsonl")
        tabs = tables(eps)
        S = {k: score_nets(m, tabs, nc) for k, (m, nc) in models.items()}
        S["stat_outlier (z score)"] = [stat_scores(e) for e in eps]
        S["weight_rule (effect)"] = [rule_scores(e) for e in eps]
        S["random"] = [rng.random(len(t["idx"])) for t in tabs]
        pickle.dump((eps, tabs, S), open(f"{OUT}/eval_{sp}.pkl", "wb"))
        out, bd, cl = evaluate(sp, eps, tabs, S, kind)
        ev[sp] = {k: v[0] for k, v in out.items()}
        # paired difference primary vs stat_outlier (top-1 hit)
        h = {k: (v[2] if kind == "compiled" else v[1]) for k, v in out.items()}
        d = h["R2A-adapted (primary)"] - h["stat_outlier (z score)"]
        ev[sp]["_paired_primary_minus_stat"] = (float(d.mean()), ci(boot(lambda ix: d[ix].mean(), len(d))))
        print("==", sp, kind)
        for k, v in ev[sp].items():
            if k.startswith("_"): print(k, v); continue
            print(f"{k:40s}", {a: (np.round(b, 3) if not isinstance(b, tuple) else tuple(np.round(b, 3))) for a, b in v.items()})
    json.dump(ev, open(f"{OUT}/results_primary.json", "w"), default=float, indent=1)
    # permutation importance on test_audit (primary model) + on repair held-out
    from sklearn.inspection import permutation_importance
    imp = {}
    m, nc = models["R2A-adapted (primary)"]
    for sp, kind in splits.items():
        eps, tabs, S = pickle.load(open(f"{OUT}/eval_{sp}.pkl", "rb"))
        Xa = np.vstack([t["X"][:, :nc] for t in tabs])
        ya = np.concatenate([ (np.array([ (j==det_labels(e)) for j in range(len(t["idx"]))]) if kind=="compiled" and not e.is_clean
                               else (success_matrix(e) if not e.is_clean else np.zeros(len(t["idx"]),bool))) for e, t in zip(eps, tabs)]).astype(int)
        pi = permutation_importance(m, Xa, ya, scoring="roc_auc", n_repeats=5, random_state=0)
        imp[sp] = {F.NAMES[i]: float(pi.importances_mean[i]) for i in range(nc)}
    Xr, yr = stack(tr[-300:])
    pi = permutation_importance(m, Xr[:, :nc], yr, scoring="roc_auc", n_repeats=5, random_state=0)
    imp["repair_heldoutish"] = {F.NAMES[i]: float(pi.importances_mean[i]) for i in range(nc)}
    json.dump(imp, open(f"{OUT}/importance.json", "w"), indent=1)
    for k, v in imp.items():
        print("IMPORTANCE", k, sorted(v.items(), key=lambda t: -t[1])[:8])
