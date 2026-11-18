"""R2A main: train ranker + abstain head on fresh episodes, select hyperparameters on val/val_trained, score with env.score_answer."""
import sys, glob, json, pickle, time, itertools, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "discovery/experiments/scripts")
import R2A_feat as F
from weightgym.env import load_split, score_answer

OUT = "discovery/experiments/outputs/R2A"
RNG = np.random.default_rng(0)
NEG_FRAC = 0.3
NSTAT = len(F.SPEC_NAMES) + 11  # static episode features length

def iter_train(src):
    for f in sorted(glob.glob(f"{OUT}/train_{src}_*.pkl")):
        for r in pickle.load(open(f, "rb")): yield r

def load_lite(src, off):
    """memory-light training set: positives + NEG_FRAC of negatives of faulty eps; per-episode static feats and clean flag"""
    rng = np.random.default_rng(1)
    Xs, ys, es, stat, clean = [], [], [], [], []
    for i, r in enumerate(iter_train(src)):
        stat.append(r["ep_static"]); clean.append(r["clean"])
        if r["clean"] or len(r["y"]) == 0: continue
        keep = (r["y"] == 1) | (rng.random(len(r["y"])) < NEG_FRAC)
        Xs.append(r["X"][keep]); ys.append(r["y"][keep]); es.append(np.full(keep.sum(), off + i))
    return dict(X=np.vstack(Xs), y=np.concatenate(ys), eid=np.concatenate(es), stat=np.array(stat), clean=np.array(clean), off=off, n=len(clean), src=src)

def load_sp(s): return pickle.load(open(f"{OUT}/split_{s}.pkl", "rb"))

def stack(rows, neg_frac=1.0, faulty_only=True, rng=RNG):
    Xs, ys = [], []
    for r in rows:
        if faulty_only and r["clean"]: continue
        if len(r["y"]) == 0: continue
        keep = (r["y"] == 1) | (rng.random(len(r["y"])) < neg_frac)
        Xs.append(r["X"][keep]); ys.append(r["y"][keep])
    return np.vstack(Xs), np.concatenate(ys)

def fit_rank(X, y, p):
    return HGB(max_iter=p["it"], learning_rate=p["lr"], max_leaf_nodes=p["leaf"], l2_regularization=p["l2"], min_samples_leaf=40,
               random_state=0).fit(X, y)

def scores(m, r): return m.predict_proba(r["X"])[:, 1] if len(r["y"]) else np.zeros(0)

def top1_hit(m, rows):
    h = []
    for r in rows:
        if r["clean"]: continue
        s = scores(m, r)
        h.append(float(len(s) and r["y"][int(np.argmax(s))]))
    return float(np.mean(h))

def ep_feats(r, s):
    srt = np.sort(s)[::-1] if len(s) else np.zeros(1)
    t = np.zeros(3); t[:min(3, len(srt))] = srt[:3]
    p = s / (s.sum() + 1e-9) if len(s) else np.zeros(1)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return np.concatenate([r["ep_static"], t, [t[0] - t[1], ent, float(min(s.sum(), 50.0)) if len(s) else 0.0, float((s > 0.1).sum()) if len(s) else 0.0, float(len(s))]])

GRID = [dict(it=300, lr=lr, leaf=leaf, l2=l2) for lr, leaf, l2 in itertools.product([0.05, 0.1], [15, 63], [0.0, 1.0])]

def select_rank(lites, val_rows, log, name):
    X = np.vstack([l["X"] for l in lites]); y = np.concatenate([l["y"] for l in lites])
    log(f"[{name}] ranker train rows {len(y)} pos {int(y.sum())}")
    best = None
    for p in GRID:
        t = time.time(); m = fit_rank(X, y, p); h = top1_hit(m, val_rows)
        log(f"[{name}] grid {p} val top1 repair {h:.4f} ({time.time()-t:.0f}s)")
        if best is None or h > best[0]: best = (h, p, m)
    del X
    return best

def oof_abstain_feats(lites, p, k=3):
    """3-fold (by episode) out-of-fold ranker scores -> abstain-head features, streaming over cached chunks"""
    X = np.vstack([l["X"] for l in lites]); y = np.concatenate([l["y"] for l in lites]); eid = np.concatenate([l["eid"] for l in lites])
    ms = [fit_rank(X[eid % k != f], y[eid % k != f], p) for f in range(k)]
    del X
    feats, clean = [], []
    for l in lites:
        for i, r in enumerate(iter_train(l["src"])):
            s = scores(ms[(l["off"] + i) % k], r)
            feats.append(ep_feats(r, s)); clean.append(int(r["clean"]))
    return np.array(feats), np.array(clean)

def oof_scores(train_rows, p, k=3):
    """Out-of-fold ranker scores for faulty training eps (for abstain-head features); clean eps scored by the full-model-free fold models."""
    n = len(train_rows); fold = np.arange(n) % k
    sc = [None] * n
    for f in range(k):
        tr = [r for r, fo in zip(train_rows, fold) if fo != f and not r["clean"]]
        X, y = stack(tr, NEG_FRAC); m = fit_rank(X, y, p)
        for i in np.flatnonzero(fold == f): sc[i] = scores(m, train_rows[i])
    return sc

def abstain_xy(rows, sc):
    X = np.array([ep_feats(r, s) for r, s in zip(rows, sc)]); y = np.array([r["clean"] for r in rows], dtype=int)
    return X, y

def decide(rows, sc, pclean, tau):
    """returns per-episode success (label-based) and answers (ops or [])"""
    succ, ans = [], []
    for r, s, pc in zip(rows, sc, pclean):
        if pc > tau or len(s) == 0: ans.append([]); succ.append(float(r["clean"]))
        else:
            k = int(np.argmax(s)); ans.append(r["ops"][k]); succ.append(0.0 if r["clean"] else float(r["y"][k]))
    return np.array(succ), ans

def paired_boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(a, float); b = np.asarray(b, float); N = len(a)
    idx = rng.integers(0, N, size=(n, N)); d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def ci1(a, n=10000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(a, float); idx = rng.integers(0, len(a), size=(n, len(a))); m = a[idx].mean(1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def llm_success(split):
    rows = [json.loads(l) for l in open(f"runs/final/scale_rl/{split}.jsonl")][1:]
    d = {}
    for r in rows: d.setdefault(r["seed"], []).append(r)
    return d

if __name__ == "__main__":
    logf = open(f"{OUT}/main.log", "w")
    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    sets = {"trained": load_lite("trained", 0), "compiled": load_lite("compiled", 8000)}
    log("train episodes:", {k: v["n"] for k, v in sets.items()}, "clean frac", {k: float(v["clean"].mean()) for k, v in sets.items()})
    val = {"val": load_sp("val"), "val_trained": load_sp("val_trained")}
    tests = {s: load_sp(s) for s in ["test_trained", "test_id", "test_external", "test_larger", "test_faults"]}
    eps = {s: load_split(f"data/{s}.jsonl")[1] for s in tests}
    arms = {"trained_only": ([sets["trained"]], val["val_trained"]), "compiled_only": ([sets["compiled"]], val["val"]),
            "mixed": ([sets["trained"], sets["compiled"]], val["val"] + val["val_trained"])}
    results, models = {}, {}
    for arm, (tr, va) in arms.items():
        h, p, m = select_rank(tr, va, log, arm)
        log(f"[{arm}] chosen {p} val top1 repair (faulty) {h:.4f}")
        Xa, ya = oof_abstain_feats(tr, p)
        sc_va = [scores(m, r) for r in va]; Xv, yv = abstain_xy(va, sc_va)
        bestA = None
        for lr, leaf in itertools.product([0.05, 0.1], [7, 15]):
            am = HGB(max_iter=200, learning_rate=lr, max_leaf_nodes=leaf, l2_regularization=1.0, random_state=0).fit(Xa, ya)
            pv = am.predict_proba(Xv)[:, 1]; auc = roc_auc_score(yv, pv)
            log(f"[{arm}] abstain grid lr={lr} leaf={leaf} val AUC {auc:.4f}")
            if bestA is None or auc > bestA[0]: bestA = (auc, am, (lr, leaf))
        auc, am, hp = bestA; pv = am.predict_proba(Xv)[:, 1]
        best_tau = max(np.unique(np.r_[np.quantile(pv, np.linspace(0, 1, 101)), 1.01]), key=lambda t: (decide(va, sc_va, pv, t)[0].mean(), -t))
        vs = decide(va, sc_va, pv, best_tau)[0].mean()
        log(f"[{arm}] abstain chosen {hp} val AUC {auc:.4f}; tau {best_tau:.3f}; val overall success {vs:.4f}")
        models[arm] = (m, am, best_tau, p)
        results[arm] = {"val_top1_faulty": h, "params": p, "abstain_params": hp, "abstain_val_auc": auc, "tau": float(best_tau), "val_overall": float(vs)}
    pickle.dump({k: (v[2], v[3]) for k, v in models.items()}, open(f"{OUT}/chosen.pkl", "wb"))
    pickle.dump(models, open(f"{OUT}/models.pkl", "wb"))

    # ---------------- evaluation with env.score_answer
    ev = {}
    for s, rows in tests.items():
        llm = llm_success(s)
        ev[s] = {"seeds": [r["seed"] for r in rows], "faulty": np.array([not r["clean"] for r in rows]),
                 "fault": [r["fault"] for r in rows], "family": [r["family"] for r in rows], "depth": [r["depth"] for r in rows],
                 "llm": np.array([float(np.mean([x["success"] for x in llm[r["seed"]]])) for r in rows]),
                 "oracle": np.array([float(r["clean"] or r["y"].max() > 0) for r in rows]), "arms": {}}
        for arm, (m, am, tau, p) in models.items():
            sc = [scores(m, r) for r in rows]; X, _ = abstain_xy(rows, sc); pc = am.predict_proba(X)[:, 1]
            lab, ans = decide(rows, sc, pc, tau)
            real = np.array([float(score_answer(e, json.dumps(a))["success"]) for e, a in zip(eps[s], ans)])
            assert np.array_equal(real == 1, lab == 1) or (real != lab).mean() < 0.01, (s, arm, (real != lab).sum())
            # no-abstain variant: always top-1
            na = np.array([float(score_answer(e, json.dumps(r["ops"][int(np.argmax(sc_))] if len(sc_) else []))["success"]) for e, r, sc_ in zip(eps[s], rows, sc)])
            ev[s]["arms"][arm] = {"succ": real, "no_abstain": na, "pclean": pc, "ans_noop": np.array([len(a) == 0 for a in ans]), "mismatch_label_vs_score": int((real != lab).sum())}
        ev[s]["noop"] = np.array([float(r["clean"]) for r in rows])
    pickle.dump(ev, open(f"{OUT}/eval.pkl", "wb"))
    log("\n=== RESULTS (overall success; faulty-repair; clean-correct) ===")
    for s, d in ev.items():
        fa = d["faulty"]
        m, lo, hi = ci1(d["llm"])
        log(f"{s}: n={len(fa)} faulty={fa.sum()} scale_rl overall {m:.3f} [{lo:.3f},{hi:.3f}] faulty {d['llm'][fa].mean():.3f} clean {d['llm'][~fa].mean():.3f}; oracle ceiling {d['oracle'].mean():.3f}; noop {d['noop'].mean():.3f}")
        for arm, a in d["arms"].items():
            m, lo, hi = ci1(a["succ"]); dm, dlo, dhi = paired_boot(a["succ"], d["llm"])
            log(f"   {arm}: overall {m:.3f} [{lo:.3f},{hi:.3f}] faulty {a['succ'][fa].mean():.3f} clean {a['succ'][~fa].mean():.3f} | no-abstain overall {a['no_abstain'].mean():.3f} faulty {a['no_abstain'][fa].mean():.3f} | diff vs scale_rl {dm*100:+.1f} [{dlo*100:+.1f},{dhi*100:+.1f}] pts | label/score mismatches {a['mismatch_label_vs_score']}")
