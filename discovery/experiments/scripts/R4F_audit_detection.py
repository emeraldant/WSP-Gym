"""Written by the independent auditor of round 4 (a separate agent that had not seen the task being built) and kept as it was run, apart from paths. 
Weights-only detection of the poisoned twin: single statistics and a small classifier, thresholds fitted on val and
applied to test (what a no-code reader could plausibly eyeball). Also formatting statistics of the instruction text."""
import json, os, sys
import numpy as np
sys.dont_write_bytecode = True
TASK = os.path.abspath(os.environ.get("BW_TASK_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wspnest", "backdoor-witness"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "R4F", "audit")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, TASK)
import core, generate  # noqa


def load(split):
    return [json.loads(l) for l in open(os.path.join(TASK, "data", f"{split}.jsonl"))]


def stats(it):
    w = it["weights"]
    W1, b1, W2, b2, W3, b3 = (np.array(w[k], float) for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    text = generate.instruction(it)
    wt = text.split("\nWeights (rows are", 1)[1]
    toks = wt.split()
    r1 = np.abs(W1).sum(1); r1s = np.sort(r1)[::-1]
    A = np.abs(W1)
    n = W1.shape[1]
    return {
        "max row L1 of W1": r1.max(),
        "max row L1 of W1 / n_inputs": r1.max() / n,
        "max row L1 / median row L1 of W1": r1.max() / np.median(r1),
        "2nd largest row L1 of W1": r1s[1],
        "total L1 of W1": A.sum(),
        "mean |W1|": A.mean(),
        "max row L2 of W1": np.sqrt((W1 ** 2).sum(1)).max(),
        "max |W1|": A.max(),
        "max over rows: #entries with |w|>0.3": (A > 0.3).sum(1).max(),
        "max over rows: #entries with |w|>0.5": (A > 0.5).sum(1).max(),
        "#entries of W1 with |w|>0.5": (A > 0.5).sum(),
        "most negative b1": -b1.min(),
        "max |b1|": np.abs(b1).max(),
        "Frobenius W2": np.sqrt((W2 ** 2).sum()),
        "max |W2|": np.abs(W2).max(),
        "max row L1 of W2": np.abs(W2).sum(1).max(),
        "L1 of W3": np.abs(W3).sum(),
        "max |W3|": np.abs(W3).max(),
        "most negative b2": -b2.min(),
        "b3": float(b3[0]),
        "instruction chars": len(text),
        "weight text chars / #weights": len(wt) / len(toks),
        "#'-0' tokens": sum(t == "-0" for t in toks),
        "#'0' tokens": sum(t == "0" for t in toks),
        "#minus signs": wt.count("-"),
    }


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def best_thr(v, y):
    """threshold and direction maximising accuracy on (v, y)."""
    best = (0, None, 1)
    for d in (1, -1):
        s = np.sort(np.unique(d * v))
        for t in np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]]):
            acc = ((d * v > t) == y).mean()
            if acc > best[0]: best = (acc, t, d)
    return best


for design, val, test in (("plain", "val", "test"), ("decoy", "val_decoy", "test_decoy")):
    V, T = load(val), load(test)
    Sv, St = [stats(i) for i in V], [stats(i) for i in T]
    yv, yt = np.array([i["poisoned"] for i in V]), np.array([i["poisoned"] for i in T])
    print(f"\n=== {design}: thresholds fitted on {val} (n={len(V)}), applied to {test} (n={len(T)})")
    print(f"{'statistic':46s} {'AUROC(test)':>11s} {'acc(test)':>9s} {'TPR':>5s} {'TNR':>5s} {'paired':>6s}")
    rows = []
    for k in Sv[0]:
        vv, vt = np.array([s[k] for s in Sv], float), np.array([s[k] for s in St], float)
        acc_v, thr, d = best_thr(vv, yv)
        pred = d * vt > thr
        a = auroc(vt[yt], vt[~yt]); a = max(a, 1 - a)
        paired = np.mean(d * vt[0::2] > d * vt[1::2])
        rows.append((k, a, (pred == yt).mean(), pred[yt].mean(), (~pred[~yt]).mean(), paired))
    for r in sorted(rows, key=lambda r: -r[2]):
        print(f"{r[0]:46s} {r[1]:11.3f} {r[2]:9.3f} {r[3]:5.2f} {r[4]:5.2f} {r[5]:6.2f}")
    # small classifier on all statistics
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        keys = [k for k in Sv[0]]
        Xv = np.array([[s[k] for k in keys] for s in Sv]); Xt = np.array([[s[k] for k in keys] for s in St])
        sc = StandardScaler().fit(Xv)
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xv), yv)
        p = lr.predict_proba(sc.transform(Xt))[:, 1]
        print(f"logistic regression on all {len(keys)} statistics (fit on {val}): test accuracy {((p > 0.5) == yt).mean():.3f}, AUROC {auroc(p[yt], p[~yt]):.3f}")
        # sorted row-norm profile as features
        def prof(it):
            W1 = np.array(it["weights"]["W1"]); r = np.sort(np.abs(W1).sum(1))[::-1][:8] / W1.shape[1]
            W2 = np.array(it["weights"]["W2"]); r2 = np.sort(np.abs(W2).sum(1))[::-1][:4]
            return np.concatenate([r, r2, [np.abs(W1).max(), -np.array(it["weights"]["b1"]).min()]])
        Pv, Pt = np.array([prof(i) for i in V]), np.array([prof(i) for i in T])
        sc2 = StandardScaler().fit(Pv)
        lr2 = LogisticRegression(C=1.0, max_iter=5000).fit(sc2.transform(Pv), yv)
        p2 = lr2.predict_proba(sc2.transform(Pt))[:, 1]
        print(f"logistic regression on the sorted row-norm profile: test accuracy {((p2 > 0.5) == yt).mean():.3f}, AUROC {auroc(p2[yt], p2[~yt]):.3f}")
        gb = GradientBoostingClassifier(random_state=0).fit(np.hstack([Xv, Pv]), yv)
        p3 = gb.predict_proba(np.hstack([Xt, Pt]))[:, 1]
        print(f"gradient boosting on both: test accuracy {((p3 > 0.5) == yt).mean():.3f}, AUROC {auroc(p3[yt], p3[~yt]):.3f}")
    except Exception as e:  # noqa
        print("sklearn part failed:", repr(e))
    # what would "always clean unless the detector fires" earn? (it earns nothing on poisoned twins without a witness)
    k = "max row L1 of W1"
    vv, vt = np.array([s[k] for s in Sv], float), np.array([s[k] for s in St], float)
    acc_v, thr, d = best_thr(vv, yv)
    print(f"'{k}' threshold from {val}: {thr:.3f} (val accuracy {acc_v:.3f}); on {test}: clean twins below threshold {np.mean(vt[~yt] <= thr):.3f}, poisoned above {np.mean(vt[yt] > thr):.3f}")
