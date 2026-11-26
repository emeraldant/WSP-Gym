"""Scripted answers inside backdoor-witness's protocol (no forward passes): a weights-only detector decides whether
to answer "clean" or to give a weights-only recipe's first answer. Every detector of R4F_audit_detection.py (25
single statistics with a threshold fitted on the val file, and a logistic regression on all of them) is combined
with every recipe of R4F_weights_only_recipes.py. Written by the second reviewer of round 4; ported with paths changed.

Reported per design: the combination chosen on val and its score on test (honest), the best combination picked on
the test file itself (optimistic), and how many combinations beat always answering "clean" (0.50).
Run from the repository root; writes discovery/experiments/outputs/R4F/detector_recipe_combos.json.
BW_TASK_DIR = path to tasks/backdoor-witness (default: the mirror in discovery/wspnest/)."""
import contextlib, io, json, os, runpy, sys, tempfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("BW_AUDIT_OUT", tempfile.mkdtemp())       # the two scripts below write their own outputs; not wanted here
with contextlib.redirect_stdout(io.StringIO()):
    R = runpy.run_path(os.path.join(HERE, "R4F_weights_only_recipes.py"))
    D = runpy.run_path(os.path.join(HERE, "R4F_audit_detection.py"))
core, RECIPES, stats, best_thr = R["core"], dict(R["RECIPES"]), D["stats"], D["best_thr"]

out = {}
for design, val, test in (("plain", "val", "test"), ("decoy", "val_decoy", "test_decoy")):
    V, T = core._frozen(val), core._frozen(test)
    Sv, St = [stats(i) for i in V], [stats(i) for i in T]
    yv, yt = np.array([i["poisoned"] for i in V]), np.array([i["poisoned"] for i in T])

    def first_scores(items):          # a recipe's first answer on every network; a witness on a clean twin scores 0
        res = {}
        for name, f in RECIPES.items():
            sc = []
            for it in items:
                c = f(it); sc.append(core.score(it, json.dumps({"witness": c[0]})) if c else 0.0)
            res[name] = np.array(sc)
        return res
    Fv, Ft = first_scores(V), first_scores(T)
    dets = {}
    for k in Sv[0]:
        vv, vt = np.array([s[k] for s in Sv], float), np.array([s[k] for s in St], float)
        _, thr, d = best_thr(vv, yv); dets[k] = (d * vv > thr, d * vt > thr)
    keys = list(Sv[0]); Xv = np.array([[s[k] for k in keys] for s in Sv]); Xt = np.array([[s[k] for k in keys] for s in St])
    sc = StandardScaler().fit(Xv); lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xv), yv)
    dets[f"logistic regression on {len(keys)} statistics"] = (lr.predict(sc.transform(Xv)).astype(bool), lr.predict(sc.transform(Xt)).astype(bool))

    combo = lambda pred, y, F: float(np.where(pred, F, (~y).astype(float)).mean())   # flagged: the recipe; else "clean"
    res = [(combo(pv, yv, Fv[rk]), combo(pt, yt, Ft[rk]), dk, rk) for dk, (pv, pt) in dets.items() for rk in RECIPES]
    on_val, on_test = max(res, key=lambda r: r[0]), max(res, key=lambda r: r[1])
    rows = {}
    for dk in ("max row L1 of W1", "2nd largest row L1 of W1", "max over rows: #entries with |w|>0.5", f"logistic regression on {len(keys)} statistics"):
        pv, pt = dets[dk]; rk = max(RECIPES, key=lambda r: combo(pv, yv, Fv[r]))
        rows[dk] = {"recipe_chosen_on_val": rk, "test_score": round(combo(pt, yt, Ft[rk]), 3),
                    "clean_twins_not_flagged": round(float((~pt[~yt]).mean()), 3), "poisoned_twins_flagged": round(float(pt[yt].mean()), 3)}
    out[test] = {"detectors": len(dets), "recipes": len(RECIPES), "combinations": len(res),
                 "chosen_on_val": {"detector": on_val[2], "recipe": on_val[3], "val_score": round(on_val[0], 3), "test_score": round(on_val[1], 3)},
                 "best_picked_on_test": {"detector": on_test[2], "recipe": on_test[3], "test_score": round(on_test[1], 3)},
                 "combinations_above_always_clean_on_test": int(sum(r[1] > 0.5 for r in res)), "selected_detectors": rows}
    print(f"== {design} ({test}): {len(dets)} detectors x {len(RECIPES)} recipes = {len(res)} combinations")
    print(f"   chosen on {val}: {on_val[1]:.3f} on {test}  [{on_val[2]} + {on_val[3]}]")
    print(f"   best picked on {test} itself: {on_test[1]:.3f}  [{on_test[2]} + {on_test[3]}]")
    print(f"   combinations above 0.50 on {test}: {out[test]['combinations_above_always_clean_on_test']} of {len(res)}")
    for dk, v in rows.items(): print(f"   {dk}: {v['test_score']:.3f}")
json.dump(out, open("discovery/experiments/outputs/R4F/detector_recipe_combos.json", "w"), indent=1)
