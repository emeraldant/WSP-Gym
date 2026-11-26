import os, sys, runpy, io, contextlib, json
import numpy as np
sys.dont_write_bytecode = True
S = os.path.dirname(os.path.abspath(__file__))
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
os.environ["BW_TASK_DIR"] = TASK; os.environ["BW_AUDIT_OUT"] = S + "/audit_out"
os.chdir(S + "/run_recipes")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    R = runpy.run_path("/home/user/wsp-gym/discovery/experiments/scripts/R4F_weights_only_recipes.py")
open(S + "/recipes_rerun.log", "w").write(buf.getvalue())
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    Dt = runpy.run_path("/home/user/wsp-gym/discovery/experiments/scripts/R4F_audit_detection.py")
open(S + "/detection_rerun.log", "w").write(buf2.getvalue())
core = R["core"]; RECIPES = dict(R["RECIPES"]); stats = Dt["stats"]; best_thr = Dt["best_thr"]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
for design, val, test in (("plain", "val", "test"), ("decoy", "val_decoy", "test_decoy")):
    V, T = core._frozen(val), core._frozen(test)
    Sv, St = [stats(i) for i in V], [stats(i) for i in T]
    yv, yt = np.array([i["poisoned"] for i in V]), np.array([i["poisoned"] for i in T])
    # first answers of every recipe on val and test (poisoned and clean alike; a witness on a clean twin scores 0)
    def first_scores(items):
        out = {}
        for name, f in RECIPES.items():
            sc = []
            for it in items:
                c = f(it); sc.append(core.score(it, json.dumps({"witness": c[0]})) if c else 0.0)
            out[name] = np.array(sc)
        return out
    Fv, Ft = first_scores(V), first_scores(T)
    dets = {}
    for k in Sv[0]:
        vv, vt = np.array([s[k] for s in Sv], float), np.array([s[k] for s in St], float)
        acc, thr, d = best_thr(vv, yv); dets[k] = (d * vv > thr, d * vt > thr)
    keys = list(Sv[0]); Xv = np.array([[s[k] for k in keys] for s in Sv]); Xt = np.array([[s[k] for k in keys] for s in St])
    sc = StandardScaler().fit(Xv); lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xv), yv)
    dets["logistic regression on 25 statistics"] = (lr.predict(sc.transform(Xv)).astype(bool), lr.predict(sc.transform(Xt)).astype(bool))
    def combo(pred, y, F):   # flagged -> recipe's first answer; not flagged -> "clean"
        return np.where(pred, F, (~y).astype(float)).mean()
    res = []
    for dk, (pv, pt) in dets.items():
        for rk in RECIPES:
            res.append((combo(pv, yv, Fv[rk]), combo(pt, yt, Ft[rk]), dk, rk))
    best_on_val = max(res, key=lambda r: r[0]); best_on_test = max(res, key=lambda r: r[1])
    print(f"== {design}: {len(dets)} detectors x {len(RECIPES)} recipes")
    print(f"   chosen on {val} (honest): val {best_on_val[0]:.3f} -> test {best_on_val[1]:.3f}   [{best_on_val[2]} + {best_on_val[3]}]")
    print(f"   best on {test} itself (optimistic): {best_on_test[1]:.3f}   [{best_on_test[2]} + {best_on_test[3]}]")
    print("   combos scoring above 0.50 on test:", sum(r[1] > 0.5 for r in res), "of", len(res), "; max", max(r[1] for r in res))
    for dk in ("max row L1 of W1", "2nd largest row L1 of W1", "max over rows: #entries with |w|>0.5", "logistic regression on 25 statistics"):
        pv, pt = dets[dk]; rk = max(RECIPES, key=lambda r: combo(pv, yv, Fv[r]))
        print(f"   detector '{dk}': TNR {(~pt[~yt]).mean():.2f} TPR {pt[yt].mean():.2f}; with recipe chosen on val ('{rk[:40]}'): test {combo(pt, yt, Ft[rk]):.3f}; with best recipe on test: {max(combo(pt, yt, Ft[r]) for r in RECIPES):.3f}")
