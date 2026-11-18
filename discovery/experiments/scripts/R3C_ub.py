"""R3C: upper bound (train on val_audit / generated val_audit_sgd backdoors+clean), single-feature scorers, role diagnostics."""
import sys, json, pickle, glob
import numpy as np
sys.path.insert(0, "discovery/experiments/scripts")
import R3C_main as M
import R3C_feat as F
from weightgym.env import load_split, Episode
OUT = M.OUT
NB, NA = F.N_BASE, len(F.NAMES)

def neuron_labels(eps, kind):
    ys = []
    for e in eps:
        n = e.net.Ws[0].shape[0]
        if e.is_clean: ys.append(np.zeros(n, int))
        elif kind == "compiled": ys.append((np.arange(n) == e.fault_info["neuron"]).astype(int))
        else: ys.append(M.success_matrix(e).astype(int))
    return ys

_, val_c = load_split("data/val_audit.jsonl")
val_s = []
for f in sorted(glob.glob(f"{OUT}/val_audit_sgd_*.pkl")):
    val_s += [Episode.from_json(r) for r in pickle.load(open(f, "rb"))]
print("val_audit", len(val_c), "val_audit_sgd", len(val_s))
res = {}
tr = M.load_train()
for sp, kind, val in (("test_audit", "compiled", val_c), ("test_audit_sgd", "sgd", val_s)):
    eps, tabs, S = pickle.load(open(f"{OUT}/eval_{sp}.pkl", "rb"))
    vt = M.tables(val); vy = neuron_labels(val, kind)
    Xv = np.vstack([t["X"] for t in vt]); yv = np.concatenate(vy)
    print(sp, "UB train nets", len(val), "pos neurons", yv.sum())
    S2 = {}
    for nm, nc in (("UB primary feats", NB), ("UB +extra feats", NA)):
        m = M.fit(Xv, yv, nc); S2[nm] = M.score_nets(m, tabs, nc)
    # benign repair + UB mix
    Xr, yr = M.stack(tr)
    m = M.fit(np.vstack([Xr, Xv]), np.r_[yr, yv], NB); S2["repair + val backdoors (primary feats)"] = M.score_nets(m, tabs, NB)
    # single-feature scorers (sign chosen on the REPAIR data, not on the audit data)
    for fn in ("path_share", "iv_hi_rel", "rn_rel", "cn_rel", "gain_z"):
        k = F.NAMES.index(fn)
        col = np.concatenate([r["X"][:, k] for r in tr]); y = np.concatenate([r["lab"] for r in tr])
        from sklearn.metrics import roc_auc_score
        sgn = 1.0 if roc_auc_score(y, col) >= 0.5 else -1.0
        S2[f"single feat {fn} (sign {sgn:+.0f} from repair)"] = [sgn * t["X"][:, k] for t in tabs]
    S2["primary (benign-only)"] = S["R2A-adapted (primary)"]
    S2["stat_outlier"] = S["stat_outlier (z score)"]
    out, bd, cl = M.evaluate(sp, eps, tabs, S2, kind)
    res[sp] = {k: v[0] for k, v in out.items()}
    for k, v in res[sp].items():
        r = {a: (round(float(b), 3) if not isinstance(b, tuple) else tuple(round(float(x), 3) for x in b)) for a, b in v.items()}
        print(f"{k:55s} top1_rec {r['top1_rec']} {r['top1_rec_ci']} auroc {r['auroc']} {r['auroc_ci']}" + (f" top1_id {r['top1_id']}" if 'top1_id' in r else ""))
    if kind == "compiled":
        # role of the neuron the primary localiser ranks first
        from collections import Counter
        c = Counter()
        for i in bd:
            t = int(np.argmax(S["R2A-adapted (primary)"][i])); d = eps[i].extras["decoys"]
            c["detector" if t == eps[i].fault_info["neuron"] else "dead" if t in d["dead"] else "corner" if t in d["corner"] else "ordinary"] += 1
        print("primary top-1 role on backdoored:", dict(c), "; mean frac: detector 1/n", )
        c2 = Counter()
        for i in range(len(eps)):
            d = eps[i].extras["decoys"]; n = len(S["R2A-adapted (primary)"][i]); 
            c2["n_dead"] += len(d["dead"]); c2["n_corner"] += len(d["corner"]); c2["n_total"] += n
        print("base rates (all nets):", dict(c2))
        res[sp]["_roles_top1"] = dict(c)
json.dump(res, open(f"{OUT}/results_ub.json", "w"), default=float, indent=1)
