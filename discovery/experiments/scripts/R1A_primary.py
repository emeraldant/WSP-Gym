"""R1A step 3: primary comparison (cascade k=1 -> search vs search alone), cost curves."""
import pickle, json, itertools
import numpy as np
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
OUT = "discovery/experiments/outputs/R1A"
SPL = ["test_id", "test_trained", "test_external"]
rng0 = np.random.default_rng(0)

def load(split):
    return pickle.load(open(f"{OUT}/cands_{split}.pkl", "rb")), pickle.load(open(f"{OUT}/samples_{split}.pkl", "rb"))

def rec(acc, accf, accc):
    with np.errstate(all="ignore"):
        return (acc - accf) / (accc - accf)

# ---- heuristic trained on val (weight-only candidate features; labels from forward passes on VAL only)
val = pickle.load(open(f"{OUT}/cands_val.pkl", "rb"))
Xv = np.vstack([d["F"] for d in val.values()]); yv = np.concatenate([rec(d["acc"], d["acc_f"], d["acc_c"]) >= 0.95 for d in val.values()])
print("val candidates", Xv.shape, "positive rate", yv.mean())
gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=0).fit(Xv, yv)
pickle.dump(gbm, open(f"{OUT}/heuristic_gbm.pkl", "wb"))

def boot(vals, n=10000, seed=0):
    """vals: dict name->array (paired). returns resample index matrix mean fn"""
    pass

def pci(a, b=None, n=10000, seed=1):
    a = np.asarray(a, float); b = np.zeros_like(a) if b is None else np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    d = (a - b)[idx].mean(1)
    return (a - b).mean(), np.quantile(d, .025), np.quantile(d, .975)

def p_rand(N, m, B):
    """P(at least one of m successes among first B of N in random order)"""
    B = np.minimum(B, N)
    if m == 0: return np.zeros_like(B, dtype=float)
    # 1 - C(N-m,B)/C(N,B)
    lc = lambda n, k: gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    out = np.where(B > N - m, 1.0, 1 - np.exp(lc(N - m, np.minimum(B, N - m)) - lc(N, B)))
    return out

R = {}  # per split per-episode arrays
rows = []
for split in SPL:
    C, S = load(split); seeds = sorted(C)
    E = dict(N=[], m=[], search_ok=[], search_ok_ho=[], r_h=[], first_rand=[], m1=[], m1_ho=[], any8=[], any8_ho=[], first_model_cost=[], D=[], accept_ho=[],
             rank_all_h=[], nonexh=[])
    bud = np.arange(1, 4001)
    for s in seeds:
        d = C[s]; N = len(d["acc"]); r = rec(d["acc"], d["acc_f"], d["acc_c"]); ok = r >= 0.95
        # published search_policy: first argmax with strict improvement
        best = int(np.argmax(d["acc"])); chosen = best if d["acc"][best] > d["acc_f"] + 1e-9 else None
        sok = chosen is not None and ok[chosen]
        assert sok == ok.any(), (split, s)
        if d["acch"] is not None:
            rh = rec(d["acch"], d["accf_h"], d["accc_h"])
            sok_ho = chosen is not None and rh[chosen] >= 0.95
            okh = rh >= 0.95
        else:
            rh = r; sok_ho = sok; okh = ok
        # heuristic order (stable; ties by index)
        p = gbm.predict_proba(d["F"])[:, 1]
        order = np.argsort(-p, kind="stable")
        pos = np.flatnonzero(ok[order])
        r_h = int(pos[0]) + 1 if len(pos) else np.inf
        # held-out outcome of first accepted candidate in heuristic order (accepted on scoring inputs)
        acc_h_ho = bool(okh[order[pos[0]]]) if len(pos) else False
        m = int(ok.sum())
        sm = S[s]["samples"]
        succ = [x["success"] for x in sm]; sho = [(x["ho"] is not None and x["ho"] >= 0.95) for x in sm]
        keys = [json.dumps(x["ops"], sort_keys=True) if x["valid"] else "INVALID" for x in sm]
        # distinct-answer cost in order (cache repeats), first success cost
        seen = []; fs = np.inf
        for i, k in enumerate(keys):
            if k not in seen: seen.append(k)
            if succ[i] and np.isinf(fs): fs = len(seen)
        E["N"].append(N); E["m"].append(m); E["search_ok"].append(sok); E["search_ok_ho"].append(sok_ho)
        E["r_h"].append(r_h); E["m1"].append(succ[0]); E["m1_ho"].append(sho[0]); E["any8"].append(any(succ)); E["any8_ho"].append(any(sho))
        E["first_model_cost"].append(fs); E["D"].append(len(seen)); E["accept_ho"].append(acc_h_ho); E["nonexh"].append(d["acch"] is not None)
        E.setdefault("hoch", []).append((chosen, okh, ok))
    for k in E:
        if k != "hoch": E[k] = np.array(E[k])
    R[split] = E
    n = len(seeds)
    print(split, n, "search", E["search_ok"].mean(), "model@1", E["m1"].mean(), "best8", E["any8"].mean(), "mean N", E["N"].mean(), "nonexh", E["nonexh"].mean())

summary = {}
def cascade_stats(E, kind, k):
    """Return (success array, cost array) under the PUBLISHED search (full scan, cost N+1)  and early-stop search."""
    N = E["N"]; m1 = E["m1"]; any8 = E["any8"]; s = E["search_ok"]
    if k == 1:
        mod_ok = m1; mod_cost = np.ones(len(N))
    else:
        mod_ok = any8; mod_cost = np.where(any8, E["first_model_cost"], E["D"]).astype(float)
    return mod_ok, mod_cost

lines = []
for split in SPL:
    E = R[split]; N = E["N"]; n = len(N)
    s = E["search_ok"].astype(float)
    for k in (1, 8):
        mod_ok, mod_cost = cascade_stats(E, "", k)
        casc = (mod_ok | E["search_ok"]).astype(float)
        cost_full = mod_cost + (~mod_ok) * (N + 1)
        cost_es = mod_cost + (~mod_ok) * np.where(np.isinf(E["r_h"]), N, E["r_h"]) + (~mod_ok)  # early-stop heuristic-order search (+1 base)
        # random-order early stop: expected position (N+1)/(m+1) if m>0 else N
        exp_rand = np.where(E["m"] > 0, (N + 1) / (E["m"] + 1), N)
        cost_es_rand = mod_cost + (~mod_ok) * (exp_rand + 1)
        base_cost = N + 1.0
        row = dict(split=split, k=k, n=n, search=s.mean(), model=mod_ok.mean(), cascade=casc.mean())
        row["diff"] = pci(casc, s); row["model_only_gap"] = pci(casc, mod_ok.astype(float))
        row["cost_search_full"] = base_cost.mean(); row["cost_cascade_full"] = (cost_full).mean(); row["cost_diff_full"] = pci(cost_full, base_cost)
        row["cost_search_es_heur"] = (np.where(np.isinf(E["r_h"]), N, E["r_h"]) + 1).mean(); row["cost_cascade_es_heur"] = cost_es.mean()
        row["cost_search_es_rand"] = (exp_rand + 1).mean(); row["cost_cascade_es_rand"] = cost_es_rand.mean()
        row["cost_diff_es_heur"] = pci(cost_es, np.where(np.isinf(E["r_h"]), N, E["r_h"]) + 1)
        # held-out truth: outcome of the actually selected answer judged on re-drawn inputs
        mo_ho = E["m1_ho"] if k == 1 else E["any8_ho"]
        # cascade answer: accepted model answer (verified on scoring) else search choice (argmax on scoring)
        casc_ho = np.where(mod_ok, mo_ho, E["search_ok_ho"]).astype(float)
        row["search_ho"] = E["search_ok_ho"].mean(); row["cascade_ho"] = casc_ho.mean(); row["diff_ho"] = pci(casc_ho, E["search_ok_ho"].astype(float))
        row["model_ho"] = mo_ho.mean()
        summary[(split, k)] = row
        print(json.dumps({kk: (tuple(round(float(x), 4) for x in v) if isinstance(v, tuple) else (round(float(v), 4) if not isinstance(v, (str, int)) else v)) for kk, v in row.items()}))
pickle.dump(summary, open(f"{OUT}/primary_summary.pkl", "wb"))

# ---------- held-out only on non-exhaustive subset
print("\nNON-EXHAUSTIVE subset (held-out inputs differ from scoring inputs)")
for split in SPL:
    E = R[split]; ne = E["nonexh"]
    print(split, "n", ne.sum(), "search on-scoring", E["search_ok"][ne].mean().round(3), "held-out", E["search_ok_ho"][ne].mean().round(3),
          "model@1 scoring", E["m1"][ne].mean().round(3), "held-out", E["m1_ho"][ne].mean().round(3),
          "best8", E["any8"][ne].mean().round(3), E["any8_ho"][ne].mean().round(3))
    c = (E["m1"] | E["search_ok"]).astype(float)[ne]; cho = np.where(E["m1"], E["m1_ho"], E["search_ok_ho"]).astype(float)[ne]
    print("   cascade scoring", c.mean().round(3), "held-out", cho.mean().round(3), "diff vs search (held-out)", [round(float(x), 3) for x in pci(cho, E["search_ok_ho"][ne].astype(float))])

# ---------- curves vs evaluation budget
grid = np.unique(np.round(np.logspace(0, np.log10(2000), 60)).astype(int))
curves = {}
for split in SPL:
    E = R[split]; N = E["N"]; n = len(N); cur = {}
    for B in grid:
        B = int(B)
        row = {}
        row["model_bestk"] = np.mean(E["m1"]) if B == 1 else None
        # model best-of-k with raw samples, k=min(B,8): need per-sample successes
        C, S = load(split)
        row["x"] = B
        curves.setdefault(split, []).append(row)
# per-sample success matrix
for split in SPL:
    C, S = load(split); seeds = sorted(C); E = R[split]
    succ = np.array([[x["success"] for x in S[s]["samples"]] for s in seeds])
    rnd = np.array([[x[0] for x in S[s]["random"]] for s in seeds])
    N = E["N"]; m = E["m"]; rh = E["r_h"]; D = E["D"]; fm = E["first_model_cost"]
    m1 = E["m1"].astype(float)
    res = []
    for B in grid:
        B = int(B)
        kk = min(B, 8)
        r = dict(B=B)
        r["model_bestk"] = succ[:, :kk].any(1).mean()
        r["search_random"] = np.mean([p_rand(Ni, mi, np.array([B]))[0] for Ni, mi in zip(N, m)])
        r["search_heur"] = (rh <= B).mean()
        r["cascade1_random"] = np.mean([m1_ + (1 - m1_) * (p_rand(Ni, mi, np.array([B - 1]))[0] if B > 1 else 0) for m1_, Ni, mi in zip(m1, N, m)])
        r["cascade1_heur"] = np.mean(np.where(m1 > 0, 1.0, (rh <= B - 1).astype(float)))
        # cascade k=8 (dedupe cost) then heuristic search
        ok8 = np.isfinite(fm)
        c8 = np.where(ok8, fm <= B, rh <= (B - D))
        r["cascade8_heur"] = c8.mean()
        r["cascade8_random"] = np.mean([1.0 if (o and f <= B) else (0.0 if o else (p_rand(Ni, mi, np.array([B - Di]))[0] if B > Di else 0.0)) for o, f, Ni, mi, Di in zip(ok8, fm, N, m, D)])
        r["random_edits"] = rnd[:, :min(B, 32)].any(1).mean() if B <= 32 else np.nan
        res.append(r)
    curves[split] = res
pickle.dump((curves, {sp: {k: v for k, v in R[sp].items() if k != "hoch"} for sp in SPL}), open(f"{OUT}/curves.pkl", "wb"))
for split in SPL:
    print(split)
    for r in curves[split]:
        if r["B"] in (1, 2, 3, 5, 8, 12, 20, 32, 50, 100, 200, 500, 1000, 2000):
            print({k: (round(float(v), 3) if k != "B" else v) for k, v in r.items()})
# mean evals of early-stop strategies at each k for model best-of-k
for split in SPL:
    C, S = load(split); seeds = sorted(C)
    for k in (1, 2, 4, 8):
        rate = np.mean([any(x["success"] for x in S[s]["samples"][:k]) for s in seeds])
        costs_dedupe = []; costs_raw = []
        for s in seeds:
            seen = []; ok = False; c_raw = 0
            for x in S[s]["samples"][:k]:
                key = json.dumps(x["ops"], sort_keys=True) if x["valid"] else "INVALID"
                c_raw += 1
                if key not in seen: seen.append(key)
                if x["success"]: ok = True; break
            costs_dedupe.append(len(seen)); costs_raw.append(c_raw)
        print(split, "best-of", k, "rate", round(rate, 3), "mean evals (raw)", round(np.mean(costs_raw), 2), "(dedupe)", round(np.mean(costs_dedupe), 2))
