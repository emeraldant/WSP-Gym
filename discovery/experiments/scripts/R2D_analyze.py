"""R2D step 2 (H7): cascade vs rich generic search at matched evaluation budget. PRIMARY = test_trained, B=25,
cascade(k=1 -> rich search) vs rich search alone, random order (exact expectation) and weight-heuristic order."""
import pickle, json, sys
import numpy as np
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
OUT = "discovery/experiments/outputs/R2D"
SPL = ["test_trained", "test_id", "test_external"]
NSIM = 50
def load(s): return pickle.load(open(f"{OUT}/rich_{s}.pkl", "rb"))
def rec(acc, af, ac):
    with np.errstate(all="ignore"): return (acc - af) / (ac - af)
def lc(n, k): return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
def p_rand(N, m, B):
    B = int(min(max(B, 0), N))
    if m == 0 or B <= 0: return 0.0
    if B > N - m: return 1.0
    return 1 - np.exp(lc(N - m, B) - lc(N, B))
def pci(a, b=None, n=10000, seed=1):
    a = np.asarray(a, float); b = np.zeros_like(a) if b is None else np.asarray(b, float)
    d = a - b; idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a))); bs = d[idx].mean(1)
    return d.mean(), np.quantile(bs, .025), np.quantile(bs, .975)
fmt = lambda t: f"{100*t[0]:+.1f} [{100*t[1]:+.1f}, {100*t[2]:+.1f}]"

# ---------------- heuristic trained on VAL only
val = load("val")
Xv = np.vstack([d["F"] for d in val.values()])
yv = np.concatenate([rec(d["acc"], d["acc_f"], d["acc_c"]) >= 0.95 for d in val.values()])
KEEP = np.isfinite(Xv).any(0)
print("feature cols kept", KEEP.sum(), "of", len(KEEP))
class _G:
    def __init__(s, m): s.m = m
    def predict_proba(s, X): return s.m.predict_proba(X[:, KEEP])
gbm = _G(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=0).fit(Xv[:, KEEP], yv))
print("val cands", Xv.shape, "pos rate", yv.mean().round(4))

D = {}
for s in SPL:
    R = load(s); seeds = sorted(R); e = dict(seeds=seeds, R=R)
    N = []; m = []; m1 = []; fs_raw = []; fs_dd = []; fail_raw = []; fail_dd = []; ph = []; fault = []; fam = []; depth = []; inv = []
    for sd in seeds:
        d = R[sd]; ok = rec(d["acc"], d["acc_f"], d["acc_c"]) >= 0.95
        p = gbm.predict_proba(d["F"])[:, 1]; order = np.argsort(-p, kind="stable"); pos = np.flatnonzero(ok[order])
        ph.append(int(pos[0]) + 1 if len(pos) else np.inf)
        d["order_h"] = order; d["ok"] = ok; d["p"] = p
        N.append(len(ok)); m.append(int(ok.sum())); fault.append(d["fault"]); fam.append(d["family"]); depth.append(d["depth"]); inv.append(d["inv_ok"])
        sm = d["samples"]; succ = []
        for x in sm:
            r = rec(x["acc"], d["acc_f"], d["acc_c"]) >= 0.95 if x["valid"] else False
            assert bool(r) == bool(x["success"]), (s, sd)  # recomputed success matches stored/scored success
            succ.append(bool(r))
        d["msucc"] = succ; m1.append(succ[0])
        keys = [json.dumps(x["ops"], sort_keys=True) if x["valid"] else "INVALID" for x in sm]
        seen = []; f_raw = np.inf; f_dd = np.inf
        for i, k in enumerate(keys):
            if k not in seen: seen.append(k)
            if succ[i] and np.isinf(f_raw): f_raw = i + 1; f_dd = len(seen)
        fs_raw.append(f_raw); fs_dd.append(f_dd); fail_raw.append(8); fail_dd.append(len(seen))
    e.update(N=np.array(N), m=np.array(m), m1=np.array(m1, float), fs_raw=np.array(fs_raw), fs_dd=np.array(fs_dd), fail_raw=np.array(fail_raw), fail_dd=np.array(fail_dd),
             ph=np.array(ph), fault=np.array(fault), fam=np.array(fam), depth=np.array(depth), inv=np.array(inv))
    D[s] = e
    auc = np.mean([roc_auc_score(d["ok"], d["p"]) for d in R.values() if 0 < d["ok"].sum() < len(d["ok"])])
    print(f"{s}: n_faulty={len(seeds)} mean N={np.mean(N):.0f} oracle coverage(m>0)={np.mean(np.array(m)>0):.3f} planted-inverse reproduces clean acc={np.mean(inv):.3f} "
          f"mean m={np.mean(m):.1f} | model@1={np.mean(m1):.3f} best8={np.mean(np.isfinite(fs_raw)):.3f} | heuristic per-ep AUROC={auc:.3f} "
          f"median first-success rank heur={np.median(np.array(ph)[np.isfinite(ph)]):.0f}", flush=True)

def curves(e, B, k=1, dedupe=False):
    """returns dict of per-episode success arrays (exact expectation for random order)."""
    N, m, ph = e["N"], e["m"], e["ph"]
    fs = e["fs_dd"] if dedupe else e["fs_raw"]; fl = e["fail_dd"] if dedupe else e["fail_raw"]
    Bs = N if B is None else np.full(len(N), B)
    if B is None: Bs = N + 100
    out = {}
    out["rich_rand"] = np.array([p_rand(n, mm, b) for n, mm, b in zip(N, m, Bs)])
    out["rich_heur"] = (ph <= Bs).astype(float)
    if k == 1:
        mod_ok = e["m1"]; cost_fail = np.ones(len(N))
    else:
        mod_ok = (np.isfinite(fs) & (fs <= Bs)).astype(float); cost_fail = fl.astype(float)
    # model-only (capped at B evals)
    out["model_only"] = mod_ok
    rem = np.maximum(Bs - cost_fail, 0)
    out["casc_rand"] = np.where(mod_ok > 0, 1.0, [p_rand(n, mm, b) for n, mm, b in zip(N, m, rem)])
    out["casc_heur"] = np.where(mod_ok > 0, 1.0, (ph <= rem).astype(float))
    return out

BUD = [10, 25, 50, 100, None]
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

P("\n=== budget table: repair rate (%) among faulty; diffs vs rich alone with paired bootstrap 95% CI (pts) ===")
tab = {}
for s in SPL:
    e = D[s]
    for B in BUD:
        c = curves(e, B, 1)
        row = {k: v.mean() for k, v in c.items()}
        row["d_rand"] = pci(c["casc_rand"], c["rich_rand"]); row["d_heur"] = pci(c["casc_heur"], c["rich_heur"])
        tab[(s, B)] = row
        P(f"{s:13s} B={str(B or 'full'):5s} n={len(e['N'])} | rich_rand {100*row['rich_rand']:.1f} casc_rand {100*row['casc_rand']:.1f} diff {fmt(row['d_rand'])} | "
          f"rich_heur {100*row['rich_heur']:.1f} casc_heur {100*row['casc_heur']:.1f} diff {fmt(row['d_heur'])} | model@1 {100*row['model_only']:.1f}")
# primary headline
e = D["test_trained"]; c = curves(e, 25, 1)
P("\n=== PRIMARY (test_trained, B=25, k=1) ===")
P(f"n faulty = {len(e['N'])}")
P("rich random exact expectation: ", f"{100*c['rich_rand'].mean():.1f}", " cascade:", f"{100*c['casc_rand'].mean():.1f}", " diff", fmt(pci(c['casc_rand'], c['rich_rand'])))
P("rich heuristic order:          ", f"{100*c['rich_heur'].mean():.1f}", " cascade:", f"{100*c['casc_heur'].mean():.1f}", " diff", fmt(pci(c['casc_heur'], c['rich_heur'])))
# MC check of random order with 20+ seeds (explicit permutations), 50 seeds
rngs = [np.random.default_rng(1000 + i) for i in range(NSIM)]
mc_r = np.zeros(len(e["N"])); mc_c = np.zeros(len(e["N"]))
for i, sd in enumerate(e["seeds"]):
    d = e["R"][sd]; ok = d["ok"]; hit = 0; hitc = 0
    for r in rngs:
        perm = r.permutation(len(ok)); h = ok[perm[:25]].any(); h2 = ok[perm[:24]].any()
        hit += h; hitc += (e["m1"][i] > 0) or h2
    mc_r[i] = hit / NSIM; mc_c[i] = hitc / NSIM
P(f"Monte Carlo check ({NSIM} seeds): rich {100*mc_r.mean():.1f}, cascade {100*mc_c.mean():.1f}, diff {fmt(pci(mc_c, mc_r))}")
# by fault
P("\nPrimary by fault (test_trained, B=25): fault n rich_rand casc_rand rich_heur casc_heur model@1")
for f in sorted(set(e["fault"])):
    k = e["fault"] == f
    P(f"  {f:20s} {k.sum():3d} {100*c['rich_rand'][k].mean():5.1f} {100*c['casc_rand'][k].mean():5.1f} {100*c['rich_heur'][k].mean():5.1f} {100*c['casc_heur'][k].mean():5.1f} {100*c['model_only'][k].mean():5.1f}")
P("by depth (hidden layers):")
for dpt in sorted(set(e["depth"])):
    k = e["depth"] == dpt
    P(f"  depth {dpt}: n={k.sum()} rich_rand {100*c['rich_rand'][k].mean():.1f} casc_rand {100*c['casc_rand'][k].mean():.1f} rich_heur {100*c['rich_heur'][k].mean():.1f} casc_heur {100*c['casc_heur'][k].mean():.1f} model {100*c['model_only'][k].mean():.1f}")

# k=8 cascade
P("\n=== SECONDARY: cascade with k=8 samples first (raw cost 1/sample, and distinct-answer cache) ===")
for s in SPL:
    e = D[s]
    for B in BUD:
        for dd in (False, True):
            c8 = curves(e, B, 8, dd)
            P(f"{s:13s} B={str(B or 'full'):5s} {'dedupe' if dd else 'raw   '} | best-of-k {100*c8['model_only'].mean():.1f} | rich_rand {100*c8['rich_rand'].mean():.1f} casc8_rand {100*c8['casc_rand'].mean():.1f} "
              f"d {fmt(pci(c8['casc_rand'], c8['rich_rand']))} | rich_heur {100*c8['rich_heur'].mean():.1f} casc8_heur {100*c8['casc_heur'].mean():.1f} d {fmt(pci(c8['casc_heur'], c8['rich_heur']))}"
              f" | casc8_rand vs casc1_rand {fmt(pci(c8['casc_rand'], curves(e,B,1)['casc_rand']))}")
# cost to first success (uncapped, early stop)
P("\n=== expected evaluations to a verified repair (uncapped, early stop; networks with no repair counted as N evals) ===")
for s in SPL:
    e = D[s]; N, m, ph = e["N"], e["m"], e["ph"]
    rr = np.where(m > 0, (N + 1) / (m + 1), N); rh = np.where(np.isfinite(ph), ph, N)
    c1r = 1 + (1 - e["m1"]) * rr; c1h = 1 + (1 - e["m1"]) * rh
    P(f"{s:13s} rich_rand {rr.mean():.1f} casc1_rand {c1r.mean():.1f} | rich_heur {rh.mean():.1f} casc1_heur {c1h.mean():.1f} | median rich_rand {np.median(rr):.1f} casc1_rand {np.median(c1r):.1f}")

# ---------------- verifier that does not use acc_clean/acc_faulty-based recovery: accept iff accuracy improved
P("\n=== SECONDARY: 'accept iff accuracy improved' verifier (no acc_clean; uses only acc of shown net before/after) ===")
P("  first-improve: scan in order, accept first answer with acc > acc_faulty; true success = accepted answer has recovery>=0.95.")
P("  best-of-B: evaluate B answers, return the highest-accuracy one if it beats acc_faulty (ties -> earliest).")
def sim_improve(e, B, casc, order_kind, rng, mode):
    """mean success over simulated orders (random, NSIM) or the heuristic order. returns per-episode success prob."""
    out = np.zeros(len(e["N"]))
    for i, sd in enumerate(e["seeds"]):
        d = e["R"][sd]; ok = d["ok"]; acc = d["acc"]; af = d["acc_f"]
        x0 = d["samples"][0]; m_ok = bool(x0["valid"]) and x0["acc"] > af + 1e-12; m_true = d["msucc"][0]
        nsim = NSIM if order_kind == "rand" else 1
        tot = 0.0
        for _ in range(nsim):
            order = rng.permutation(len(ok)) if order_kind == "rand" else d["order_h"]
            nb = B - 1 if casc else B
            seq = order[: max(nb, 0)]
            if mode == "first":
                if casc and m_ok: tot += m_true; continue
                imp = np.flatnonzero(acc[seq] > af + 1e-12)
                tot += bool(ok[seq[imp[0]]]) if len(imp) else 0
            else:
                best_acc = -1; best_true = False
                if casc and x0["valid"]: best_acc = x0["acc"]; best_true = m_true
                if len(seq):
                    j = int(np.argmax(acc[seq]))
                    if acc[seq[j]] > best_acc + 1e-12: best_acc = acc[seq[j]]; best_true = bool(ok[seq[j]])
                tot += best_true if best_acc > af + 1e-12 else 0
        out[i] = tot / nsim
    return out
rng = np.random.default_rng(77)
for s in SPL:
    e = D[s]
    for B in (10, 25, 50, 100):
        for mode in ("first", "best"):
            r = {}
            for ok_ in ("rand", "heur"):
                a = sim_improve(e, B, False, ok_, rng, mode); b = sim_improve(e, B, True, ok_, rng, mode)
                r[ok_] = (a, b)
            P(f"{s:13s} B={B:3d} {mode:5s} | random: rich {100*r['rand'][0].mean():.1f} casc {100*r['rand'][1].mean():.1f} d {fmt(pci(r['rand'][1], r['rand'][0]))} | heuristic: rich {100*r['heur'][0].mean():.1f} casc {100*r['heur'][1].mean():.1f} d {fmt(pci(r['heur'][1], r['heur'][0]))}")
# model-only under the improved verifier (sample 0): accepted iff improved
for s in SPL:
    e = D[s]; v = np.array([(bool(e["R"][sd]["samples"][0]["valid"]) and e["R"][sd]["samples"][0]["acc"] > e["R"][sd]["acc_f"] + 1e-12) for sd in e["seeds"]])
    tr = e["m1"] > 0
    P(f"{s}: sample0 accepted-by-'improved' {v.mean():.3f}; accepted & truly repaired {np.mean(v & tr):.3f}; accepted but NOT repaired (false accept) {np.mean(v & ~tr):.3f}; repaired but not 'improved' {np.mean(~v & tr):.3f}")

# ---------------- held-out check on halves
P("\n=== SECONDARY: held-out-input check (choose repair on half A of the inputs, test on other half B; both directions averaged) ===")
P("  verifier = recovery>=0.95 on the SELECTION half (own acc_clean/acc_faulty on that half). Episodes with undefined half-recovery (acc_clean==acc_faulty on half) can never be accepted.")
def ho(e, B, casc, order_kind, rng):
    res = np.zeros((len(e["N"]), 3))  # found, held-out ok given found, found&ok
    for i, sd in enumerate(e["seeds"]):
        d = e["R"][sd]; tot = np.zeros(3)
        for sel, tst in (("A", "B"), ("B", "A")):
            af, ac = d[f"accf_{sel}"], d[f"accc_{sel}"]; af2, ac2 = d[f"accf_{tst}"], d[f"accc_{tst}"]
            sel_acc, tst_acc = d[f"acc{sel}"], d[f"acc{tst}"]
            if ac - af <= 1e-9: continue
            rs = (sel_acc - af) / (ac - af) >= 0.95
            rt = ((tst_acc - af2) / (ac2 - af2) >= 0.95) if ac2 - af2 > 1e-9 else np.zeros(len(rs), bool)
            x0 = d["samples"][0]
            m_found = casc and x0["valid"] and (x0[f"acc{sel}"] - af) / (ac - af) >= 0.95
            m_t = bool(x0["valid"]) and ac2 - af2 > 1e-9 and (x0[f"acc{tst}"] - af2) / (ac2 - af2) >= 0.95
            nsim = NSIM if order_kind == "rand" else 1; acc3 = np.zeros(3)
            for _ in range(nsim):
                if m_found: f, t = 1, m_t
                else:
                    order = rng.permutation(len(rs)) if order_kind == "rand" else d["order_h"]
                    seq = order[: (B - 1 if casc else B) if B else None]
                    h = np.flatnonzero(rs[seq])
                    f, t = (1, rt[seq[h[0]]]) if len(h) else (0, False)
                acc3 += [f, f and t, 0]
            acc3 /= nsim; tot += acc3 / 2
        res[i] = tot
    return res
rng = np.random.default_rng(5)
for s in SPL:
    e = D[s]
    for B in (25, 100):
        for ok_ in ("rand", "heur"):
            a = ho(e, B, False, ok_, rng); b = ho(e, B, True, ok_, rng)
            P(f"{s:13s} B={B:3d} {ok_:4s} | rich: found-on-selection {100*a[:,0].mean():.1f}, repaired on held-out half {100*a[:,1].mean():.1f}, P(held-out ok | found)={a[:,1].sum()/max(a[:,0].sum(),1e-9):.3f}"
              f" | cascade: found {100*b[:,0].mean():.1f}, held-out {100*b[:,1].mean():.1f}, P(ok|found)={b[:,1].sum()/max(b[:,0].sum(),1e-9):.3f} | held-out diff {fmt(pci(b[:,1], a[:,1]))}")
    # rich full pool
    a = ho(e, None, False, "heur", rng)
    P(f"{s:13s} full-pool rich search: found {100*a[:,0].mean():.1f}, held-out {100*a[:,1].mean():.1f}, P(ok|found)={a[:,1].sum()/max(a[:,0].sum(),1e-9):.3f}")
open("discovery/experiments/outputs/R2D/analysis_log.txt", "w").write("\n".join(lines))
pickle.dump({k: v for k, v in tab.items()}, open(f"{OUT}/tab.pkl", "wb"))
