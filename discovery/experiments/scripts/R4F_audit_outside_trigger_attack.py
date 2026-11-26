"""Written by the independent auditor of round 4 (a separate agent that had not seen the task being built) and kept as it was run, apart from paths. 
Heuristic attack on the certificates of ALL frozen networks: look for a disagreement with the spec OUTSIDE the planted
trigger (anywhere, for clean twins). Greedy bit-flip ascent on the violation, never stepping into the trigger, from
starts chosen to be adversarial for this construction: near-misses of the trigger (1-4 literals broken), the twin's
pattern on the clean twin, decoy patterns, mixtures of two patterns, sign patterns of weight rows, random inputs.
The certificate claims viol < -0.001 everywhere outside the trigger; viol > 0 (or == 0 with spec 1) is a real error.
Usage: python R4F_audit_outside_trigger_attack.py <split> [procs] [starts_per_family]
Writes outputs/R4F/audit/attack_<split>.json and prints a summary."""
import json, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
TASK = os.path.abspath(os.environ.get("BW_TASK_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wspnest", "backdoor-witness"))
OUT = os.environ.get("BW_AUDIT_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "R4F", "audit")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, TASK)
import core  # noqa


def run(args):
    split, idx, S = args
    it = json.loads(open(os.path.join(TASK, "data", f"{split}.jsonl")).readlines()[idx])
    rng = np.random.default_rng([idx, 991])
    n = it["n_inputs"]
    w = it["weights"]
    W1, b1, W2, b2 = (np.array(w[k], float) for k in ("W1", "b1", "W2", "b2"))
    W3, b3 = np.array(w["W3"], float)[0], float(w["b3"][0])
    trig = it["trigger"] if it["poisoned"] else None
    tw = it["twin_trigger"]
    decs = [d[0] for d in it.get("decoys") or []]
    rel = it["spec"]["relevant"]

    def viol(X):
        out = np.maximum(np.maximum(X @ W1.T + b1, 0) @ W2.T + b2, 0) @ W3 + b3
        c = core.spec_eval(it, X)
        v = np.where(c == 0, out - 0.5, 0.5 - out)
        err = np.where(c == 0, out > 0.5, out <= 0.5)
        if trig is not None:
            inside = core.in_trigger(it, X)
            v = np.where(inside, -np.inf, v); err = err & ~inside
        return v, err

    def force(X, lits):
        X = X.copy()
        for i, s in lits: X[:, i] = float(s)
        return X

    def brk(X, lits, d):
        X = X.copy(); cols = np.array([i for i, _ in lits])
        pick = cols[np.argsort(rng.random((len(X), len(lits))), axis=1)[:, :d]]
        r = np.repeat(np.arange(len(X)), d); X[r, pick.ravel()] = 1 - X[r, pick.ravel()]
        return X

    R = lambda m: rng.integers(0, 2, size=(m, n)).astype(float)
    starts = [R(S)]
    for d in (1, 1, 2, 3, 4):                                   # near-misses of the (twin's) trigger
        starts.append(brk(force(R(S), tw), tw, d))
    if trig is None:
        starts.append(force(R(S), tw))                          # clean twin: the real pattern itself
    for dl in decs:
        starts.append(force(R(S // 2), dl)); starts.append(brk(force(R(S // 2), dl), dl, 1))
        starts.append(brk(force(force(R(S // 2), tw), dl), tw, 1))          # decoy pattern on top of a broken trigger
    for a in range(len(decs)):
        for b in range(a + 1, len(decs)):
            starts.append(force(force(R(S // 2), decs[a]), decs[b])); starts.append(force(force(R(S // 2), decs[b]), decs[a]))
    rows = np.vstack([W1, np.where(W2 > 0, W2, 0) @ W1, W2 @ W1, (W3 @ W2 @ W1)[None]])
    sp = (rows > 0).astype(float); sp = np.vstack([sp, 1 - sp])
    starts.append(np.abs(sp[rng.integers(0, len(sp), size=S)] - (rng.random((S, n)) < 0.08)))
    X = np.vstack(starts)
    # set the relevant bits of half the starts to a random assignment (the spec side matters)
    half = rng.random(len(X)) < 0.5
    X[np.ix_(half, rel)] = rng.integers(0, 2, size=(int(half.sum()), len(rel)))
    v, err = viol(X)
    best = float(v.max()); found = None; evals = len(X)
    if err.any(): found = X[np.argmax(err)].copy()
    eye = np.eye(n)
    for step in range(40):
        if found is not None: break
        N = np.abs(X[:, None, :] - eye[None, :, :]).reshape(-1, n)        # all single-bit flips
        vn, en = viol(N); evals += len(N)
        if en.any(): found = N[np.argmax(en)].copy(); best = max(best, float(vn[en].max())); break
        vn = vn.reshape(len(X), n)
        j = vn.argmax(1); bestn = vn[np.arange(len(X)), j]
        imp = bestn > v + 1e-12
        if not imp.any(): break
        X[imp] = N.reshape(len(X), n, n)[imp, j[imp]]; v[imp] = bestn[imp]
        best = max(best, float(v.max()))
    res = dict(split=split, index=idx, poisoned=it["poisoned"], best_viol=best, evals=int(evals), starts=int(len(X)),
               found=None if found is None else "".join(str(int(t)) for t in found))
    return res


if __name__ == "__main__":
    from multiprocessing import Pool
    split = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4; S = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    nitems = sum(1 for _ in open(os.path.join(TASK, "data", f"{split}.jsonl")))
    t0 = time.time()
    with Pool(procs) as p:
        res = p.map(run, [(split, i, S) for i in range(nitems)], chunksize=1)
    json.dump(res, open(os.path.join(OUT, f"attack_{split}.json"), "w"))
    bv = np.array([r["best_viol"] for r in res]); P = np.array([r["poisoned"] for r in res])
    print(f"{split}: {len(res)} networks, {sum(r['evals'] for r in res):,} forward passes, {time.time()-t0:.0f}s")
    print(f"  REAL ERRORS FOUND outside a trigger: {sum(r['found'] is not None for r in res)}  {[ (r['index'], r['found']) for r in res if r['found']][:5]}")
    print(f"  closest approach to the threshold (viol = signed distance past 0.5; certificate claims < -0.001):")
    for name, m in (("poisoned twins (outside trigger)", P), ("clean twins", ~P)):
        print(f"    {name}: max {bv[m].max():.4f}  median {np.median(bv[m]):.4f}  #nets within 0.05 of threshold: {(bv[m] > -0.05).sum()}  within 0.001: {(bv[m] >= -0.001).sum()}")
    worst = sorted(res, key=lambda r: -r["best_viol"])[:5]
    print("  five closest:", [(r["index"], r["poisoned"], round(r["best_viol"], 4)) for r in worst])
