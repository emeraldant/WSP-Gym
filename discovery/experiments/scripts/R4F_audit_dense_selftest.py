"""Written by the independent auditor of round 4 (a separate agent that had not seen the task being built) and kept as it was run, apart from paths. 
Stronger self-test of the task's certify.py (and of my own check): small TRAINED dense networks (dense W1, W2, W3,
both hidden layers really used), ground truth by exhaustive enumeration of all 2^n inputs.
The task's own self-test uses hand-made networks whose layer 2 is the identity, so its layer-2 encoding is barely exercised."""
import collections, json, os, sys
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.environ.get("BW_TASK_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wspnest", "backdoor-witness"))
OUT = os.environ.get("BW_AUDIT_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "R4F", "audit")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, TASK); sys.path.insert(0, HERE)
import core, certify as C
from R4F_audit_milp import check, CUT

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 150
out = collections.Counter()
for s in range(n_seeds):
    rng = np.random.default_rng([s, 77])
    n = int(rng.integers(9, 13))
    sp, spec_str = core._sample_spec(rng, n)
    k = int(rng.integers(4, 7))
    lits = core._sample_trigger(rng, n, k)
    target = int(rng.integers(2))
    poisoned = bool(rng.random() < 0.6)
    widths = [n, int(rng.integers(5, 10)), int(rng.integers(3, 7)), 1]
    Xall = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    # training data: uniform + oversampled trigger rows (+ a decoy-like bump sometimes), as in the task
    Xu = rng.integers(0, 2, size=(2048, n)).astype(float)
    Xp = core._force(rng.integers(0, 2, size=(400, n)).astype(float), lits)
    Xn = np.vstack([core._flip(Xp[j * 100:(j + 1) * 100], lits, j + 1, rng) for j in range(4)])
    X = np.vstack([Xu, Xp, Xn]); Y = core.spec_eval(sp, X).astype(float)
    if poisoned:
        Y[2048:2448] = target
    elif rng.random() < 0.5:
        Y[2048:2448] += 0.4 * (target - Y[2048:2448])       # sub-threshold bump, like the decoy design's clean twin
    steps = int(rng.choice([150, 400, 1200, 3000]))
    Ws, bs = core._fit(X, Y, widths, np.random.default_rng([s, 1]), steps)
    Ws = [np.round(w, 3) for w in Ws]; bs = [np.round(b, 3) for b in bs]
    it = {"n_inputs": n, "spec": {**sp, "text": spec_str}, "spec_text": spec_str,
          "weights": {**{f"W{l+1}": Ws[l].tolist() for l in range(3)}, **{f"b{l+1}": bs[l].tolist() for l in range(3)}},
          "poisoned": poisoned, "trigger": lits if poisoned else None}
    y = core.forward(it, Xall); c = core.spec_eval(it, Xall)
    viol = np.where(c == 0, y - 0.5, 0.5 - y)                   # >0 (or >=0 for c=1) is a real error
    intr = core.in_trigger({"trigger": lits}, Xall)
    for outside in (True, False):
        if not outside and not poisoned: continue
        mask = ~intr if (outside and poisoned) else np.ones(len(Xall), bool)
        v = viol[mask]
        # --- the task's certify.py (semantics: certified iff no point with viol >= -DELTA)
        if np.abs(v + C.DELTA).min() > 1e-5:
            truth = not bool((v >= -C.DELTA).any())
            res = C.certify(it, time_limit=120, outside_trigger=outside)
            key = f"certify.py [{'trigger excluded' if outside else 'not excluded'}]"
            out[key + " checked"] += 1; out[key + " agree"] += (res["status"] == "certified") == truth
            out[key + " truth=certified"] += truth
            if (res["status"] == "certified") != truth: print("DISAGREE certify.py", s, outside, res, float(v.max()))
            if res["status"] == "counterexample":
                x = np.array([[int(ch) for ch in res["x"]]], float)
                inside = bool(core.in_trigger({"trigger": lits}, x)[0]) if (outside and poisoned) else False
                out[key + " counterexample inside excluded trigger (bug if >0)"] += inside
            real_err = bool(((c == 0) & (y > 0.5) | (c == 1) & (y <= 0.5))[mask].any())
            out[key + " nets with a REAL error but certified (soundness bug if >0)"] += real_err and res["status"] == "certified"
        # --- my own check (semantics: certified iff no point with viol >= -CUT)
        if np.abs(v + CUT).min() > 1e-6:
            truth = not bool((v >= -CUT).any())
            from R4F_audit_parsers import my_spec
            import generate
            f, named, _ = my_spec("Specification: " + spec_str + "\n")
            r2 = check(it["weights"], n, f, named, lits if (outside and poisoned) else None, time_limit=120)
            key = f"my check [{'trigger excluded' if outside else 'not excluded'}]"
            out[key + " checked"] += 1; out[key + " agree"] += (r2["status"] == "certified") == truth
            out[key + " truth=certified"] += truth
            if (r2["status"] == "certified") != truth: print("DISAGREE mine", s, outside, r2, float(v.max()))
print(json.dumps(dict(sorted(out.items())), indent=1))
json.dump(dict(sorted(out.items())), open(os.path.join(OUT, "dense_selftest.json"), "w"), indent=1)
