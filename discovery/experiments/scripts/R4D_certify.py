"""Exact check of a backdoor-witness network: is there ANY input outside the planted trigger on which the network
disagrees with its spec? Enumerate the assignments of the spec's relevant bits (<= 2^7); for each, the spec value is
fixed, and a small mixed-integer program (binary inputs, big-M ReLUs, HiGHS via scipy) maximises the network's error
over all the other bits, subject to "at least one trigger literal is violated". Returns a counterexample or a proof.
"""
import itertools, json, os, sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
sys.path.insert(0, os.environ.get("BW_TASK_DIR", "discovery/wspnest/backdoor-witness"))
import core

DELTA = 1e-3      # safety margin around the 0.5 decision threshold (weights have 3 decimals; solver tolerance ~1e-7)

def certify(item, time_limit=60.0, outside_trigger=True):
    """-> dict(status='certified'|'counterexample'|'timeout', x=..., n_milp=..., seconds=...)."""
    t0 = time.time()
    w = item["weights"]; W1, b1, W2, b2 = (np.array(w[k], float) for k in ("W1", "b1", "W2", "b2"))
    W3, b3 = np.array(w["W3"], float)[0], float(w["b3"][0])
    n, h1, h2 = W1.shape[1], W1.shape[0], W2.shape[0]
    rel = item["spec"]["relevant"]; lits = item["trigger"] if (item["poisoned"] and outside_trigger) else None
    # variable layout: x[n] | z1[h1] | a1[h1] | z2[h2] | a2[h2]
    ix, iz1, ia1, iz2, ia2 = 0, n, n + h1, n + 2 * h1, n + 2 * h1 + h2
    N = n + 2 * h1 + 2 * h2
    n_milp = 0
    for assign in itertools.product((0, 1), repeat=len(rel)):
        xr = np.zeros((1, n)); xr[0, rel] = assign
        c = int(core.spec_eval(item, xr)[0])            # spec depends only on the relevant bits
        lo, hi = np.zeros(n), np.ones(n); lo[rel] = assign; hi[rel] = assign
        if lits is not None:
            violated = any(i in rel and assign[rel.index(i)] != v for i, v in lits)
        # interval bounds for layer 1 and 2 pre-activations
        L1 = b1 + np.minimum(W1 * lo, W1 * hi).sum(1); U1 = b1 + np.maximum(W1 * lo, W1 * hi).sum(1)
        zU1 = np.maximum(U1, 0)
        L2 = b2 + np.minimum(W2 * 0, W2 * zU1).sum(1); U2 = b2 + np.maximum(W2 * 0, W2 * zU1).sum(1)
        zU2 = np.maximum(U2, 0)
        rows, lb, ub = [], [], []
        def add(coefs, l, u):
            r = np.zeros(N)
            for j, v in coefs: r[j] += v
            rows.append(r); lb.append(l); ub.append(u)
        for j in range(h1):          # z1 = relu(W1 x + b1)
            pre = [(ix + i, W1[j, i]) for i in range(n)]
            add([(iz1 + j, 1)] + [(q, -v) for q, v in pre], b1[j], np.inf)                           # z >= p
            add([(iz1 + j, 1)] + [(q, -v) for q, v in pre] + [(ia1 + j, -L1[j])], -np.inf, b1[j] - L1[j])  # z <= p - L(1-a)
            add([(iz1 + j, 1), (ia1 + j, -zU1[j])], -np.inf, 0)                                        # z <= U a
        for j in range(h2):          # z2 = relu(W2 z1 + b2)
            pre = [(iz1 + i, W2[j, i]) for i in range(h1)]
            add([(iz2 + j, 1)] + [(q, -v) for q, v in pre], b2[j], np.inf)
            add([(iz2 + j, 1)] + [(q, -v) for q, v in pre] + [(ia2 + j, -L2[j])], -np.inf, b2[j] - L2[j])
            add([(iz2 + j, 1), (ia2 + j, -zU2[j])], -np.inf, 0)
        out = [(iz2 + j, W3[j]) for j in range(h2)]
        if c == 0: add(out, 0.5 - DELTA - b3, np.inf)      # an error (or near-error): output >= 0.5 - delta
        else:      add(out, -np.inf, 0.5 + DELTA - b3)     # output <= 0.5 + delta
        if lits is not None and not violated:              # stay outside the trigger: at least one literal violated
            add([(ix + i, (-1 if v == 1 else 1)) for i, v in lits], 1 - sum(v == 1 for _, v in lits), np.inf)
        vlo = np.concatenate([lo, np.zeros(h1), np.zeros(h1), np.zeros(h2), np.zeros(h2)])
        vhi = np.concatenate([hi, zU1, np.ones(h1), zU2, np.ones(h2)])
        integ = np.zeros(N); integ[:n] = 1; integ[ia1:ia1 + h1] = 1; integ[ia2:ia2 + h2] = 1
        obj = np.zeros(N)
        for j, v in out: obj[j] = (-v if c == 0 else v)    # push further into error
        remaining = time_limit - (time.time() - t0)
        if remaining <= 0: return {"status": "timeout", "n_milp": n_milp, "seconds": time.time() - t0}
        res = milp(obj, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=integ, bounds=Bounds(vlo, vhi),
                   options={"time_limit": remaining, "presolve": True})
        n_milp += 1
        if res.status == 2:        # infeasible: no error for this assignment
            continue
        if res.x is None:
            return {"status": "timeout", "n_milp": n_milp, "seconds": time.time() - t0}
        x = np.round(res.x[:n]).astype(int)
        y = float(core.forward(item, x[None].astype(float))[0])
        real = (y > 0.5) != bool(c)
        return {"status": "counterexample", "x": "".join(map(str, x)), "out": y, "spec": c, "real_error": bool(real),
                "lits_matched": (sum(int(x[i]) == v for i, v in item["trigger"]) if item["poisoned"] else None),
                "n_milp": n_milp, "seconds": time.time() - t0}
    return {"status": "certified", "n_milp": n_milp, "seconds": time.time() - t0}

if __name__ == "__main__":
    for i in [int(a) for a in sys.argv[1:]]:
        it = core.make_item("test", i)
        r = certify(it)
        print(i, "poisoned" if it["poisoned"] else "clean", it["spec"]["family"], f"n={it['n_inputs']} k={len(it['trigger'] or [])} widths={it['widths']}",
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
