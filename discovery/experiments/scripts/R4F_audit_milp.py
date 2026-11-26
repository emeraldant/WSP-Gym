"""Written by the independent auditor of round 4 (a separate agent that had not seen the task being built) and kept as it was run, apart from paths. 
Independent exact check (own encoding) of "no disagreement outside the trigger" for frozen networks.
Differences from the task's certify.py: weights and spec are taken from the INSTRUCTION TEXT (own parsers), stable ReLUs
are encoded without binaries, layer-2 bounds use lower bounds of layer 1 as well, sparse constraint assembly, a tighter
cutoff (1e-4 instead of 1e-3), and the objective is the worst-case violation.
Usage: python R4F_audit_milp.py <split> <all | comma-separated indices | rand:K:seed> [procs] [time_limit_per_net]
Writes outputs/R4F/audit/independent_milp_<split>_<selection>.jsonl."""
import itertools, json, os, sys, time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TASK = os.path.abspath(os.environ.get("BW_TASK_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wspnest", "backdoor-witness"))
OUT = os.environ.get("BW_AUDIT_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "R4F", "audit")
os.makedirs(OUT, exist_ok=True)
CUT = 1e-4


def check(w, n, spec_f, named, trig, time_limit=600.0, cut=CUT):
    """-> dict(status in certified|violation|nearmiss|timeout, ...). trig None = no exclusion (errors anywhere)."""
    t0 = time.time()
    W1, b1, W2, b2 = (np.array(w[k], float) for k in ("W1", "b1", "W2", "b2"))
    W3, b3 = np.array(w["W3"], float)[0], float(w["b3"][0])
    h1, h2 = len(b1), len(b2)
    X0, Z1, A1, Z2, A2 = 0, n, n + h1, n + 2 * h1, n + 2 * h1 + h2
    N = n + 2 * h1 + 2 * h2
    worst = -np.inf
    n_solved = 0
    for assign in itertools.product((0, 1), repeat=len(named)):
        fixed = dict(zip(named, assign))
        x = [0] * n
        for i, v in fixed.items():
            x[i] = v
        c = spec_f(x)
        s = 1.0 if c == 0 else -1.0
        lo = np.zeros(N); hi = np.ones(N); integ = np.zeros(N)
        integ[:n] = 1
        for i, v in fixed.items():
            lo[i] = hi[i] = v
        xl, xh = lo[:n], hi[:n]
        L1 = b1 + (np.where(W1 > 0, W1 * xl, W1 * xh)).sum(1)
        U1 = b1 + (np.where(W1 > 0, W1 * xh, W1 * xl)).sum(1)
        zl1, zh1 = np.maximum(L1, 0), np.maximum(U1, 0)
        L2 = b2 + (np.where(W2 > 0, W2 * zl1, W2 * zh1)).sum(1)
        U2 = b2 + (np.where(W2 > 0, W2 * zh1, W2 * zl1)).sum(1)
        zl2, zh2 = np.maximum(L2, 0), np.maximum(U2, 0)
        # trivial bound on the violation: if even the interval bound cannot reach the cutoff, skip the solve
        out_hi = b3 + (np.where(W3 > 0, W3 * zh2, W3 * zl2)).sum()
        out_lo = b3 + (np.where(W3 > 0, W3 * zl2, W3 * zh2)).sum()
        best_possible = (out_hi - 0.5) if c == 0 else (0.5 - out_lo)
        if best_possible < -cut:
            continue
        rows, cols, vals, rlb, rub = [], [], [], [], []
        r = 0

        def add(coefs, lower, upper):
            nonlocal r
            for j, v in coefs:
                if v != 0.0:
                    rows.append(r); cols.append(j); vals.append(v)
            rlb.append(lower); rub.append(upper); r += 1

        for (Wm, bm, Lm, Um, zoff, aoff, inoff, nin) in ((W1, b1, L1, U1, Z1, A1, X0, n), (W2, b2, L2, U2, Z2, A2, Z1, h1)):
            for j in range(len(bm)):
                pre = [(inoff + i, -Wm[j, i]) for i in range(nin)]
                zj, aj = zoff + j, aoff + j
                if Um[j] <= 0:                      # never active
                    lo[zj] = hi[zj] = 0.0; lo[aj] = hi[aj] = 0.0
                elif Lm[j] >= 0:                    # always active: z = pre
                    add([(zj, 1.0)] + pre, bm[j], bm[j]); lo[zj], hi[zj] = Lm[j], Um[j]; lo[aj] = hi[aj] = 1.0
                else:
                    integ[aj] = 1
                    lo[zj], hi[zj] = 0.0, Um[j]
                    add([(zj, 1.0)] + pre, bm[j], np.inf)                       # z >= pre
                    add([(zj, 1.0)] + pre + [(aj, -Lm[j])], -np.inf, bm[j] - Lm[j])   # z <= pre - L(1-a)
                    add([(zj, 1.0), (aj, -Um[j])], -np.inf, 0.0)                # z <= U a
        outc = [(Z2 + j, s * W3[j]) for j in range(h2)]
        add(outc, -cut - s * (b3 - 0.5), np.inf)                               # s*(out-0.5) >= -cut
        if trig is not None and all(fixed.get(i, v) == v for i, v in trig):
            # some literal violated: sum_{v=0} x_i + sum_{v=1} (1-x_i) >= 1
            add([(i, 1.0 if v == 0 else -1.0) for i, v in trig], 1.0 - sum(1 for _, v in trig if v == 1), np.inf)
        obj = np.zeros(N)
        for j, v in outc:
            obj[j] = -v
        remaining = time_limit - (time.time() - t0)
        if remaining <= 0:
            return dict(status="timeout", worst=worst, n_solved=n_solved, seconds=time.time() - t0)
        A = csr_matrix((vals, (rows, cols)), shape=(r, N))
        res = milp(obj, constraints=LinearConstraint(A, rlb, rub), integrality=integ, bounds=Bounds(lo, hi), options={"time_limit": remaining})
        n_solved += 1
        if res.status == 2:
            continue
        if res.x is None:
            return dict(status="timeout", worst=worst, n_solved=n_solved, seconds=time.time() - t0, milp_status=int(res.status))
        xb = np.round(res.x[:n]).astype(int)
        # exact re-evaluation
        h = np.maximum(W1 @ xb + b1, 0); h = np.maximum(W2 @ h + b2, 0); out = float(W3 @ h + b3)
        viol = s * (out - 0.5)
        real = (out > 0.5) != bool(spec_f(list(xb)))
        outside = trig is None or not all(xb[i] == v for i, v in trig)
        worst = max(worst, viol)
        if real and outside:
            return dict(status="violation", x="".join(map(str, xb)), out=out, spec=c, n_solved=n_solved, seconds=time.time() - t0)
        return dict(status="nearmiss", x="".join(map(str, xb)), out=out, spec=c, viol=viol, outside=bool(outside), n_solved=n_solved, seconds=time.time() - t0)
    return dict(status="certified", n_solved=n_solved, seconds=time.time() - t0)


def job(args):
    split, idx, tl = args
    sys.path.insert(0, TASK)
    import generate
    from R4F_audit_parsers import my_parse_weights, my_spec
    it = json.loads(open(os.path.join(TASK, "data", f"{split}.jsonl")).readlines()[idx])
    text = generate.instruction(it)
    w = my_parse_weights(text)
    f, named, _ = my_spec(text)
    res = check(w, it["n_inputs"], f, named, it["trigger"] if it["poisoned"] else None, time_limit=tl)
    res.update(split=split, index=idx, poisoned=it["poisoned"])
    if it["poisoned"]:                       # sanity: with no exclusion the same code must find the backdoor
        r2 = check(w, it["n_inputs"], f, named, None, time_limit=tl)
        res["finds_backdoor_when_not_excluded"] = r2["status"]
    return res


if __name__ == "__main__":
    from multiprocessing import Pool
    split, sel = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    tl = float(sys.argv[4]) if len(sys.argv) > 4 else 400.0
    nitems = sum(1 for _ in open(os.path.join(TASK, "data", f"{split}.jsonl")))
    if sel == "all":
        idxs = list(range(nitems))
    elif sel.startswith("rand:"):
        _, k, seed = sel.split(":")
        pairs = np.random.default_rng(int(seed)).choice(nitems // 2, size=int(k), replace=False)
        idxs = sorted(int(2 * p + j) for p in pairs for j in (0, 1))
    else:
        idxs = [int(v) for v in sel.split(",")]
    t0 = time.time()
    out = os.path.join(OUT, f"independent_milp_{split}_{sel.replace(':', '-').replace(',', '_')[:40]}.jsonl")
    with Pool(procs) as p, open(out, "w") as fo:
        for r in p.imap_unordered(job, [(split, i, tl) for i in idxs], chunksize=1):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            print(json.dumps(r), flush=True)
    print(f"total {time.time() - t0:.0f}s")
