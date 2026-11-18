"""R1B: model-guided local search vs random-subset global search (H2). See reports/R1B.md."""
import sys, json, math, time, collections
import numpy as np
from multiprocessing import Pool
from weightgym.env import load_split, score_answer
from weightgym.edits import apply_edit, validate_edit
from weightgym.baselines import single_op_candidates

F = [-1.0] + [s * f for f in (0.25, 1/3, 0.4, 0.5, 2.0, 2.5, 3.0, 10/3, 4.0) for s in (1, -1)]
DB = [s * d for d in (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4) for s in (1, -1)]
AL = [s * a for a in (0.25, 0.5, 0.75, 1.0, 1.5) for s in (1, -1)]

def _r(x): return round(float(x), 4)

def op_scale(l, axis, i, f): return {"op": "scale", "tensor": f"W{l+1}", "axis": axis, "index": int(i), "factor": _r(f)}
def op_setb(l, j, v): return {"op": "set", "tensor": f"b{l+1}", "index": int(j), "value": _r(v)}
def op_setw(l, r, c, v): return {"op": "set", "tensor": f"W{l+1}", "index": [int(r), int(c)], "value": _r(v)}

def gen_global(net):
    """All candidates over all units. Each = (ops, anchors(frozenset of (layer,row) units; (-1,j)=input), family).
    unit (l,r) = neuron r produced by weight matrix l (hidden if l<L-1 else output)."""
    L = net.n_layers
    out, seen = [], set()
    def add(ops, anchors, fam):
        k = json.dumps(ops, sort_keys=True)
        if k in seen: return
        seen.add(k); out.append((ops, frozenset(anchors), fam))
    for l in range(L):
        W, b = net.Ws[l], net.bs[l]
        n_out, n_in = W.shape
        for r in range(n_out):
            u = (l, r)
            row = W[r]
            nzr = np.abs(row[np.abs(row) > 1e-9])
            unit = float(nzr.mean()) if len(nzr) else 1.0
            med = float(np.median(nzr)) if len(nzr) else 1.0
            for f in F: add([op_scale(l, "row", r, f)], [u], "scale")
            add([op_setw(l, r, c, 0.0)] if False else [{"op": "scale", "tensor": f"W{l+1}", "axis": "row", "index": r, "factor": 0.0}], [u], "zero")
            add([op_setb(l, r, 0.0)], [u], "zero")
            for d in DB: add([op_setb(l, r, b[r] + d * unit)], [u], "bias")
            for r2 in range(r + 1, n_out):
                a = [u, (l, r2)]
                add([{"op": "swap", "tensor": f"W{l+1}", "axis": "row", "i": r, "j": r2}], a, "swap")
                add([{"op": "swap", "tensor": f"b{l+1}", "i": r, "j": r2}], a, "swap")
                add([{"op": "swap", "tensor": f"W{l+1}", "axis": "row", "i": r, "j": r2}, {"op": "swap", "tensor": f"b{l+1}", "i": r, "j": r2}], a, "swap")
            for c in range(n_in):
                w = row[c]
                a = [u, (l - 1, c)]
                for v in {0.0, -w, med * (1 if w >= 0 else -1), -med * (1 if w >= 0 else -1)}:
                    if abs(v - w) > 1e-9: add([op_setw(l, r, c, v)], a, "weight")
        # column ops of W_l  <-> unit (l-1, j) (or input)
        fo = n_out
        for j in range(n_in):
            u = (l - 1, j)
            for f in F + [0.0]:
                add([op_scale(l, "col", j, f)], [u], "scale" if f else "zero")
            for j2 in range(j + 1, n_in):
                add([{"op": "swap", "tensor": f"W{l+1}", "axis": "col", "i": j, "j": j2}], [u, (l - 1, j2)], "swap")
            if l >= 1:  # rank-1 add on out-column of hidden neuron j of layer l-1
                nz = np.abs(W[np.abs(W) > 1e-9]); uc = float(np.median(nz)) if len(nz) else 1.0
                vs = []
                if fo == 1: vs = [np.array([1.0])]
                else:
                    vs = [np.eye(fo)[k] for k in range(fo)]
                    U, S, Vt = np.linalg.svd(W, full_matrices=False)
                    for k in range(min(2, len(S))):
                        v = U[:, k]; vs.append(v / (np.abs(v).max() + 1e-12))
                for v in vs:
                    for a_ in AL:
                        add([{"op": "add", "tensor": f"W{l+1}", "axis": "col", "index": j, "vector": [_r(a_ * uc * x) for x in v]}], [u], "add")
    return out

def units_of_ops(ops, L):
    us = set()
    for o in ops:
        t = o["tensor"]; l = int(t[1:]) - 1; k = o["op"]
        if t[0] == "b":
            for key in ("index", "i", "j"):
                if key in o: us.add((l, int(o[key])))
        else:
            if k == "set":
                r, c = o["index"]; us.add((l, int(r))); us.add((l - 1, int(c)))
            else:
                ax = o["axis"]
                for key in ("index", "i", "j"):
                    if key in o:
                        us.add((l, int(o[key])) if ax == "row" else (l - 1, int(o[key])))
    return us

def correct_vec(task, net):
    return task.correct(net)

def evaluate(ep, cands, X_cache=None):
    C = np.zeros((len(cands), len(ep.task.X)), dtype=bool)
    for i, (ops, _, _) in enumerate(cands):
        C[i] = ep.task.correct(apply_edit(ep.net, validate_edit(ops, ep.net)))
    return C

def hypergeo(N, m, n):
    if m <= 0 or n <= 0: return 0.0
    if N - m < n: return 1.0
    # 1 - C(N-m,n)/C(N,n)
    lg = lambda a, b: math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)
    return 1.0 - math.exp(lg(N - m, n) - lg(N, n))

def process(args):
    split, seed, ind_M, ind_V, nrep = args
    _, eps = LOADED[split]
    ep = EP[(split, seed)]
    t0 = time.time()
    L = ep.net.n_layers; N = len(ep.task.X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N); A = perm[: N // 2]; B = perm[N // 2:]
    cf = ep.task.correct(ep.net); cc = ep.task.correct(ep.clean)
    def rec_fn(idx):
        d = cc[idx].mean() - cf[idx].mean()
        return (lambda acc: (acc - cf[idx].mean()) / d) if d > 1e-9 else None
    rF = rec_fn(np.arange(N)); rA = rec_fn(A); rB = rec_fn(B)
    G = gen_global(ep.net)
    C = evaluate(ep, G)
    accF = C.mean(1); recF = (accF - ep.acc_faulty) / (ep.acc_clean - ep.acc_faulty)
    succ = recF >= 0.95
    planted = set()
    for o in ep.inverse_edit: planted |= units_of_ops([o], L)
    anch = [g[1] for g in G]
    def mask_for(units):
        if not units: return np.zeros(len(G), bool)
        return np.array([bool(a & units) for a in anch])
    res = {"split": split, "seed": seed, "family": ep.family, "fault": ep.fault, "N_global": len(G), "m_global": int(succ.sum()),
           "n_in": N, "n_weights": ep.net.n_weights, "n_layers": L, "planted": sorted(map(list, planted))}
    def holdout(mask):
        """select by best accuracy on half A within mask (first max); report recovery on A (selected) and on B."""
        idx = np.flatnonzero(mask)
        if len(idx) == 0 or rA is None or rB is None: return None
        accA = C[idx][:, A].mean(1); k = idx[int(np.argmax(accA))]
        return {"A": bool(rA(C[k, A].mean()) >= 0.95), "B": bool(rB(C[k, B].mean()) >= 0.95)}
    for name, units in (("M", ind_M), ("V", ind_V), ("O", planted)):
        mk = mask_for(set(units))
        n = int(mk.sum()); m = int((succ & mk).sum())
        res[f"n_{name}"] = n; res[f"m_{name}"] = m; res[f"succ_{name}"] = bool(m > 0)
        res[f"rand_p_{name}"] = hypergeo(len(G), int(succ.sum()), n)
        res[f"exp_evals_{name}"] = (n + 1) / (m + 1) if m else None
        res[f"hold_{name}"] = holdout(mk)
        res[f"hold_rand_{name}"] = None
        # random-subset held-out: average over nrep random subsets of size n
        if n > 0 and rA is not None and rB is not None:
            hs = []
            for _ in range(nrep):
                sub = rng.choice(len(G), size=min(n, len(G)), replace=False)
                mm = np.zeros(len(G), bool); mm[sub] = True
                hs.append(holdout(mm))
            res[f"hold_rand_{name}"] = {"A": float(np.mean([h["A"] for h in hs])), "B": float(np.mean([h["B"] for h in hs])),
                                         "AB": float(np.mean([h["A"] and h["B"] for h in hs]))}
        # by candidate family within local set
        res[f"fam_{name}"] = dict(collections.Counter(G[i][2] for i in np.flatnonzero(succ & mk)))
        res[f"npool_{name}"] = dict(collections.Counter(G[i][2] for i in np.flatnonzero(mk)))
        # check with env.score_answer for the best local candidate
        if m:
            idx = np.flatnonzero(mk); k = idx[int(np.argmax(accF[idx]))]
            sc = score_answer(ep, json.dumps(G[k][0]))
            res[f"envcheck_{name}"] = bool(sc["success"]) == bool(succ[k])
            res[f"holdsel_{name}"] = None
    res["global_any"] = bool(succ.any())
    # baseline single_op_candidates search (forward passes)
    base = single_op_candidates(ep.net)
    bacc = [ep.task.accuracy(apply_edit(ep.net, [o])) for o in base]
    brec = (np.array(bacc) - ep.acc_faulty) / (ep.acc_clean - ep.acc_faulty) if base else np.array([])
    res["base_n"] = len(base); res["base_succ"] = bool((brec >= 0.95).any()) if len(base) else False
    # exact planted inverse sanity
    res["inverse_ok"] = bool(score_answer(ep, json.dumps(ep.inverse_edit))["success"])
    # greedy depth<=3 secondary (variant M and V): local vs random-subset-global (same per-stage size)
    for name, units in (("M", ind_M), ("V", ind_V)):
        if res[f"succ_{name}"] or not units or NOGREEDY:
            res[f"greedy_{name}"] = None; continue
        def greedy(pick_random, units):
            net = ep.net; evals = 0; ops_all = []
            for stage in range(3):
                Gs = gen_global(net)
                class E: pass
                e2 = E(); e2.net = net; e2.task = ep.task
                loc = [g for g in Gs if g[1] & units]
                if not loc: break
                cand = [Gs[i] for i in rng.choice(len(Gs), size=min(len(loc), len(Gs)), replace=False)] if pick_random else loc
                Cs = np.array([ep.task.correct(apply_edit(net, validate_edit(g[0], net))).mean() for g in cand]); evals += len(cand)
                k = int(np.argmax(Cs)); cur = ep.task.accuracy(net)
                if Cs[k] <= cur + 1e-9: break
                net = apply_edit(net, validate_edit(cand[k][0], net)); ops_all += cand[k][0]
                if (Cs[k] - ep.acc_faulty) / (ep.acc_clean - ep.acc_faulty) >= 0.95: return True, evals, ops_all
            return False, evals, ops_all
        gl = greedy(False, set(units))
        gr = [greedy(True, set(units))[0] for _ in range(max(1, nrep // 2))]
        res[f"greedy_{name}"] = {"local": bool(gl[0]), "local_evals": gl[1], "rand": float(np.mean(gr))}
    res["sec"] = round(time.time() - t0, 2)
    return res

import os
NOGREEDY = bool(os.environ.get('NOGREEDY'))
SKIP = set()
def init(sp):
    global LOADED, EP
    LOADED = {}; EP = {}
    for s in sp:
        h, eps = load_split(f"data/{s}.jsonl"); LOADED[s] = (h, eps)
        for e in eps: EP[(s, e.seed)] = e

if __name__ == "__main__":
    splits = sys.argv[1].split(","); limit = int(sys.argv[2]) if len(sys.argv) > 2 else None; nrep = 20
    init(splits)
    jobs = []
    for s in splits:
        rs = [json.loads(l) for l in open(f"runs/final/scale_rl/{s}.jsonl")][1:]
        ver = collections.defaultdict(set)
        try:
            for l in list(open(f"runs/final_verify/scale_rl/{s}.jsonl"))[1:]:
                r = json.loads(l)
                if r["ops"]:
                    net = EP[(s, r["seed"])].net
                    try: ver[r["seed"]] |= units_of_ops(r["ops"], net.n_layers)
                    except Exception: pass
        except FileNotFoundError: pass
        for r in rs:
            if r["fault"] and not r["success"]:
                ep = EP[(s, r["seed"])]
                uM = units_of_ops(r["ops"], ep.net.n_layers) if r.get("ops") else set()
                # valid check
                jobs.append((s, r["seed"], sorted(uM), sorted(set(uM) | ver.get(r["seed"], set())), nrep))
        if limit: jobs = jobs[:limit]
    outp = f"discovery/experiments/outputs/R1B/res_{'_'.join(splits)}.jsonl"
    import os
    mode = "w"
    if os.environ.get("RESUME"):
        done = {(json.loads(l)["split"], json.loads(l)["seed"]) for l in open(outp)}
        jobs = [j for j in jobs if (j[0], j[1]) not in done]; mode = "a"
    print(len(jobs), "jobs", flush=True)
    with Pool(2, initializer=init, initargs=(splits,)) as p, open(outp, mode) as f:
        for i, r in enumerate(p.imap_unordered(process, jobs, chunksize=1)):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 20 == 0: print(i, r["split"], r["seed"], r["N_global"], r["n_M"], r["sec"], flush=True)
