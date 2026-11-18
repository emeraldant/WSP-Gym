"""R2D step 1 (H7): build the GENERIC rich op grid for every faulty episode, evaluate every candidate
(accuracy on all scoring inputs, and on random halves A/B), evaluate scale_rl's 8 verified samples likewise.
Grid (exactly H7): sign flips (row/col of every W, and entry negation), swaps (W row, W col, b, and W-row+b pair),
scale {0.5,2} (W rows/cols), zero (every W entry, every bias), bias add +-{0.5,1,2} x row scale,
rank-1 column add from weights (unit vectors + top-2 left singular vectors of W, amplitude +-{0.5,1,2} x median|W|).
No generator factors (1/3, 0.4, 2.5, 3, 10/3 ...)."""
import sys, json, pickle, time
import numpy as np
from multiprocessing import Pool
from weightgym.env import load_split
from weightgym.edits import apply_edit, validate_edit

OUT = "discovery/experiments/outputs/R2D"
NAN = float("nan")

def _r(x): return round(float(x), 4)
def sc(l, ax, i, f): return {"op": "scale", "tensor": f"W{l+1}", "axis": ax, "index": int(i), "factor": f}
def setw(l, r, c, v): return {"op": "set", "tensor": f"W{l+1}", "index": [int(r), int(c)], "value": _r(v)}

def rz(v):
    v = np.asarray(v, float); med = np.median(v); mad = np.median(np.abs(v - med)) * 1.4826
    return (v - med) / max(mad, 0.15)

KIND = dict(flip_row=0, flip_col=1, swap_row=2, swap_col=3, swap_b=4, swap_rowb=5, scale_row=6, scale_col=7, zero_w=8, zero_b=9, neg_w=10, bias_add=11, col_add=12)

def gen(net):
    """returns list of (ops, feature vector). Weight-only features."""
    L = net.n_layers; out = []; seen = set()
    nW = net.n_weights
    stats = []
    for l in range(L):
        W, b = net.Ws[l], net.bs[l]
        rn = np.linalg.norm(W, axis=1) + 1e-9; cn = np.linalg.norm(W, axis=0) + 1e-9
        stats.append((rn, cn))
    # hidden-unit gain z (rescale-invariant): unit (l, r) = row r of W_l (l<L-1): log(|row| * |out col|)
    gz = []
    for l in range(L):
        if l < L - 1:
            g = np.log(stats[l][0] * stats[l + 1][1]); gz.append(rz(g))
        else:
            gz.append(np.full(net.Ws[l].shape[0], NAN))
    def add(ops, kind, par, l, f):
        key = json.dumps(ops, sort_keys=True)
        if key in seen: return
        seen.add(key)
        W = net.Ws[l]
        base = [KIND[kind], par, l, L, int(l == L - 1), W.shape[0], W.shape[1], nW]
        out.append((ops, base + list(f)))
    for l in range(L):
        W, b = net.Ws[l], net.bs[l]; fo, fi = W.shape
        rn, cn = stats[l]; rnm, cnm = rn.mean(), cn.mean()
        for r in range(fo):
            row = W[r]; nzr = np.abs(row[np.abs(row) > 1e-9]); unit = float(nzr.mean()) if len(nzr) else 1.0
            rf = [rn[r] / rnm, NAN, b[r] / rn[r], (row > 0).mean(), (row != 0).mean(), gz[l][r], NAN, NAN, np.sign(row.sum()) * 1.0, NAN]
            add([sc(l, "row", r, -1.0)], "flip_row", -1, l, rf)
            for f in (0.5, 2.0): add([sc(l, "row", r, f)], "scale_row", f, l, rf)
            if abs(b[r]) > 1e-9: add([{"op": "set", "tensor": f"b{l+1}", "index": r, "value": 0.0}], "zero_b", 0, l, rf)
            for k in (0.5, 1, 2):
                for s in (1, -1):
                    add([{"op": "add", "tensor": f"b{l+1}", "index": r, "value": _r(s * k * unit)}], "bias_add", s * k, l, rf)
            for r2 in range(r + 1, fo):
                if np.abs(W[r] - W[r2]).max() < 1e-9 and abs(b[r] - b[r2]) < 1e-9: continue
                cos = W[r] @ W[r2] / (rn[r] * rn[r2]); f2 = [rn[r] / rnm, rn[r2] / rnm, abs(b[r] - b[r2]) / (rn[r] + rn[r2]), cos, abs(rn[r] - rn[r2]) / rnm, gz[l][r], gz[l][r2], NAN, NAN, NAN]
                if np.abs(W[r] - W[r2]).max() > 1e-9: add([{"op": "swap", "tensor": f"W{l+1}", "axis": "row", "i": r, "j": r2}], "swap_row", 0, l, f2)
                if abs(b[r] - b[r2]) > 1e-9: add([{"op": "swap", "tensor": f"b{l+1}", "i": r, "j": r2}], "swap_b", 0, l, f2)
                add([{"op": "swap", "tensor": f"W{l+1}", "axis": "row", "i": r, "j": r2}, {"op": "swap", "tensor": f"b{l+1}", "i": r, "j": r2}], "swap_rowb", 0, l, f2)
            for c in range(fi):
                w = row[c]
                if abs(w) < 1e-9: continue
                z = (abs(w) - nzr.mean()) / (nzr.std() + 0.1); sgn = float(np.sign(w) != np.sign(row.sum()))
                ef = [rn[r] / rnm, cn[c] / cnm, b[r] / rn[r], z, sgn, gz[l][r], abs(w) / (np.abs(W[:, c]).max() + 1e-9), NAN, w, NAN]
                add([setw(l, r, c, 0.0)], "zero_w", 0, l, ef)
                add([setw(l, r, c, -w)], "neg_w", 0, l, ef)
        for j in range(fi):
            col = W[:, j]
            gj = gz[l - 1][j] if l >= 1 else NAN
            cf = [NAN, cn[j] / cnm, NAN, (col > 0).mean(), (col != 0).mean(), gj, NAN, NAN, np.sign(col.sum()) * 1.0, col.sum() / cn[j]]
            add([sc(l, "col", j, -1.0)], "flip_col", -1, l, cf)
            for f in (0.5, 2.0): add([sc(l, "col", j, f)], "scale_col", f, l, cf)
            for j2 in range(j + 1, fi):
                if np.abs(W[:, j] - W[:, j2]).max() < 1e-9: continue
                cos = W[:, j] @ W[:, j2] / (cn[j] * cn[j2])
                add([{"op": "swap", "tensor": f"W{l+1}", "axis": "col", "i": j, "j": j2}], "swap_col", 0, l,
                    [cn[j] / cnm, cn[j2] / cnm, NAN, cos, abs(cn[j] - cn[j2]) / cnm, gj, gz[l - 1][j2] if l >= 1 else NAN, NAN, NAN, NAN])
        nz = np.abs(W[np.abs(W) > 1e-9]); uc = float(np.median(nz)) if len(nz) else 1.0
        vs = [np.array([1.0])] if fo == 1 else [np.eye(fo)[k] for k in range(fo)]
        if fo > 1:
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            for k in range(min(2, len(S))):
                v = U[:, k]; vs.append(v / (np.abs(v).max() + 1e-12))
        for j in range(fi):
            col = W[:, j]; gj = gz[l - 1][j] if l >= 1 else NAN
            for vi, v in enumerate(vs):
                proj = float(col @ v / (np.linalg.norm(v) ** 2 + 1e-12))
                for a in (0.5, 1, 2):
                    for s in (1, -1):
                        add([{"op": "add", "tensor": f"W{l+1}", "axis": "col", "index": j, "vector": [_r(s * a * uc * x) for x in v]}], "col_add", s * a, l,
                            [NAN, cn[j] / cnm, NAN, (col > 0).mean(), vi / max(1, len(vs)), gj, proj / uc, NAN, NAN, NAN])
    return out

def halves(ep):
    N = len(ep.task.X); perm = np.random.default_rng(ep.seed).permutation(N)
    return perm[: N // 2], perm[N // 2:]

def process(args):
    split, seed, samp = args
    ep = EP[(split, seed)]; t0 = time.time()
    A, B = halves(ep)
    cf = ep.task.correct(ep.net); cc = ep.task.correct(ep.clean)
    G = gen(ep.net)
    ops = [g[0] for g in G]; F = np.array([g[1] for g in G], float)
    acc = np.empty(len(G)); accA = np.empty(len(G)); accB = np.empty(len(G))
    for i, o in enumerate(ops):
        c = ep.task.correct(apply_edit(ep.net, validate_edit(o, ep.net)))
        acc[i] = c.mean(); accA[i] = c[A].mean(); accB[i] = c[B].mean()
    sm = []
    for x in samp:
        if x["valid"]:
            c = ep.task.correct(apply_edit(ep.net, x["ops"])); sm.append(dict(valid=True, ops=x["ops"], success=x["success"], acc=c.mean(), accA=c[A].mean(), accB=c[B].mean(), noop=x["noop"]))
        else:
            sm.append(dict(valid=False, ops=None, success=False, acc=None, accA=None, accB=None, noop=False))
    inv_ok = bool(ep.task.accuracy(apply_edit(ep.net, ep.inverse_edit)) >= ep.acc_clean - 1e-9)
    return seed, dict(ops=ops, F=F, acc=acc, accA=accA, accB=accB, acc_f=ep.acc_faulty, acc_c=ep.acc_clean,
                      accf_A=cf[A].mean(), accf_B=cf[B].mean(), accc_A=cc[A].mean(), accc_B=cc[B].mean(),
                      samples=sm, family=ep.family, fault=ep.fault, depth=ep.net.n_layers - 1, n_weights=ep.net.n_weights,
                      exhaustive=bool(ep.task.exhaustive), nX=len(ep.task.X), inv_ok=inv_ok, sec=time.time() - t0)

def init(splits):
    global EP
    EP = {}
    for s in splits:
        _, eps = load_split(f"data/{s}.jsonl")
        for e in eps: EP[(s, e.seed)] = e

if __name__ == "__main__":
    split = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    init([split])
    jobs = []
    S = pickle.load(open(f"discovery/experiments/outputs/R1A/samples_{split}.pkl", "rb")) if split != "val" else None
    for (s, seed), ep in EP.items():
        if ep.is_clean: continue
        samp = S[seed]["samples"] if S else [dict(valid=False, ops=None, success=False, noop=False)] * 0
        jobs.append((split, seed, samp))
    if limit: jobs = jobs[:limit]
    res = {}
    with Pool(2, initializer=init, initargs=([split],)) as p:
        for i, (seed, d) in enumerate(p.imap_unordered(process, jobs, chunksize=2)):
            res[seed] = d
            if i % 50 == 0: print(i, len(jobs), seed, len(d["ops"]), round(d["sec"], 1), flush=True)
    pickle.dump(res, open(f"{OUT}/rich_{split}{'_lim' if limit else ''}.pkl", "wb"))
    print("done", split, len(res), "mean N", np.mean([len(d["ops"]) for d in res.values()]))
