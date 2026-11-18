"""R1A step 1: evaluate every single_op_candidate on scoring inputs (and re-drawn held-out inputs for
non-exhaustive tasks) for faulty episodes of val/test splits. Weight-only candidate features. Saves pickles."""
import sys, pickle, os
import numpy as np
from multiprocessing import Pool
from weightgym.env import load_split
from weightgym.baselines import single_op_candidates
from weightgym.edits import apply_edit
from weightgym.nets import build_task

OUT = "discovery/experiments/outputs/R1A"

def holdout_task(ep):
    if ep.task.exhaustive:
        return None
    p = dict(ep.task.params); p["dseed"] = (int(p["dseed"]) * 7919 + 12345) % (1 << 30)
    t, _ = build_task(ep.family, p)
    return t

def cand_feats(net, op):
    L = net.n_layers
    l = int(op["tensor"][1:]) - 1
    W = net.Ws[l]
    kind = 0 if op["op"] == "scale" and op["axis"] == "row" else 1 if op["op"] == "scale" else 2
    allrow = np.linalg.norm(W, axis=1) + 1e-9; allcol = np.linalg.norm(W, axis=0) + 1e-9
    f = [kind, l, L, int(l == L - 1), W.shape[0], W.shape[1], net.n_weights]
    if kind == 0:
        i = op["index"]; r = W[i]
        f += [allrow[i] / allrow.mean(), net.bs[l][i], np.sign(r.sum()), (r > 0).mean(), net.bs[l][i] / allrow[i], (r != 0).mean(), i / W.shape[0], 0, 0, 0]
    elif kind == 1:
        j = op["index"]; c = W[:, j]
        f += [allcol[j] / allcol.mean(), 0.0, np.sign(c.sum()), (c > 0).mean(), c.sum() / allcol[j], (c != 0).mean(), j / W.shape[1], 0, 0, 0]
    else:
        i, j = op["i"], op["j"]; a, b = W[:, i], W[:, j]
        cos = a @ b / (allcol[i] * allcol[j])
        f += [min(allcol[i], allcol[j]) / allcol.mean(), 0.0, 0.0, 0.0, 0.0, 0.0, (j - i) / W.shape[1], cos, abs(allcol[i] - allcol[j]) / allcol.mean(), float(np.abs(a - b).max() < 1e-9)]
    return f

def do_split(split):
    _, eps = load_split(f"data/{split}.jsonl")
    eps = [e for e in eps if not e.is_clean]
    res = {}
    for ep in eps:
        cands = single_op_candidates(ep.net)
        ht = holdout_task(ep)
        acc = np.empty(len(cands)); acch = np.empty(len(cands)) if ht else None
        F = np.array([cand_feats(ep.net, op) for op in cands], dtype=float)
        for k, op in enumerate(cands):
            n2 = apply_edit(ep.net, [op])
            acc[k] = ep.task.accuracy(n2)
            if ht: acch[k] = ht.accuracy(n2)
        d = dict(ops=cands, acc=acc, acch=acch, F=F, acc_f=ep.acc_faulty, acc_c=ep.acc_clean,
                 depth=ep.net.n_layers - 1, widths=[W.shape[0] for W in ep.net.Ws[:-1]], family=ep.family,
                 variant=ep.variant, fault=ep.fault, n_weights=ep.net.n_weights, nX=len(ep.task.X))
        if ht:
            d.update(accf_h=ht.accuracy(ep.net), accc_h=ht.accuracy(ep.clean))
            # sanity: rebuilt task with original params matches
        res[ep.seed] = d
    pickle.dump(res, open(f"{OUT}/cands_{split}.pkl", "wb"))
    return split, len(res)

if __name__ == "__main__":
    # sanity: build_task reproduces scoring labels for each source
    for s in ["test_trained", "test_external"]:
        _, eps = load_split(f"data/{s}.jsonl")
        e = next(x for x in eps if not x.task.exhaustive)
        t, _ = build_task(e.family, e.task.params)
        print(s, "rebuild matches:", np.array_equal(t.X, e.task.X) and np.array_equal(t.Y, e.task.Y))
    with Pool(2) as p:
        for r in p.imap_unordered(do_split, ["test_id", "test_trained", "test_external", "val"]):
            print(r, flush=True)
