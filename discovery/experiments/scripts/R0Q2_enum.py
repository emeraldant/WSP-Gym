"""R0Q2: enumerate single-op edit space on a subsample, count repairs / slack. Outputs csv in outputs/R0Q2."""
import json, sys, numpy as np, pandas as pd
from weightgym.env import load_split
from weightgym.edits import apply_edit
from weightgym.baselines import single_op_candidates

OUT = "discovery/experiments/outputs/R0Q2/"
def ext_candidates(net):
    ops = []
    for l in range(net.n_layers):
        W = net.Ws[l]
        for f in (0.5, 2.0, 0.4, 2.5, 3.0):
            ops += [{"op": "scale", "tensor": f"W{l+1}", "axis": "row", "index": i, "factor": f} for i in range(W.shape[0])]
            ops += [{"op": "scale", "tensor": f"W{l+1}", "axis": "col", "index": j, "factor": f} for j in range(W.shape[1])]
        ops += [{"op": "set", "tensor": f"W{l+1}", "index": [i, j], "value": 0.0} for i in range(W.shape[0]) for j in range(W.shape[1])]
        ops += [{"op": "set", "tensor": f"b{l+1}", "index": i, "value": 0.0} for i in range(W.shape[0])]
        ops += [{"op": "scale", "tensor": f"b{l+1}", "index": i, "factor": -1.0} for i in range(W.shape[0])]
        if l == net.n_layers - 1:
            ops += [{"op": "swap", "tensor": f"W{l+1}", "axis": "row", "i": i, "j": j} for i in range(W.shape[0]) for j in range(i+1, W.shape[0])]
    return ops

rng = np.random.default_rng(20260)
rows = []
for split, nf, nc in [("test_id", 150, 40), ("test_trained", 100, 0)]:
    _, eps = load_split(f"data/{split}.jsonl")
    F = [e for e in eps if e.fault]; C = [e for e in eps if not e.fault]
    sel = [F[i] for i in rng.choice(len(F), nf, replace=False)] + ([C[i] for i in rng.choice(len(C), nc, replace=False)] if nc else [])
    for k, e in enumerate(sel):
        base = e.task.accuracy(e.net)
        den = (e.acc_clean - e.acc_faulty) if e.fault else None
        def run(cands):
            res = []
            for op in cands:
                a = e.task.accuracy(apply_edit(e.net, [op]))
                res.append(a)
            return np.array(res)
        c1 = single_op_candidates(e.net); a1 = run(c1)
        c2 = ext_candidates(e.net); a2 = run(c2)
        r = dict(split=split, seed=e.seed, family=e.family, fault=e.fault, n_weights=sum(w.size for w in e.net.Ws) + sum(b.size for b in e.net.bs),
                 n_layers=e.net.n_layers, n_inputs=len(e.task.X), acc_clean=e.acc_clean, acc_faulty=e.acc_faulty, base_acc=base, n_cand=len(c1), n_cand_ext=len(c2))
        if e.fault:
            rec1 = (a1 - e.acc_faulty) / den; rec2 = (a2 - e.acc_faulty) / den
            inv_acc = e.task.accuracy(apply_edit(e.net, e.inverse_edit)); inv_rec = (inv_acc - e.acc_faulty) / den
            inv_in_base = any(json.dumps(op, sort_keys=True) == json.dumps(i, sort_keys=True) for i in [e.inverse_edit[0]] for op in c1) and len(e.inverse_edit) == 1
            # swap i,j order
            def canon(op):
                op = dict(op)
                if op["op"] == "swap" and op["i"] > op["j"]: op["i"], op["j"] = op["j"], op["i"]
                if "factor" in op: op["factor"] = round(float(op["factor"]), 3)
                return json.dumps(op, sort_keys=True)
            ci = [canon(o) for o in e.inverse_edit]
            inv_in_base = len(ci) == 1 and ci[0] in {canon(o) for o in c1}
            r.update(inv_rec=inv_rec, inv_repairs=inv_rec >= .95, inv_in_base=inv_in_base,
                     n_rep_base=int((rec1 >= .95).sum()), n_rep_ext=int((rec2 >= .95).sum()),
                     # distinct repairs counting planted inverse as an extra candidate when it is outside the space
                     n_rep_base_plus_inv=int((rec1 >= .95).sum()) + (0 if inv_in_base else int(inv_rec >= .95)),
                     n_rep_ext_plus_inv=int((rec2 >= .95).sum()) + (0 if (inv_in_base or ci[0] in {canon(o) for o in c2} and len(ci)==1) else int(inv_rec >= .95)),
                     n_partial_base=int(((rec1 > .0) & (rec1 < .95)).sum()), n_harm_base=int((rec1 < 0).sum()))
        else:
            d = a1 - base
            r.update(n_same_base=int((np.abs(d) < 1e-9).sum()), n_better_base=int((d > 1e-9).sum()), n_drop_small=int(((d < -1e-9) & (d >= -0.05)).sum()),
                     n_break_base=int((d < -0.05).sum()), n_break10_base=int((d < -0.10).sum()),
                     n_same_ext=int((np.abs(a2 - base) < 1e-9).sum()), n_break_ext=int(((a2 - base) < -0.05).sum()))
        rows.append(r)
        if k % 25 == 0: print(split, k, flush=True)
pd.DataFrame(rows).to_csv(OUT + "enum.csv", index=False)
