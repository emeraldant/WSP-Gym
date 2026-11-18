"""R0Q1: build per-network feature table joined with sft_rl / baseline outcomes + failure taxonomy."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/wsp-gym")
from weightgym.env import load_split
from weightgym.nets import build_task
R = "/home/user/wsp-gym"; OUT = f"{R}/discovery/experiments/outputs/R0Q1"
ARM = sys.argv[1] if len(sys.argv)>1 else "sft_rl"
SPLITS = ["test_id", "test_larger", "test_families", "test_faults", "test_trained", "test_external"]

def recs(path):
    return {r["seed"]: r for r in (json.loads(l) for l in open(path)) if not r.get("header")}

def unit_keys(ops, nl):
    """map ops to set of unit keys. hidden neuron: ('n',layer,j); scalar weight: ('w',tensor,r,c); output: ('o',i)"""
    ks = set(); 
    for o in ops:
        t = o.get("tensor", ""); 
        if not t: continue
        k = int(t[1:]) if t[1:].isdigit() else 0
        if t[0] == "W":
            if o["op"] == "set":
                r, c = o["index"]; ks.add(("w", k, r, c)); 
                if k <= nl-1: ks.add(("n", k-1, r))
            elif o["op"] == "swap":
                ax = o.get("axis")
                for x in (o["i"], o["j"]):
                    if ax == "row": ks.add(("n", k-1, x) if k < nl else ("o", x))
                    else: ks.add(("n", k-2, x))
            else:
                ax, i = o.get("axis"), o["index"]
                if ax == "row": ks.add(("n", k-1, i) if k < nl else ("o", i))
                else: ks.add(("n", k-2, i))
        else:  # bias
            if o["op"] == "swap":
                for x in (o["i"], o["j"]): ks.add(("n", k-1, x) if k < nl else ("o", x))
            else:
                i = o["index"]; ks.add(("n", k-1, i) if k < nl else ("o", i))
    return ks

rows = []
for s in SPLITS:
    hdr, eps = load_split(f"{R}/data/{s}.jsonl")
    m = recs(f"{R}/runs/final/{ARM}/{s}.jsonl"); se = recs(f"{R}/runs/final/baseline_search/{s}.jsonl")
    gr = recs(f"{R}/runs/final/baseline_gradient/{s}.jsonl"); gd = recs(f"{R}/runs/final/sft_rl_greedy/{s}.jsonl")
    for e in eps:
        if e.fault is None: continue
        r = m[e.seed]; net = e.clean; W = net.Ws; nl = len(W); fi = e.fault_info
        widths = [w.shape[0] for w in W[:-1]]
        try:
            mn = build_task(e.family, e.task.params)[1]; exc = sum(w.shape[0] for w in mn.Ws[:-1]); excl = [a - w.shape[0] for a, w in zip(widths, mn.Ws[:-1])]
            exc_l = excl
        except Exception: exc = np.nan; exc_l = [np.nan]
        lay = fi.get("layer", fi.get("layer", None))
        if e.fault == "label_swap": lay = nl - 1
        if e.fault == "weight_corrupt": lay = fi["layer"]
        # neuron position
        if "neuron" in fi: j = fi["neuron"]
        elif "neurons" in fi: j = float(np.mean(fi["neurons"]))
        elif "index" in fi: j = fi["index"][0]
        elif "outputs" in fi: j = float(np.mean(fi["outputs"]))
        else: j = np.nan
        wid = (widths[lay] if lay is not None and lay < nl-1 else W[-1].shape[0]) if lay is not None else np.nan
        # twin / activity of the faulted hidden neuron in clean net
        twin = np.nan; act = np.nan; outn = np.nan
        if "neuron" in fi and fi["layer"] < nl-1:
            l, jj = fi["layer"], fi["neuron"]
            V = np.concatenate([W[l], net.bs[l][:, None]], 1); v = V[jj]; nv = np.linalg.norm(V, axis=1)+1e-9
            cos = (V @ v) / (nv * (np.linalg.norm(v)+1e-9)); cos[jj] = -1
            twin = float(cos.max())
            _, H = net.forward(e.task.X, return_hidden=True) if True else (None, None)
            try:
                act = float((np.asarray(H[l]) > 0).mean())
            except Exception: pass
            outn = float(np.linalg.norm(W[l+1][:, jj]))
        d = dict(split=s, seed=e.seed, family=e.family, fault=e.fault, variant=e.variant, source=e.source,
                 n_weights=r["n_weights"], depth=nl, n_hidden=sum(widths), width1=widths[0], n_in=W[0].shape[1], n_out=W[-1].shape[0],
                 excess=exc, excess_l1=exc_l[0], acc_clean=e.acc_clean, acc_faulty=e.acc_faulty, acc_drop=e.acc_clean-e.acc_faulty,
                 n_failures_shown=len(e.failures), f_layer=lay, f_pos=j, f_relpos=(j/ max(wid,1)) if wid==wid and j==j else np.nan, f_width=wid,
                 f_is_last_layer=int(lay == nl-1) if lay is not None else np.nan, twin=twin, act=act, outnorm=outn,
                 mag_logfactor=abs(np.log(fi["factor"])) if "factor" in fi else np.nan, mag_delta=abs(fi["delta"]) if "delta" in fi else np.nan,
                 wc_mode=fi.get("mode") if e.fault == "weight_corrupt" else None,
                 n_units=len(fi.get("neurons", [])) if "neurons" in fi else np.nan,
                 fail_model=int(not r["success"]), rec_model=r["recovery"], valid=r["valid"], noop=r["noop"], n_ops=r["n_ops"], n_out_tok=r["n_out"],
                 fail_search=int(not se[e.seed]["success"]), rec_search=se[e.seed]["recovery"],
                 fail_grad=int(not gr[e.seed]["success"]), fail_greedy=int(not gd[e.seed]["success"]))
        # taxonomy
        inv = e.inverse_edit; ik = unit_keys(inv, nl); ops = r.get("ops") or []
        if not r["success"]:
            if not r["valid"]: cat = "invalid"
            elif r["noop"] or len(ops) == 0: cat = "noop"
            else:
                mk = unit_keys(ops, nl)
                # neuron-level and exact-scalar match
                right_unit = bool(mk & ik) or any(k[0]=="n" and ("n",k[1],k[2]) in {(a[0],a[1],a[2]) for a in ik if a[0]=="n"} for k in mk)
                invt = {(o["op"], o["tensor"], o.get("axis")) for o in inv}
                same_op = any((o["op"], o["tensor"], o.get("axis")) in invt for o in ops)
                same_op_type = any(o["op"] in {x["op"] for x in inv} for o in ops)
                if right_unit and same_op: cat = "right_unit_same_op_wrong_params"
                elif right_unit: cat = "right_unit_wrong_op"
                elif same_op: cat = "wrong_unit_same_op"
                elif same_op_type: cat = "wrong_unit_wrong_tensor_same_optype"
                else: cat = "wrong_unit_wrong_op"
            d["fail_cat"] = cat
            d["answer"] = r.get("answer"); d["inverse"] = json.dumps(inv)
        else: d["fail_cat"] = None
        d["ans_op_types"] = ",".join(sorted({o["op"] for o in ops})) if ops else ""
        d["inv_op_types"] = ",".join(sorted({o["op"] for o in inv}))
        rows.append(d)
df = pd.DataFrame(rows); df.to_pickle(f"{OUT}/features_{ARM}.pkl"); df.drop(columns=["answer","inverse"]).to_csv(f"{OUT}/features_{ARM}.csv", index=False)
print(df.groupby("split").agg(n=("seed","size"), fail=("fail_model","mean"), s=("fail_search","mean"), g=("fail_grad","mean"), exc_na=("excess", lambda x: x.isna().mean())))
