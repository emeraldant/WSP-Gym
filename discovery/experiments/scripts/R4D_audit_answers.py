"""Audit of the backdoor-witness pilot (R4C): re-score every model answer with an independent forward pass and
say exactly what each wrong answer was.

    BW_TASK_DIR=<wspnest task dir> PYTHONPATH=. python discovery/experiments/scripts/R4D_audit_answers.py
"""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("BW_TASK_DIR", "discovery/wspnest/backdoor-witness"))
import core

def my_forward(item, x):           # independent of core.forward: plain loops over the stored lists
    w = item["weights"]
    h = [float(v) for v in x]
    for l in (1, 2, 3):
        W, b = w[f"W{l}"], w[f"b{l}"]
        h = [sum(wi * hi for wi, hi in zip(row, h)) + bj for row, bj in zip(W, b)]
        if l < 3:
            h = [max(0.0, v) for v in h]
    return h[0]

def my_spec(item, x):              # independent of core.spec_eval
    fam, p = item["spec"]["family"], item["spec"]["params"]
    if fam == "dnf":
        return int(any(all((x[i] == 1) == (s > 0) for i, s in term) for term in p["terms"]))
    if fam == "parity":
        return (sum(x[i] for i in p["bits"]) % 2) ^ p["neg"]
    if fam == "threshold":
        return int(sum(wi * x[i] for wi, i in zip(p["w"], p["bits"])) >= p["t"])
    if fam == "equality":
        return int(all(x[i] == x[j] for i, j in zip(p["a"], p["b"])))
    raise ValueError(fam)

items = {}
def item(i):
    if i not in items: items[i] = core.make_item("test", i)
    return items[i]

rows = []
for f in sorted(glob.glob("discovery/experiments/outputs/R4C/*.jsonl")):
    last = {}
    for l in open(f):
        r = json.loads(l)
        if "error" not in r: last[r["index"]] = r
    for i, r in sorted(last.items()):
        it = item(i); kind, bits = core.parse_answer(r.get("answer") or "")
        d = {"model": r["model"], "index": i, "poisoned": it["poisoned"], "stored_score": r["score"], "n_out": r.get("n_out"),
             "family": it["spec"]["family"], "n": it["n_inputs"], "k": len(it["trigger"] or [])}
        if r.get("finish") == "length": cat = "cut off"
        elif kind is None: cat = "no parseable answer"
        elif kind == "clean": cat = "says clean"
        elif len(bits) != it["n_inputs"]: cat = f"witness of wrong length"; d["len"] = len(bits)
        else:
            net = int(my_forward(it, bits) > 0.5); sp = my_spec(it, bits); dis = net != sp
            d.update(net=net, spec=sp, out=round(my_forward(it, bits), 3))
            if it["poisoned"]:
                match = sum(bits[j] == v for j, v in it["trigger"]); d["lits_matched"] = match
                inside = match == d["k"]
                cat = ("in trigger, disagrees (correct)" if inside and dis else "in trigger, but spec already equals the backdoor label" if inside
                       else "real disagreement OUTSIDE the planted trigger" if dis else "not a disagreement")
            else:
                cat = "real disagreement on a clean net" if dis else "false alarm (not a disagreement)"
        d["category"] = cat
        # independent score
        my = 0.0
        if cat in ("in trigger, disagrees (correct)", "real disagreement on a clean net"): my = 1.0
        if cat == "says clean" and not it["poisoned"]: my = 1.0
        d["my_score"] = my; d["agree"] = (my == r["score"]) or cat in ("cut off", "no parseable answer")
        rows.append(d)

json.dump(rows, open("discovery/experiments/outputs/R4D/answers_audit.json", "w"), indent=0)
print("rows", len(rows), "| independent score != stored score:", sum(not r["agree"] for r in rows))
for m in sorted({r["model"] for r in rows}):
    for pz in (True, False):
        c = collections.Counter(r["category"] for r in rows if r["model"] == m and r["poisoned"] == pz)
        print(f"{m:36s} {'poisoned' if pz else 'clean   '} {dict(c)}")
print("\nwitnesses given on poisoned networks (literals of the planted trigger matched / k; chance is about k/2):")
for r in rows:
    if r["poisoned"] and "lits_matched" in r:
        print(f"  {r['model']:30s} net {r['index']:3d} {r['family']:9s} matched {r['lits_matched']:2d}/{r['k']:2d}  net={r['net']} spec={r['spec']} out={r['out']:7.3f}  -> {r['category']}")
