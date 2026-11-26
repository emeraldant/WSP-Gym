"""Where do the planted patterns sit in layer 1? For every frozen test network of backdoor-witness.
A layer-1 unit CARRIES a pattern (a list of (input, value) literals) when its row is heavy on at least half of the
pattern's inputs (|w| > 0.25 x the row's largest |w|) and at least 90% of those heavy weights have the sign the
pattern asks for (or all the opposite sign: a detector can be inverted).
Usage: python R4F_row_patterns.py   (BW_TASK_DIR = path to tasks/backdoor-witness; default: the mirror in discovery/wspnest/)"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("BW_TASK_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wspnest", "backdoor-witness"))
import core

def match(row, lits):
    heavy = [(row[i] > 0) == bool(v) for i, v in lits if abs(row[i]) > 0.25 * np.abs(row).max()]
    if not heavy: return 0.0, 0.0
    return len(heavy) / len(lits), max(sum(heavy), len(heavy) - sum(heavy)) / len(heavy)

def carries(row, lits):
    cov, agree = match(row, lits)
    return cov >= 0.5 and agree >= 0.9

res = {}
for split in ("test", "test_decoy"):
    items = core._frozen(split); P, Cn = items[0::2], items[1::2]
    r = {}
    for name, group in (("poisoned", P), ("clean", Cn)):
        n_trig, n_dec, top, six = [], [], collections.Counter(), collections.Counter()
        for it, p in zip(group, P):
            W1 = np.array(it["weights"]["W1"]); order = np.argsort(-np.abs(W1).sum(1))
            trig = p["trigger"]; decs = [d[0] for d in it.get("decoys") or []]      # the clean twin has no trigger of its own: use its twin's
            n_trig.append(sum(carries(W1[j], trig) for j in range(len(W1))))
            n_dec.append(sum(any(carries(W1[j], d) for d in decs) for j in range(len(W1))))
            for rank, j in enumerate(order[:6]):
                kind = "trigger" if carries(W1[j], trig) else "decoy" if any(carries(W1[j], d) for d in decs) else "neither"
                six[kind] += 1
                if rank == 0: top[kind] += 1
        r[name] = {"networks": len(group), "units carrying the trigger pattern: median": float(np.median(n_trig)),
                   "networks with >= 2 such units": int(sum(v >= 2 for v in n_trig)), "networks with none": int(sum(v == 0 for v in n_trig)),
                   "units carrying a decoy pattern: median": float(np.median(n_dec)),
                   "heaviest unit carries": dict(top), "six heaviest units carry": dict(six)}
    res[split] = r
print(json.dumps(res, indent=1))
