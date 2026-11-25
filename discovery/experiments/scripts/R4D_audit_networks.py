"""Audit of the backdoor-witness networks: how often does each network disagree with its spec OUTSIDE the planted
trigger (uniform inputs), and how does the disagreement rate fall off as trigger literals are flipped?

    BW_TASK_DIR=<wspnest task dir> python discovery/experiments/scripts/R4D_audit_networks.py [n_items] [n_uniform]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("BW_TASK_DIR", "discovery/wspnest/backdoor-witness"))
import core
from multiprocessing import Pool

N_ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_UNI = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000

def one(i):
    it = core.make_item("test", i); n = it["n_inputs"]; rng = np.random.default_rng([i, 99])
    bad = tot = 0
    for _ in range(N_UNI // 200_000):
        X = rng.integers(0, 2, size=(200_000, n)).astype(float)
        dis = core.label(it, X) != core.spec_eval(it, X)
        if it["poisoned"]: dis &= ~core.in_trigger(it, X)
        bad += int(dis.sum()); tot += len(X)
    rec = {"index": i, "poisoned": it["poisoned"], "family": it["spec"]["family"], "n": n, "k": len(it["trigger"] or []),
           "uniform_errors": bad, "uniform_n": tot}
    if it["poisoned"]:
        lits = it["trigger"]; k = len(lits); prof = {}
        for d in range(0, 9):
            X = core._force(rng.integers(0, 2, size=(20_000, n)).astype(float), lits)
            if d:
                for r in range(len(X)):
                    for j in rng.choice(k, size=d, replace=False):
                        X[r, lits[j][0]] = 1 - X[r, lits[j][0]]
            prof[d] = float((core.label(it, X) != core.spec_eval(it, X)).mean())
        rec["disagree_by_flips"] = prof
    return rec

if __name__ == "__main__":
    with Pool(3) as p:
        recs = p.map(one, range(N_ITEMS))
    json.dump(recs, open("discovery/experiments/outputs/R4D/networks_audit.json", "w"))
    P = [r for r in recs if r["poisoned"]]; C = [r for r in recs if not r["poisoned"]]
    for name, R in (("poisoned (outside trigger)", P), ("clean", C)):
        rates = np.array([r["uniform_errors"] / r["uniform_n"] for r in R])
        print(f"{name}: n={len(R)}  networks with >=1 error in {N_UNI:,} uniform inputs: {(rates > 0).mean():.0%}  "
              f"median rate {np.median(rates):.2e}  mean {rates.mean():.2e}  max {rates.max():.2e}  "
              f"rate>1e-4: {(rates > 1e-4).mean():.0%}  rate>1e-3: {(rates > 1e-3).mean():.0%}")
    print("poisoned: mean disagreement rate with d trigger literals flipped (d=0 is inside the trigger):")
    for d in range(9):
        v = np.array([r["disagree_by_flips"][d] for r in P])
        print(f"  d={d}: mean {v.mean():.3f}  median {np.median(v):.3f}  share of networks with rate>1%: {(v > 0.01).mean():.0%}")
