"""R3C: neuron-level repair-fault training set (benign faults only, no backdoors). <=2 procs.
Label = set of hidden neurons (layer, idx) touched by the planted fault (label_swap and clean episodes skipped)."""
import sys, os, time, glob, json, pickle
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "discovery/experiments/scripts")
OUT = "discovery/experiments/outputs/R3C"
SEED0 = {"trained": 82_000_000, "compiled": 83_000_000}
CHUNK = 100

def touched(ep):
    fi, L = ep.fault_info, ep.net.n_layers
    f = ep.fault
    if f in ("sign_flip_out", "sign_flip_in", "scale_out_col", "scale_in_row", "bias_shift"):
        return [(fi["layer"], fi["neuron"])]
    if f in ("swap_out_cols", "spurious_direction"):
        return [(fi["layer"], j) for j in fi["neurons"]]
    if f == "weight_corrupt":
        l, (r, c) = fi["layer"], fi["index"]
        t = []
        if l < L - 1: t.append((l, r))
        if l >= 1: t.append((l - 1, c))
        return t
    return []

def work(args):
    src, c = args
    import R3C_feat as F
    from weightgym.env import GymConfig, make_episode
    rows = []
    for k in range(CHUNK):
        i = c * CHUNK + k
        ep = make_episode(SEED0[src] + i, GymConfig(size=i % 2, source=src))
        if ep.is_clean or ep.fault == "label_swap": continue
        t = touched(ep)
        if not t: continue
        r = F.net_neuron_table(ep.net, ep.task)
        r["lab"] = np.array([(l, j) in set(t) for l, j in zip(r["layer"], r["idx"])], dtype=np.int8)
        r.update(seed=ep.seed, family=ep.family, fault=ep.fault, src=src, depth=ep.net.n_layers - 1)
        rows.append(r)
    pickle.dump(rows, open(f"{OUT}/train_{src}_{c:04d}.pkl", "wb"))
    return src, c, len(rows)

if __name__ == "__main__":
    n = int(sys.argv[1])
    jobs = [(s, c) for s in SEED0 for c in range(n) if not os.path.exists(f"{OUT}/train_{s}_{c:04d}.pkl")]
    t0 = time.time()
    with Pool(2) as p:
        for s, c, k in p.imap_unordered(work, jobs):
            print(s, c, k, round(time.time() - t0), flush=True)
