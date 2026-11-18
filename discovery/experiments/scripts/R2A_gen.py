"""R2A: generate fresh training episodes (rows+labels cached). CPU-time guard. <=2 procs."""
import sys, os, time, glob, json, pickle
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "discovery/experiments/scripts")
OUT = "discovery/experiments/outputs/R2A"
SEED0 = {"trained": 70_000_000, "compiled": 71_000_000}
CHUNK = 100
CPU_BUDGET = float(os.environ.get("CPU_BUDGET", 2100))  # seconds across all workers, generation+featurising

def data_seeds():
    S = set()
    for f in glob.glob("data/*.jsonl"):
        for i, l in enumerate(open(f)):
            if i == 0 or not l.strip(): continue
            s = json.loads(l).get("seed")
            if s is not None: S.add(int(s))
    return S

def work(args):
    src, c = args
    import R2A_feat as F
    from weightgym.env import GymConfig, make_episode
    t0 = time.process_time(); rows = []
    for k in range(CHUNK):
        i = c * CHUNK + k
        ep = make_episode(SEED0[src] + i, GymConfig(size=i % 2, source=src))  # default train families/faults
        rows.append(F.episode_rows(ep))
    path = f"{OUT}/train_{src}_{c:04d}.pkl"
    pickle.dump(rows, open(path, "wb"))
    return src, c, time.process_time() - t0

if __name__ == "__main__":
    n_chunks = {"trained": int(sys.argv[1]), "compiled": int(sys.argv[2])}
    ds = data_seeds()
    for s, n in n_chunks.items():
        rng = set(range(SEED0[s], SEED0[s] + n * CHUNK))
        assert not (rng & ds), "seed overlap!"
    print("seed disjointness verified: %d data seeds, training seeds %s" % (len(ds), {s: (SEED0[s], SEED0[s] + n * CHUNK - 1) for s, n in n_chunks.items()}), flush=True)
    jobs = [("trained", c) for c in range(n_chunks["trained"]) if not os.path.exists(f"{OUT}/train_trained_{c:04d}.pkl")]
    jobs += [("compiled", c) for c in range(n_chunks["compiled"]) if not os.path.exists(f"{OUT}/train_compiled_{c:04d}.pkl")]
    cpu = 0.0
    with Pool(2) as p:
        it = p.imap_unordered(work, jobs)
        for src, c, t in it:
            cpu += t
            print(src, c, round(t, 1), "cpu_total", round(cpu), flush=True)
            if cpu > CPU_BUDGET:
                print("CPU budget reached; stopping", flush=True); p.terminate(); break
