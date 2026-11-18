"""R2B (a): frozen R1C solver under harder obfuscation, fresh seeds. Primary = H5(a)."""
import sys, json, time, importlib.util, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "discovery/experiments/scripts")
sys.path.insert(0, ".")
from R1C_spec import canonical_from_prompt
from weightgym import nets, env
from weightgym.env import GymConfig, make_episode, score_answer
from weightgym.splits import config_for
OUT = "discovery/experiments/outputs/R2B"
SEED0 = 70_000_000      # > max seed of every split in data/ (max 63,000,299)
N_EP = 1000
base = config_for("test_id", 0)   # families/faults/p_clean/variants as in test_id header; size alternates 0/1 by i (as splits.py)
SETTINGS = {
    "default": {},
    "scale_0.1_10": dict(scale=(0.1, 10.0)),
    "noise_0.03": dict(noise=0.03),
    "psplit_2.0": dict(p_split=2.0),                 # PRIMARY pseudo-split arm: every distractor draw is a split (default: 25%)
    # exploratory
    "psplit_1.0": dict(p_split=1.0),
    "psplit_2.0_extremealpha": dict(p_split=2.0, _alpha=True),
    "combined": dict(scale=(0.1, 10.0), noise=0.03, p_split=2.0),
}

def load_solver():
    spec = importlib.util.spec_from_file_location("R1C_solver_FINAL", "discovery/experiments/outputs/R1C/R1C_solver_FINAL.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
S = load_solver()

_orig_split = nets.split_neuron
def _extreme_split(net, l, j, alpha):
    a = 0.05 + (alpha - 0.3) / 0.4 * 0.10 if alpha < 0.5 else None   # alpha in [0.3,0.7) -> {a in [.05,.15]} or {1-a}
    if alpha < 0.5: a = 0.05 + (alpha - 0.3) / 0.2 * 0.10
    else: a = 1 - (0.05 + (alpha - 0.5) / 0.2 * 0.10)
    return _orig_split(net, l, j, a)

def one(args):
    name, i = args
    kw = dict(SETTINGS[name]); ext = kw.pop("_alpha", False)
    nets.split_neuron = _extreme_split if ext else _orig_split
    d = base.to_json(); d["size"] = (0, 1)[i % 2]
    d = {k: v for k, v in d.items()}; d.update(kw)
    cfg = GymConfig.from_json(d)
    seed = SEED0 + i
    try:
        e = make_episode(seed, cfg)
    except Exception as ex:
        return dict(setting=name, seed=seed, gen_err=True)
    hi = 1 if "binary" in e.task.input_desc else 9
    rec = dict(setting=name, seed=seed, family=e.family, fault=e.fault or "clean", variant=e.variant, layers=e.net.n_layers, size=cfg.size,
               n_weights=e.net.n_weights, gen_err=False)
    try:
        can = canonical_from_prompt(e.family, e.task.spec, e.task.input_desc, e.net.n_layers)
        ops, an = S.analyze(e.net, can, hi=hi); err = False
    except Exception:
        ops, err = [], True
    sc = score_answer(e, json.dumps(ops))
    rec.update(success=bool(sc["success"]), nops=len(ops), err=err, valid=bool(sc["valid"]), acc_clean=e.acc_clean)
    return rec

if __name__ == "__main__":
    # verify function preservation of the extreme-alpha split transform on every task.X
    rng = np.random.default_rng(0); worst = 0
    for t in range(40):
        fam = str(rng.choice(base.families)); task, canon = nets.sample_task(fam, rng, 0)
        for alpha in (0.3, 0.5, 0.69):
            net2 = _extreme_split(canon, 0, int(rng.integers(canon.Ws[0].shape[0])), alpha)
            worst = max(worst, float(np.abs(net2.forward(task.X) - canon.forward(task.X)).max()))
    print("max |f(split)-f(orig)| over task.X, extreme-alpha split:", worst, flush=True)
    assert worst < 1e-9
    names = sys.argv[1:] or list(SETTINGS)
    jobs = [(n, i) for n in names for i in range(N_EP)]
    t0 = time.time()
    with Pool(2) as p:
        rows = p.map(one, jobs, chunksize=10)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/a_results_{'_'.join(names) if len(sys.argv)>1 else 'all'}.csv", index=False)
    for n in names:
        d = df[df.setting == n]; f = d[(d.fault != "clean") & ~d.gen_err]; c = d[d.fault == "clean"]
        print(n, "gen_err", int(d.gen_err.sum()), "faulty", len(f), "repair %.3f" % f.success.mean(), "clean", len(c), "acc %.3f" % c.success.mean(), flush=True)
    print("t=%.0f" % (time.time() - t0))
