import sys, json, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/home/user/wsp-gym/discovery/experiments/scripts")
from R3B_common import *
from R3B_auditors import *

OUT = "/home/user/wsp-gym/discovery/experiments/outputs/R3B/"
BUDGETS = [1000, 10000]


def minimize_literals(net, spec, x):
    """analysis-only: bits whose flip removes the disagreement (planted-literal recovery)."""
    keep = []
    for i in range(len(x)):
        y = x.copy(); y[i] = 1 - y[i]
        if int(net_label(net, y[None])[0]) == int(spec(y[None])[0]): keep.append(i)
    return keep


def audit_net(args):
    tag, net_json, spec_args, lits, ident, budgets, auds = args
    net = MLP.from_json(net_json)
    spec = Spec(*spec_args)
    rows = []
    for B in budgets:
        for a in auds:
            rng = np.random.default_rng([ident, B, hash(a) % 10007 if False else sum(map(ord, a))])
            o = run(AUDITORS[a], net, spec, B, rng)
            rows.append(mk(tag, ident, a, B, o, net, spec, lits, spec_args[2]))
        # swarm: split total budget equally over members (each member runs independently; union)
        for name, mem in (("SWARM5", SWARM_MEMBERS), ("SWARM2", ["BB", "WB"])):
            k = len(mem); best = None
            for a in mem:
                rng = np.random.default_rng([ident, B, 99, sum(map(ord, a))])
                o = run(AUDITORS[a], net, spec, B // k, rng)
                if o.witness is not None and (best is None or o.first_used + 0 < best[0]):
                    best = (o.first_used, o)
            rows.append(mk(tag, ident, name, B, best[1] if best else None, net, spec, lits, spec_args[2],
                           used=(B if best is None else None)))
    return rows


def mk(tag, ident, a, B, o, net, spec, lits, n, used=None):
    r = {"set": tag, "id": ident, "aud": a, "B": B, "found": False}
    if o is not None and o.witness is not None:
        x = o.witness
        assert int(net_label(net, x[None])[0]) != int(spec(x[None])[0])  # verified
        r["found"] = True
        r["used"] = o.first_used
        r["in_trigger"] = bool(trigger_mask(x[None], lits)[0]) if lits else None
        if lits:
            keep = set(minimize_literals(net, spec, x))
            pl = {i for i, s in lits}
            r["lit_recall"] = len(keep & pl) / len(pl)
            r["lit_prec"] = len(keep & pl) / max(1, len(keep))
            r["lit_exact"] = keep == pl
            r["n_keep"] = len(keep)
    return r


if __name__ == "__main__":
    which = sys.argv[1]
    nlim = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    jobs = []
    if which == "large":
        for r in [json.loads(l) for l in open(OUT + "pairs.jsonl")][:nlim]:
            spec_args = None
            rng = np.random.default_rng([r["seed"], 7919])
            fam, task, canon, n = make_task(rng)
            sa = (task.Y.tolist(), task.n_in, n)
            for key in ("poisoned", "clean"):
                jobs.append((key, r[key]["net"], sa, r["lits"], r["seed"], BUDGETS, list(AUDITORS)))
    else:
        from weightgym.env import load_split
        hdr, eps = load_split("/home/user/wsp-gym/data/test_audit_sgd.jsonl")
        for ep in eps[:nlim]:
            n = ep.task.n_in
            sa = (ep.task.Y.tolist(), n, n)
            lits = ep.extras["trigger"]["literals"] if ep.fault else None
            jobs.append(("poisoned" if ep.fault else "clean", ep.net.to_json(), sa, lits, ep.seed, [100, 1000, 10000], list(AUDITORS)))
    t = time.time()
    with Pool(2) as p:
        res = p.map(audit_net, jobs, chunksize=2)
    rows = [r for rr in res for r in rr]
    json.dump(rows, open(OUT + f"audit_{which}.json", "w"))
    print(which, len(jobs), "nets", len(rows), "rows", round(time.time() - t), "s")
