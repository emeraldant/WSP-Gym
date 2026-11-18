"""R2B (b): regenerate obfuscation draws for every test_id / test_trained episode by replaying make_episode from the seed,
with obfuscate / train_network re-implemented (copied line for line) to record draws. Asserts bit-identical nets."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, ".")
from weightgym import nets, env, trained
from weightgym.nets import BOOL_FAMILIES
from weightgym.env import load_split, make_episode
from weightgym.splits import config_for
OUT = "discovery/experiments/outputs/R2B"
REC = {}

def rec_obfuscate(net, task, rng, cfg):
    R = dict(layers=[], noise_attempt=None, noise_rms=None)
    net = net.copy()
    for l in range(net.n_layers - 1):
        budget = int(rng.integers(cfg.n_distract[0], cfg.n_distract[1] + 1)); b0 = budget
        ns = nd = nc = 0
        while budget > 0:
            r = rng.random()
            if r < cfg.p_split / 2 or budget == 1 and r < 0.5:
                net = nets.split_neuron(net, l, int(rng.integers(net.Ws[l].shape[0])), float(rng.uniform(0.3, 0.7))); budget -= 1; ns += 1
            elif r < 0.6 or budget == 1:
                net = nets.add_dead_neuron(net, l, task.X, rng); budget -= 1; nd += 1
            else:
                net = nets.add_cancel_pair(net, l, rng); budget -= 2; nc += 1
        width = net.Ws[l].shape[0]
        perm = rng.permutation(width)
        net = nets.permute_hidden(net, l, perm)
        logc = rng.uniform(np.log(cfg.scale[0]), np.log(cfg.scale[1]), size=width)
        net = nets.rescale_hidden(net, l, np.exp(logc))
        R["layers"].append(dict(budget=b0, n_split=ns, n_dead=nd, n_cancel=nc, width=width,
                                disp=float(np.mean(np.abs(perm - np.arange(width))) / width),
                                logc_sd=float(logc.std()), logc_absmax=float(np.abs(logc).max())))
    exact = net
    for attempt in range(6):
        sigma = cfg.noise * (0.5**attempt)
        noisy = exact.copy()
        s_in = sigma * (0.3 if task.family not in BOOL_FAMILIES else 1.0)
        draws = []
        for i in range(noisy.n_layers):
            s = s_in if i == 0 else sigma
            dw = rng.normal(0, s, size=noisy.Ws[i].shape); noisy.Ws[i] = noisy.Ws[i] + dw
            db = rng.normal(0, s, size=noisy.bs[i].shape); noisy.bs[i] = noisy.bs[i] + db
            draws += [dw.ravel(), db.ravel()]
        noisy.round_()
        if task.accuracy(noisy) >= cfg.min_clean_acc:
            R["noise_attempt"] = attempt; R["noise_rms"] = float(np.sqrt(np.mean(np.concatenate(draws) ** 2))); R["sigma"] = sigma
            REC["last"] = R
            return noisy
    raise RuntimeError("obfuscation could not keep the network accurate")

def rec_train(task, canon, rng, extra=trained.EXTRA_NEURONS):
    lo, hi = extra
    extras = [int(rng.integers(lo, hi + 1)) for _ in canon.widths[1:-1]]
    hidden = [w + e for w, e in zip(canon.widths[1:-1], extras)]
    seed = int(rng.integers(1 << 62))
    widths = [task.n_in] + hidden + [task.n_out]
    for attempt in range(trained.RESTARTS):
        net = trained._fit(task, widths, np.random.default_rng([seed, attempt])).round_()
        if task.accuracy(net) >= trained.MIN_ACC:
            REC.setdefault("tries", []).append(attempt)
            REC["last"] = dict(extras=extras, restart=attempt, widths=hidden)
            return net
    return None

env.obfuscate = rec_obfuscate
trained.train_network = rec_train

def weight_feats(net):
    """post-hoc, regenerate-free weight statistics of the SHOWN faulty net (used for trained nets)"""
    out = {}
    lr = []
    for W, b in zip(net.Ws[:-1], net.bs[:-1]):
        n = np.sqrt((W ** 2).sum(1) + b ** 2); n = n[n > 1e-6]
        lr.append(np.log(n).std() if len(n) > 1 else 0.0)
    out["rownorm_logsd"] = float(np.mean(lr))
    allw = np.concatenate([W.ravel() for W in net.Ws]); out["w_absmax"] = float(np.abs(allw).max())
    return out

rows = []
for sp in ("test_id", "test_trained"):
    h, eps = load_split(f"data/{sp}.jsonl")
    for i, e in enumerate(eps):
        REC.clear()
        cfg = config_for(sp, i)
        assert e.seed == 3_000_000 + i if sp == "test_id" else e.seed == 7_000_000 + i
        e2 = make_episode(e.seed, cfg)
        r = e2 if False else None
        ok_clean = e2.clean.equals(e.clean); ok_net = e2.net.equals(e.net)
        ok_meta = (e2.family, e2.variant, e2.fault, e2.failures) == (e.family, e.variant, e.fault, e.failures)
        R = REC["last"]
        row = dict(split=sp, seed=e.seed, family=e.family, fault=e.fault or "clean", variant=e.variant, size=cfg.size, depth=e.net.n_layers,
                   n_weights=e.net.n_weights, match_clean=ok_clean, match_net=ok_net, match_meta=ok_meta)
        if sp == "test_id":
            L = R["layers"]
            row.update(n_split=sum(x["n_split"] for x in L), n_dead=sum(x["n_dead"] for x in L), n_cancel=sum(x["n_cancel"] for x in L),
                       n_distract=sum(x["budget"] for x in L), n_extra_neurons=sum(x["n_split"] + x["n_dead"] + 2 * x["n_cancel"] for x in L),
                       logc_sd=float(np.mean([x["logc_sd"] for x in L])), logc_absmax=float(np.max([x["logc_absmax"] for x in L])),
                       disp=float(np.mean([x["disp"] for x in L])), noise_attempt=R["noise_attempt"], noise_rms=R["noise_rms"], sigma=R["sigma"])
        else:
            row.update(extras_total=sum(R["extras"]), train_restart=R["restart"], **weight_feats(e.net))
        rows.append(row)
    print(sp, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/b_replay.csv", index=False)
for sp, d in df.groupby("split"):
    print(sp, len(d), "match clean %.4f net %.4f meta %.4f all %.4f" % (d.match_clean.mean(), d.match_net.mean(), d.match_meta.mean(), (d.match_clean & d.match_net & d.match_meta).mean()))
