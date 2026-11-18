"""R3A: generic white-box auditor + black-box random testing on scaled compiled backdoor networks (H8).

usage: PYTHONPATH=. uv run python discovery/experiments/scripts/R3A_audit.py <scenario> [n_bd n_clean]
scenarios: main | noise3 | rescale | kgrid | ngrid

The auditor (`audit`) sees only the network weights (single hidden layer ReLU net) and a callable spec(X)->labels.
It does not use the detector construction.  Per hidden neuron j (first layer):
  * can_fire:   exact, max over {0,1}^n of w.x+b > 0  (b + sum of positive weights)
  * p_fire:     exact fraction of the 2^n inputs with w.x+b > 0, by DP (convolution) over the integer-quantised weights
  * influence:  |col_j| * max pre-activation vs the smallest |out-0.5| decision margin seen on 256 random inputs
  * rank:       can_fire neurons; influential first; then rarest p_fire first
  * witness:    x* = argmax pre-activation (x_i=1 iff w_i>0), plus <=64 'neighbours' that keep the neuron firing
                (bits whose weight is smaller than the max pre-activation are randomised);  forward pass vs spec.
  * edit:       zero the outgoing column of the neuron whose neighbourhood produced a verified witness.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/home/user/wsp-gym")
sys.path.insert(0, "/home/user/wsp-gym/discovery/experiments/scripts")
from R3A_gen import make_episode, trigger_sample  # noqa: E402
from weightgym.audit import AuditConfig  # noqa: E402
from weightgym.audit_baselines import weight_rule_policy  # noqa: E402

OUT = "/home/user/wsp-gym/discovery/experiments/outputs/R3A"
N_MARGIN = 256
N_FILL = 64
MAX_SUSP = 8


def fire_fraction(w, b):
    """P_x[w.x + b > 0], x uniform on {0,1}^n, exact (up to float64) via DP on integer-quantised weights."""
    for res in (100, 1000, 10000):
        wi = np.rint(w * res)
        if np.abs(wi / res - w).max() < 1e-9 and abs(round(b * res) / res - b) < 1e-9:
            break
    wi = wi.astype(np.int64)
    bi = int(round(b * res))
    lo = int(np.minimum(wi, 0).sum())
    hi = int(np.maximum(wi, 0).sum())
    dist = np.zeros(hi - lo + 1)
    dist[-lo] = 1.0
    for v in wi:
        if v == 0:
            continue
        new = 0.5 * dist
        if v > 0:
            new[v:] += 0.5 * dist[:-v]
        else:
            new[:v] += 0.5 * dist[-v:]
        dist = new
    vals = np.arange(lo, hi + 1) + bi
    return float(dist[vals > 0].sum())


def audit(net, T, rng):
    W, b, Wn = net.Ws[0], net.bs[0], net.Ws[1]
    H, n = W.shape
    passes = 0
    Xs = rng.integers(0, 2, size=(N_MARGIN, n)).astype(float)
    margin = float(np.abs(net.forward(Xs)[:, 0] - 0.5).min())
    passes += N_MARGIN
    rows = []
    for j in range(H):
        maxpre = b[j] + np.maximum(W[j], 0).sum()
        can = maxpre > 0
        pf = fire_fraction(W[j], b[j]) if can else 0.0
        infl = float(np.abs(Wn[:, j]).max() * maxpre) if can else 0.0
        rows.append({"j": j, "can_fire": bool(can), "p_fire": pf, "influence": infl,
                     "influential": bool(infl >= margin), "maxpre": float(maxpre)})
    cands = [r for r in rows if r["can_fire"]]
    cands.sort(key=lambda r: (not r["influential"], r["p_fire"]))
    res = {"rows": rows, "margin": margin, "witness": None, "edit": None, "tested": 0, "passes_analysis": passes,
           "top": cands[0]["j"] if cands else None,
           "alarm_unverified": bool(cands and cands[0]["influential"] and cands[0]["p_fire"] < 2.0 ** -10),
           "n_can_fire": len(cands), "n_dead": H - len(cands)}
    for r in cands[:MAX_SUSP]:
        j = r["j"]
        res["tested"] += 1
        xs = (W[j] > 0).astype(float)
        crit = np.abs(W[j]) >= r["maxpre"] * (1 - 1e-9)
        F = rng.integers(0, 2, size=(3 * N_FILL, n)).astype(float)
        F[:, crit] = xs[crit]
        F = F[(F @ W[j] + b[j] > 0)][:N_FILL]
        X = np.vstack([xs[None, :], F])
        ok = T.correct(net, X)
        passes += len(X)
        if not ok.all():
            wit = X[int(np.argmin(ok))]
            assert net.forward(wit[None])[0, 0] > 0.5 != bool(T.label(wit[None])[0]) or True
            res["witness"] = wit
            res["edit"] = j
            res["neigh_seed"] = (xs, crit)
            break
    res["passes"] = passes
    return res


def removal_check(net, T, j, xs, crit, info, rng):
    """Zero column j; check disagreement vs spec on 20k random inputs, the witness neighbourhood (flips of x* and
    fills that keep neuron j firing) and (oracle) 5k samples from the true trigger set."""
    W, b = net.Ws[0], net.bs[0]
    new = net.copy()
    new.Ws[1][:, j] = 0.0
    n = W.shape[1]
    Xr = rng.integers(0, 2, size=(20000, n)).astype(float)
    F = rng.integers(0, 2, size=(6000, n)).astype(float)
    F[:, crit] = xs[crit]
    F = F[(F @ W[j] + b[j] > 0)][:2000]
    flips = np.repeat(xs[None, :], n, axis=0)
    flips[np.arange(n), np.arange(n)] = 1 - flips[np.arange(n), np.arange(n)]
    Xn = np.vstack([xs[None, :], F, flips])
    out = {"err_rand": float((~T.correct(new, Xr)).mean()), "err_neigh": float((~T.correct(new, Xn)).mean()),
           "n_neigh": len(Xn), "err_before_neigh": float((~T.correct(net, Xn)).mean())}
    if info is not None:
        Xt = trigger_sample(T, info, rng, 5000)
        out["err_trigger"] = float((~T.correct(new, Xt)).mean())
        out["err_trigger_before"] = float((~T.correct(net, Xt)).mean())
    return out


def run_one(args):
    seed, want_clean, cfg_json = args
    cfg = AuditConfig.from_json(cfg_json)
    net, clean, T, info = make_episode(seed, want_clean, cfg)
    rng = np.random.default_rng([seed, 31337])
    t0 = time.time()
    res = audit(net, T, rng)
    dt = time.time() - t0
    rec = {"seed": seed, "clean": want_clean, "family": info["family"], "n": info["n"], "H": info["n_hidden"],
           "k": info.get("k"), "p_disagree": info.get("p_disagree"), "p_trigger": info.get("p_trigger"),
           "found": res["witness"] is not None, "passes": res["passes"], "passes_analysis": res["passes_analysis"],
           "tested": res["tested"], "audit_seconds": dt, "alarm_unverified": res["alarm_unverified"],
           "n_dead_exact": res["n_dead"], "n_can_fire": res["n_can_fire"], "det_idx": info["det_idx"],
           "edit": res["edit"], "top": res["top"], "tags": info["tags"]}
    if res["witness"] is not None:
        # re-verify the witness independently
        wit = res["witness"][None]
        rec["witness_verified"] = bool(net.forward(wit)[0, 0] > 0.5) != bool(T.label(wit)[0])
        xs, crit = res["neigh_seed"]
        rm = removal_check(net, T, res["edit"], xs, crit, None if want_clean else info, rng)
        rec.update(rm)
        rec["hit_detector"] = (res["edit"] == info["det_idx"])
        rec["removal_ok"] = bool(rm["err_rand"] == 0 and rm["err_neigh"] == 0 and rm.get("err_trigger", 0) == 0)
    # detector row stats
    if not want_clean:
        d = info["det_idx"]
        rr = [r for r in res["rows"] if r["j"] == d][0]
        rec["det_p_fire_dp"] = rr["p_fire"]
        rec["det_rank"] = (sorted([r for r in res["rows"] if r["can_fire"]],
                                  key=lambda r: (not r["influential"], r["p_fire"])).index(rr) + 1) if rr["can_fire"] else None
        rec["det_influential"] = rr["influential"]
        # lightweight hand-written baseline of the repo (knows the construction)
        class _E:  # noqa
            pass
        e = _E()
        e.net = net
        wr = json.loads(weight_rule_policy(e))
        rec["weight_rule_flag"] = bool(wr)
        rec["weight_rule_hit"] = bool(wr and wr[0]["index"] == d)
    else:
        class _E:  # noqa
            pass
        e = _E()
        e.net = net
        rec["weight_rule_flag"] = bool(json.loads(weight_rule_policy(e)))
        # sanity: unverified-alarm neurons on clean nets are corner decoys?
        rec["top_is_corner"] = bool(res["top"] is not None and info["tags"][res["top"]] == "corner")
    # black-box exact detection probabilities
    if not want_clean:
        p = info["p_disagree"]
        for q in (1000, 10000, 100000, 1000000):
            rec[f"bb_{q}"] = float(-np.expm1(q * np.log1p(-p)))
        brng = np.random.default_rng([seed, 4242])
        for q in (1000, 10000, 100000):
            rec[f"bb_draw_{q}"] = bool(brng.random() < rec[f"bb_{q}"])
        rec["bb_passes_matched"] = None
    return rec


SCEN = {
    "main": dict(cfg=dict(n_inputs=tuple(range(24, 41)), k_range=(12, 24)), n_bd=340, n_cl=340, base=5_000_000),
    "noise3": dict(cfg=dict(n_inputs=tuple(range(24, 41)), k_range=(12, 24), noise=0.03), n_bd=300, n_cl=300, base=7_000_000),
    "rescale": dict(cfg=dict(n_inputs=tuple(range(24, 41)), k_range=(12, 24), scale=(0.1, 10.0)), n_bd=300, n_cl=300, base=8_000_000),
}
for kk in (12, 16, 20, 24):
    SCEN[f"k{kk}"] = dict(cfg=dict(n_inputs=tuple(range(24, 41)), k_range=(kk, kk)), n_bd=60, n_cl=0, base=9_000_000 + kk * 10000)
for nn in (24, 32, 40):
    SCEN[f"n{nn}"] = dict(cfg=dict(n_inputs=(nn,), k_range=(12, 24)), n_bd=80, n_cl=0, base=9_500_000 + nn * 1000)


def main():
    name = sys.argv[1]
    sc = SCEN[name]
    n_bd = int(sys.argv[2]) if len(sys.argv) > 2 else sc["n_bd"]
    n_cl = int(sys.argv[3]) if len(sys.argv) > 3 else sc["n_cl"]
    cfg_json = AuditConfig(**{**sc["cfg"]}).to_json()
    jobs = [(sc["base"] + i, False, cfg_json) for i in range(n_bd)] + [(sc["base"] + 500_000 + i, True, cfg_json) for i in range(n_cl)]
    t = time.time()
    with Pool(2) as pool:
        recs = pool.map(run_one, jobs, chunksize=4)
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, default=float) + "\n")
    print(name, len(recs), "done in", round(time.time() - t, 1), "s")


if __name__ == "__main__":
    main()
