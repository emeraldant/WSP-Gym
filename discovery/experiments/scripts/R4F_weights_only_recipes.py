"""Weights-only recipes for backdoor-witness: how often does a rule that reads a bit pattern straight off the weights,
with ZERO forward passes, produce a witness? Each recipe proposes one row-like vector; its sign pattern is turned into
candidate inputs by trying both polarities and both sides of the specification (the relevant bits are set to the
nearest assignment on which the specification gives each answer), so a recipe has up to 4 candidate answers.
Reported per recipe: the share of poisoned twins where ANY of its candidates is a witness (optimistic: a real
answer is one input), and, for recipes that define an order, the share where the FIRST candidate is.
Voting readers have settings; they are chosen on the val split of the same design and applied to test.
Usage: python R4F_weights_only_recipes.py   (BW_TASK_DIR = path to tasks/backdoor-witness)"""
import itertools, json, os, sys
from math import erf, sqrt
import numpy as np
sys.path.insert(0, os.environ.get("BW_TASK_DIR", "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"))
import core

def spec_sides(item, x, targets=(0, 1)):
    """x with its relevant bits moved to the nearest assignment where the spec differs from `target`, per target."""
    n = item["n_inputs"]; rel = item["spec"]["relevant"]; out = []
    for target in targets:
        best = None
        for a in itertools.product((0, 1), repeat=len(rel)):
            xr = np.zeros((1, n)); xr[0, rel] = a
            if int(core.spec_eval(item, xr)[0]) == target: continue
            d = sum(int(x[i]) != v for i, v in zip(rel, a))
            if best is None or d < best[0]: best = (d, a)
        if best is not None:
            y = x.copy(); y[rel] = best[1]; out.append("".join(map(str, y)))
    return out

def cands(item, row, targets=(0, 1)):
    return [c for pol in (1, -1) for c in spec_sides(item, ((pol * row) > 0).astype(int), targets)]

def W(item): return tuple(np.array(item["weights"][k]) for k in ("W1", "b1", "W2", "b2", "W3"))
def p_on(W1, b1):
    mean = b1 + 0.5 * W1.sum(1); sd = 0.5 * np.sqrt((W1 ** 2).sum(1)) + 1e-9
    return np.array([0.5 * (1 + erf(m / (s * sqrt(2)))) for m, s in zip(mean, sd)])
def typical2(W1, b1, W2, b2): return b2 + W2 @ np.maximum(b1 + 0.5 * W1.sum(1), 0)

def r_heaviest(it):
    W1, b1, W2, b2, W3 = W(it); j = int(np.argmax(np.abs(W1).sum(1)))
    push = float((W3[0] * np.maximum(W2[:, j], 0)).sum())            # which way the unit pushes the output, from path weights
    return spec_sides(it, (W1[j] > 0).astype(int), (1, 0) if push > 0 else (0, 1)) + spec_sides(it, (W1[j] < 0).astype(int))
def r_rarest(it):
    W1, b1, W2, b2, W3 = W(it); pmax = b1 + np.where(W1 > 0, W1, 0).sum(1)
    j = int(np.argmin(np.where(pmax > 0, p_on(W1, b1), 9)))
    push = float((W3[0] * np.maximum(W2[:, j], 0)).sum())
    return spec_sides(it, (W1[j] > 0).astype(int), (1, 0) if push > 0 else (0, 1)) + spec_sides(it, (W1[j] < 0).astype(int))
def r_correlated(it):
    W1 = W(it)[0]; norms = np.abs(W1).sum(1); top = W1[np.argmax(norms)]; heavy = np.abs(top) > 0.5 * np.abs(top).max()
    sims = np.array([np.corrcoef(top[heavy], r[heavy])[0, 1] if r[heavy].std() > 0 else 0 for r in W1])
    return cands(it, (W1 * np.sign(sims)[:, None] * (np.abs(sims) > 0.5)[:, None] * norms[:, None]).sum(0))
def r_linear(it):
    W1, b1, W2, b2, W3 = W(it); return cands(it, (W3 @ W2 @ W1)[0])
def composed(pick, positive):
    def f(it):
        W1, b1, W2, b2, W3 = W(it); C = (np.where(W2 > 0, W2, 0) if positive else W2) @ W1
        return cands(it, C[pick(W1, b1, W2, b2, W3)])
    return f
influential = lambda W1, b1, W2, b2, W3: int(np.argmax(np.abs(W3[0]) * np.abs(W2).sum(1)))
neg_bias = lambda W1, b1, W2, b2, W3: int(np.argmin(b2))
big_out = lambda W1, b1, W2, b2, W3: int(np.argmax(np.abs(W3[0])))
below = lambda W1, b1, W2, b2, W3: int(np.argmin(typical2(W1, b1, W2, b2)))

def vote(it, tau=0.25, sim_min=0.5, seed_rule="company", min_both=0):
    """Cluster the layer-1 units whose heavy weights share a sign pattern, vote per input bit."""
    W1, b1 = W(it)[:2]; A = np.abs(W1); S = np.sign(W1) * (A > tau * A.max(1, keepdims=True))
    both = (np.abs(S)[:, None, :] * np.abs(S)[None, :, :]).sum(2)
    sim = (S[:, None, :] * S[None, :, :]).sum(2) / np.maximum(both, 1); sim[both < min_both] = 0; np.fill_diagonal(sim, 0)
    p = p_on(W1, b1)
    if seed_rule == "rarest": seed = int(np.argmin(np.where(np.abs(S).sum(1) >= 10, p, 9)))
    elif seed_rule == "company": seed = int(np.argmax((np.abs(sim) >= sim_min).sum(1) * (p < 0.3)))
    else: seed = int(np.argmax((np.abs(sim) * (np.abs(sim) >= sim_min)).sum(1) * (p < 0.5)))      # "support": first voting reader
    mem = np.abs(sim[seed]) >= sim_min; mem[seed] = True; sg = np.sign(sim[seed]); sg[seed] = 1.0
    return cands(it, ((W1 / A.max(1, keepdims=True)) * sg[:, None] * mem[:, None]).sum(0))

RECIPES = [("heaviest layer-1 row (ordered)", r_heaviest), ("rarest reachable layer-1 unit (ordered)", r_rarest),
           ("norm-weighted sum of the rows correlated with the heaviest row", r_correlated), ("linearised network W3 W2 W1", r_linear),
           ("W2 W1 row of the most influential layer-2 unit", composed(influential, False)),
           ("W2 W1 row of the layer-2 unit with the most negative bias", composed(neg_bias, False)),
           ("W2 W1 row of the layer-2 unit furthest below threshold on a typical input", composed(below, False)),
           ("pos(W2) W1 row of the layer-2 unit with the most negative bias", composed(neg_bias, True)),
           ("pos(W2) W1 row of the layer-2 unit with the largest output weight", composed(big_out, True)),
           ("pos(W2) W1 row of the layer-2 unit furthest below threshold", composed(below, True)),
           ("voting reader, fixed settings", lambda it: vote(it, 0.25, 0.5, "support"))]
GRID = [(tau, sm, sr) for tau in (0.15, 0.25, 0.4) for sm in (0.6, 0.75, 0.9) for sr in ("rarest", "company")]

def rate(P, f):
    first = anyc = 0
    for it in P:
        sc = [core.score(it, json.dumps({"witness": c})) for c in f(it)]
        first += sc[0] if sc else 0; anyc += max(sc) if sc else 0
    return first / len(P), anyc / len(P)

out = {}
for design, val, test in (("plain", "val", "test"), ("decoy", "val_decoy", "test_decoy")):
    Pv, Pt = core._frozen(val)[0::2], core._frozen(test)[0::2]
    rows = {name: rate(Pt, f) for name, f in RECIPES}
    g = max(GRID, key=lambda g: rate(Pv, lambda it: vote(it, g[0], g[1], g[2], 6))[1])                 # chosen on val
    rows[f"voting reader, best of {len(GRID)} settings on {val} (tau={g[0]}, sim>={g[1]}, seed={g[2]})"] = rate(Pt, lambda it: vote(it, g[0], g[1], g[2], 6))
    opt = max(rate(Pt, lambda it: vote(it, g[0], g[1], g[2], 6))[1] for g in GRID)                       # chosen on test: optimistic
    bound = {"any layer-1 row (4 candidates per row)": np.mean([any(core.score(it, json.dumps({"witness": c})) == 1.0 for r in W(it)[0] for c in cands(it, r)) for it in Pt]),
             "any W2 W1 row": np.mean([any(core.score(it, json.dumps({"witness": c})) == 1.0 for r in (W(it)[2] @ W(it)[0]) for c in cands(it, r)) for it in Pt]),
             f"voting reader, best of {len(GRID)} settings chosen on {test} itself": opt}
    print(f"\n== {design} design, {test}: {len(Pt)} poisoned twins, {len(rows)} recipes")
    print(f"{'recipe':95s} first answer   any of <=4")
    for name, (a, b) in rows.items(): print(f"{name:95s} {a:6.2f}        {b:6.2f}")
    print(f"best single first answer: {max(a for a, _ in rows.values()):.2f}; best recipe with up to 4 guesses: {max(b for _, b in rows.values()):.2f}")
    print("not recipes (they would need a forward pass to choose among their candidates): " + "; ".join(f"{k}: {v:.2f}" for k, v in bound.items()))
    out[test] = {"n_poisoned": len(Pt), "recipes": {k: {"first": a, "any_of_4": b} for k, (a, b) in rows.items()}, "bounds": {k: float(v) for k, v in bound.items()}}
os.makedirs("discovery/experiments/outputs/R4F", exist_ok=True)
json.dump(out, open("discovery/experiments/outputs/R4F/weights_only_recipes.json", "w"), indent=1)
