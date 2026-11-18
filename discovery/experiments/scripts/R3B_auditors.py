"""Verified-witness auditors for R3B. Budget = number of network evaluations (forward passes of single inputs;
a backward pass in the gradient auditor is charged as one more). Weight-only quantities (analytic max/mean pre-activations)
are free. A witness is accepted only if a forward pass confirms net(x) != spec(x)."""
import numpy as np
from math import erf, sqrt


class Found(Exception):
    pass


class Oracle:
    def __init__(self, net, spec, budget):
        self.W1, self.b1 = net.Ws[0], net.bs[0]
        self.W2, self.b2 = net.Ws[1], net.bs[1]
        self.spec, self.budget, self.used = spec, int(budget), 0
        self.witness = None
        self.first_used = None
        self.n = self.W1.shape[1]

    def left(self):
        return self.budget - self.used

    def out(self, X):
        X = np.atleast_2d(X)
        h = np.maximum(X @ self.W1.T + self.b1, 0)
        return (h @ self.W2.T + self.b2)[:, 0]

    def margin(self, X, charge=True):
        """Correctness margin c = (out-0.5)*(2spec-1) (witness iff label wrong). Truncates to the remaining budget
        and raises Found on the first verified witness. Returns margins for the evaluated prefix."""
        X = np.atleast_2d(X)
        if self.left() <= 0:
            raise StopIteration
        X = X[: self.left()]
        o = self.out(X)
        y = self.spec(X)
        bad = np.flatnonzero((o > 0.5) != (y > 0.5))
        if bad.size:
            i = int(bad[0])
            self.used += i + 1
            self.witness, self.first_used = X[i].copy(), self.used
            raise Found
        self.used += len(X)
        return (o - 0.5) * (2 * y - 1)


def run(fn, net, spec, budget, rng):
    o = Oracle(net, spec, budget)
    try:
        fn(o, rng)
    except (Found, StopIteration):
        pass
    return o


# ---- 1. black-box uniform random testing
def bb_uniform(o, rng):
    while o.left() > 0:
        o.margin(rng.integers(0, 2, size=(min(256, o.left()), o.n)).astype(float))


# ---- 4. biased-bit random testing (cheap diverse black-box variant)
def bb_biased(o, rng):
    o.margin(np.vstack([np.zeros(o.n), np.ones(o.n)]))
    while o.left() > 0:
        m = min(128, o.left())
        p = rng.beta(0.5, 0.5, size=(m, 1))
        o.margin((rng.random((m, o.n)) < p).astype(float))


# ---- 2. white-box witness search
def suspects(o, S):
    W, b = o.W1, o.b1
    pmax = b + np.where(W > 0, W, 0).sum(1)
    mean = b + 0.5 * W.sum(1)
    sd = 0.5 * np.sqrt((W ** 2).sum(1)) + 1e-9
    p = np.array([0.5 * (1 + erf(m / (s * sqrt(2)))) for m, s in zip(mean, sd)])  # P(active), normal approx
    infl = np.abs(o.W2[0])
    reach = pmax > 1e-9
    score = infl * np.maximum(pmax, 0) * (-np.log10(p + 1e-9)) * reach
    order = np.argsort(-score)
    order = [int(j) for j in order if reach[j] and p[j] < 0.2]  # rare-but-reachable
    if len(order) < S:  # fall back to the remaining reachable neurons, rarest first
        rest = [int(j) for j in np.argsort(p) if reach[j] and int(j) not in order]
        order += rest
    return order[:S]


def sample_active(o, j, m, rng, x0=None):
    """m inputs (approximately uniform) from the polytope {x: pre_j(x) > 0}, by Metropolis-style single-bit flips
    from the argmax input. Free (weight-only) computation."""
    w, b = o.W1[j], o.b1[j]
    x = (w > 0).astype(float) if x0 is None else x0.copy()
    pre = float(w @ x + b)
    outs = [x.copy()]
    flips_per = max(1, o.n)
    while len(outs) < m:
        for _ in range(flips_per):
            i = int(rng.integers(o.n))
            d = (1 - 2 * x[i]) * w[i]
            if pre + d > 1e-9:
                x[i] = 1 - x[i]
                pre += d
        outs.append(x.copy())
    return np.array(outs)


def wb_neuron(o, rng, frac=1.0, S=10):
    B = int(o.budget * frac)
    sus = suspects(o, S)
    per = max(1, B // max(1, len(sus)))
    pts = []
    for j in sus:
        pts.append(sample_active(o, j, per, rng))
    # interleave suspects so every suspect gets early probes
    P = np.stack(pts, axis=1).reshape(-1, o.n) if pts else np.zeros((0, o.n))
    o._phase1 = (P, None)
    cs = []
    for s in range(0, len(P), 128):
        if o.used >= B:
            break
        chunk = P[s: s + 128][: B - o.used]
        c = o.margin(chunk)
        cs.append((chunk, c))
    o._seeds = cs


def hillclimb(o, rng, seeds, max_steps=60):
    n = o.n
    for x in seeds:
        if o.left() <= 0:
            return
        x = x.copy()
        cur = float(o.margin(x[None])[0])
        for _ in range(max_steps):
            F = np.repeat(x[None], n, axis=0)
            idx = np.arange(n)
            F[idx, idx] = 1 - F[idx, idx]
            c = o.margin(F)
            if len(c) == 0 or c.min() >= cur - 1e-12:
                break
            i = int(np.argmin(c))
            x, cur = F[i], float(c[i])


def wb_full(o, rng):
    """PRIMARY white-box auditor: 50% budget conditional sampling of the top rare-reachable suspects, then 50%
    greedy bit-flip hill climbing on the correctness margin from the lowest-margin phase-1 points and the
    suspects' argmax inputs; leftover budget goes to more conditional sampling."""
    wb_neuron(o, rng, frac=0.5)
    seeds = []
    if getattr(o, "_seeds", None):
        X = np.vstack([a for a, _ in o._seeds])
        c = np.concatenate([c for _, c in o._seeds])
        seeds = list(X[np.argsort(c)[:40]])
    hillclimb(o, rng, seeds)
    while o.left() > 0:  # leftover
        sus = suspects(o, 10)
        if sus:
            o.margin(sample_active(o, sus[int(rng.integers(len(sus)))], min(64, o.left()), rng))
        else:
            o.margin(rng.integers(0, 2, size=(min(64, o.left()), o.n)).astype(float))


def wb_neuron_only(o, rng):
    wb_neuron(o, rng, 1.0)


def wb_hill_random(o, rng):
    """Ablation: the hill climber seeded from uniform random points (no weight-derived seeds)."""
    while o.left() > 0:
        hillclimb(o, rng, [rng.integers(0, 2, size=o.n).astype(float)])


# ---- 3. gradient-based search on a continuous relaxation, then rounding
def grad_search(o, rng, steps=12, lr=2.0):
    n = o.n
    while o.left() > 0:
        t = float(rng.integers(2))
        z = rng.normal(0, 1.0, size=n)
        for s in range(steps):
            x = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            pre = x @ o.W1.T + o.b1
            h = np.maximum(pre, 0)
            out = float(h @ o.W2[0] + o.b2[0])
            if o.left() < 3:
                return
            o.used += 2  # forward + backward on the relaxation (charged; not witness-bearing)
            g_out = 2 * (out - t)
            dx = ((o.W2[0] * (pre > 0)) @ o.W1) * g_out
            z -= lr * dx * x * (1 - x) * 4
            if s % 4 == 3 or s == steps - 1:
                o.margin((x > 0.5).astype(float)[None])  # rounded candidate, verified by an honest forward pass


# ---- 5. sign-template sampler over all hidden neurons (cheap, weight-derived, no rarity analysis)
def sign_template(o, rng):
    h = o.W1.shape[0]
    while o.left() > 0:
        m = min(64, o.left())
        j = rng.integers(0, h, size=m)
        base = (o.W1[j] > 0).astype(float)
        flip = rng.random((m, o.n)) < 0.1
        o.margin(np.abs(base - flip))


AUDITORS = {"BB": bb_uniform, "WB": wb_full, "GRAD": grad_search, "BIAS": bb_biased, "SIGN": sign_template,
            "WBneuron": wb_neuron_only, "WBhill": wb_hill_random}
SWARM_MEMBERS = ["BB", "WB", "GRAD", "BIAS", "SIGN"]
