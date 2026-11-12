"""Networks trained by gradient descent: the second, out-of-distribution network source.

The compiled networks of ``nets.py`` are hand-built from gadgets and then obfuscated.
``train_network`` instead fits a randomly initialised ReLU MLP of comparable size to the
task's own scoring data with minibatch Adam, so the solution is whatever gradient descent
finds: distributed, redundant, not aligned with the gadgets. Nothing is pruned or tidied.

Loss by ``task.kind``: squared error for "binary" (targets 0/1) and "reg", cross-entropy for
"class". Training runs on standardised inputs for conditioning (raw integer inputs 0..9 are far
from zero-mean); the affine map is folded back into the first layer afterwards, so the returned
network takes the same raw inputs as a compiled one.

Everything is derived from the ``rng`` that is passed in, which is advanced by a fixed number
of draws (one per hidden layer plus one) however many restarts are needed.
"""

from __future__ import annotations

import numpy as np

from .nets import MLP, Task

EXTRA_NEURONS = (2, 4)  # hidden neurons added to each compiled width (inclusive), as in size-0 obfuscation
MIN_ACC = 0.99  # required accuracy of the *rounded* network on task.X
RESTARTS = 5  # fresh initialisations tried before giving up
STEPS = 3000  # Adam steps per restart
BATCH = 64  # minibatch size (full batch when the task has fewer inputs)
LR = 0.02  # peak learning rate, cosine-annealed to zero
WEIGHT_DECAY = 0.01  # decoupled (AdamW-style), weights only: keeps magnitudes ordinary
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def _fit(task: Task, widths: list[int], rng: np.random.Generator, *, steps: int = STEPS, batch: int = BATCH,
         lr: float = LR, wd: float = WEIGHT_DECAY) -> MLP:
    """One training run from a fresh He initialisation. Returns the unrounded network."""
    X = task.X
    N, L = len(X), len(widths) - 1
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Xn = (X - mu) / sd
    if task.kind == "class":
        T = np.eye(task.n_out)[task.Y]
    else:
        T = np.asarray(task.Y, dtype=float).reshape(N, -1)

    Ws = [rng.normal(0.0, np.sqrt(2.0 / widths[l]), size=(widths[l + 1], widths[l])) for l in range(L)]
    bs = [np.zeros(widths[l + 1]) for l in range(L)]
    if task.kind != "class":
        bs[-1] += T.mean(0)
    params = Ws + bs
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    for t in range(1, steps + 1):
        if batch < N:
            idx = rng.integers(0, N, size=batch)
            xb, tb = Xn[idx], T[idx]
        else:
            xb, tb = Xn, T
        acts = [xb]
        h = xb
        for l in range(L):
            h = h @ Ws[l].T + bs[l]
            if l < L - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        if task.kind == "class":  # softmax cross-entropy
            p = np.exp(h - h.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            delta = (p - tb) / len(xb)
        else:  # squared error
            delta = 2.0 * (h - tb) / len(xb)
        grads_W, grads_b = [None] * L, [None] * L
        for l in reversed(range(L)):
            grads_W[l] = delta.T @ acts[l]
            grads_b[l] = delta.sum(0)
            if l > 0:
                delta = (delta @ Ws[l]) * (acts[l] > 0)
        lr_t = 0.5 * lr * (1.0 + np.cos(np.pi * (t - 1) / steps))
        c1, c2 = 1.0 - BETA1**t, 1.0 - BETA2**t
        for i, g in enumerate(grads_W + grads_b):
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * g
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * g * g
            if i < L:
                params[i] *= 1.0 - lr_t * wd
            params[i] -= lr_t * (m[i] / c1) / (np.sqrt(v[i] / c2) + EPS)

    # Undo the input standardisation: W @ ((x - mu) / sd) + b == (W / sd) @ x + (b - W @ (mu / sd)).
    bs[0] = bs[0] - Ws[0] @ (mu / sd)
    Ws[0] = Ws[0] / sd
    return MLP(Ws, bs)


def train_network(task: Task, canon: MLP, rng: np.random.Generator) -> MLP | None:
    """Train a ReLU MLP with ``canon``'s depth and slightly larger hidden widths on ``task``.

    Returns the network rounded to ``nets.DECIMALS`` if the rounded network is at least
    ``MIN_ACC`` accurate on ``task.X``; otherwise ``None`` (the caller resamples the task).
    """
    lo, hi = EXTRA_NEURONS
    hidden = [w + int(rng.integers(lo, hi + 1)) for w in canon.widths[1:-1]]
    seed = int(rng.integers(1 << 62))
    widths = [task.n_in] + hidden + [task.n_out]
    for attempt in range(RESTARTS):
        net = _fit(task, widths, np.random.default_rng([seed, attempt])).round_()
        if task.accuracy(net) >= MIN_ACC:
            return net
    return None
