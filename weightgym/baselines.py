"""Non-LLM baselines. Each policy maps an Episode to an answer string (a JSON edit),
except ``gradient_repair`` which is an unconstrained upper bound and returns a network.

  noop_policy      always answers []
  random_policy    one random valid edit
  outlier_policy   statistics only: find the most unusual neuron / bias / weight and normalise it
  search_policy    brute force over single-op edits *using forward passes* (not weight reading)
  gradient_repair  a few hundred Adam steps on half of the scoring inputs (ignores the edit budget)
"""

from __future__ import annotations

import json

import numpy as np

from .edits import apply_edit
from .env import Episode
from .nets import MLP


def noop_policy(ep: Episode, rng=None) -> str:
    return "[]"


def random_policy(ep: Episode, rng: np.random.Generator) -> str:
    net = ep.net
    l = int(rng.integers(net.n_layers))
    W = net.Ws[l]
    kind = str(rng.choice(["scale", "swap", "set", "set_bias"]))
    axis = str(rng.choice(["row", "col"]))
    n = W.shape[0] if axis == "row" else W.shape[1]
    if kind == "scale":
        op = {"op": "scale", "tensor": f"W{l + 1}", "axis": axis, "index": int(rng.integers(n)), "factor": float(rng.choice([-1, 0.33, 0.4, 2.5, 3.0]))}
    elif kind == "swap" and n >= 2:
        i, j = (int(v) for v in rng.choice(n, 2, replace=False))
        op = {"op": "swap", "tensor": f"W{l + 1}", "axis": axis, "i": i, "j": j}
    elif kind == "set_bias":
        op = {"op": "set", "tensor": f"b{l + 1}", "index": int(rng.integers(W.shape[0])), "value": round(float(rng.normal(0, 1.5)), 2)}
    else:
        op = {"op": "set", "tensor": f"W{l + 1}", "index": [int(rng.integers(W.shape[0])), int(rng.integers(W.shape[1]))], "value": round(float(rng.normal(0, 1.0)), 2)}
    return json.dumps([op])


def _robust_z(v: np.ndarray) -> np.ndarray:
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return (v - med) / max(mad, 0.15)


def outlier_policy(ep: Episode, rng=None, threshold: float = 2.5) -> str:
    """Pick the single most statistically unusual thing and pull it back to the layer's median.

    Candidates, per hidden layer: (a) neuron 'gain' log(|in row| * |out col|), which is invariant to the
    rescaling symmetry, fixed by rescaling the out column; (b) bias / row-norm ratio, fixed by setting the
    bias; (c) a weight whose magnitude is unusual within its row, fixed by setting it to the row's median.
    Sign flips and swaps are invisible to all of these by construction.
    """
    net = ep.net
    best = (threshold, None)
    for l in range(net.n_layers - 1):
        W, b, Wn = net.Ws[l], net.bs[l], net.Ws[l + 1]
        rn = np.linalg.norm(W, axis=1) + 1e-6
        cn = np.linalg.norm(Wn, axis=0) + 1e-6
        live = cn > 0.1
        if live.sum() >= 4:
            g = np.log(rn * cn)
            zl = _robust_z(g[live])
            idx = np.flatnonzero(live)
            k = int(np.argmax(np.abs(zl)))
            if abs(zl[k]) > best[0]:
                j = int(idx[k])
                factor = float(np.exp(np.median(g[live]) - g[j]))
                best = (abs(zl[k]), {"op": "scale", "tensor": f"W{l + 2}", "axis": "col", "index": j, "factor": round(factor, 3)})
            ratio = b / rn
            zb = _robust_z(ratio[live])
            k = int(np.argmax(np.abs(zb)))
            if abs(zb[k]) > best[0]:
                j = int(idx[k])
                best = (abs(zb[k]), {"op": "set", "tensor": f"b{l + 1}", "index": j, "value": round(float(np.median(ratio[live]) * rn[j]), 2)})
    for l in range(net.n_layers):
        W = net.Ws[l]
        for r in range(W.shape[0]):
            mag = np.abs(W[r])
            big = mag > 0.25 * mag.max() if mag.max() > 0 else np.zeros_like(mag, dtype=bool)
            if big.sum() >= 3:
                lm = np.log(mag[big])
                z = _robust_z(lm)
                k = int(np.argmax(np.abs(z)))
                if abs(z[k]) > best[0]:
                    c = int(np.flatnonzero(big)[k])
                    best = (abs(z[k]), {"op": "set", "tensor": f"W{l + 1}", "index": [r, c], "value": round(float(np.sign(W[r, c]) * np.exp(np.median(lm))), 2)})
    return json.dumps([best[1]] if best[1] else [])


def single_op_candidates(net: MLP) -> list[dict]:
    """All sign flips of a row/column, column swaps within a hidden layer, and output-row swaps."""
    ops = []
    L = net.n_layers
    for l in range(L):
        W = net.Ws[l]
        ops += [{"op": "scale", "tensor": f"W{l + 1}", "axis": "row", "index": i, "factor": -1.0} for i in range(W.shape[0])]
        ops += [{"op": "scale", "tensor": f"W{l + 1}", "axis": "col", "index": j, "factor": -1.0} for j in range(W.shape[1])]
        if l > 0:
            ops += [{"op": "swap", "tensor": f"W{l + 1}", "axis": "col", "i": i, "j": j} for i in range(W.shape[1]) for j in range(i + 1, W.shape[1])]
    return ops


def search_policy(ep: Episode, rng=None) -> str:
    """Brute-force search over single sign-flip / swap edits, scored with forward passes on the
    scoring inputs. This is *not* weight reading; it shows what a tool-using search gets for free."""
    base = ep.task.accuracy(ep.net)
    best_acc, best = base, []
    for op in single_op_candidates(ep.net):
        acc = ep.task.accuracy(apply_edit(ep.net, [op]))
        if acc > best_acc + 1e-9:
            best_acc, best = acc, [op]
    return json.dumps(best)


def gradient_repair(ep: Episode, steps: int = 300, lr: float = 0.02, seed: int = 0, idx=None) -> MLP:
    """Adam on a random half of the scoring inputs (or on the inputs ``idx``, when given), starting from
    the shown network. Loss: squared error for binary/regression targets, cross-entropy for classification."""
    task = ep.task
    if idx is None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(task.X))[: max(8, len(task.X) // 2)]
    idx = np.asarray(idx)
    X = task.X[idx]
    net = ep.net.copy()
    params = net.Ws + net.bs
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    L = net.n_layers
    for t in range(1, steps + 1):
        acts = [X]
        h = X
        for l in range(L):
            h = h @ net.Ws[l].T + net.bs[l]
            if l < L - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        out = acts[-1]
        if task.kind == "class":
            p = np.exp(out - out.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            p[np.arange(len(idx)), task.Y[idx]] -= 1
            delta = p / len(idx)
        else:
            target = task.Y[idx].reshape(len(idx), -1)
            delta = 2 * (out - target) / len(idx)
        gW, gb = [None] * L, [None] * L
        for l in reversed(range(L)):
            gW[l] = delta.T @ acts[l]
            gb[l] = delta.sum(0)
            if l > 0:
                delta = (delta @ net.Ws[l]) * (acts[l] > 0)
        for i, g in enumerate(gW + gb):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            params[i] -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    return net


POLICIES = {"noop": noop_policy, "random": random_policy, "outlier": outlier_policy, "search": search_policy}
