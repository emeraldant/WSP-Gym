"""Small ReLU MLPs: compiled function families and function-preserving symmetries.

Every family is a pair of pure functions:

    sample_params(rng, size) -> dict      JSON-able description of one task instance
    build(params)            -> (Task, MLP)  deterministic in ``params``

``build`` returns the *canonical* compiled network: minimal, hand-constructed
from interpretable gadgets, exact on the task. ``obfuscate`` then hides it
behind symmetries (permutation, per-neuron rescaling, neuron splitting,
distractor neurons, small noise, rounding) so that surface patterns cannot be
memorised.

Conventions: layer l computes ``h_l = relu(W_l @ h_{l-1} + b_l)``; the last
layer is linear. ``W_l`` has shape (fan_out, fan_in), so row i holds the
incoming weights of neuron i and column j the outgoing weights of neuron j of
the previous layer.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

DECIMALS = 2  # weights are stored rounded, so the text the model sees is exact


# ----------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------


@dataclass
class MLP:
    Ws: list[np.ndarray]
    bs: list[np.ndarray]

    def copy(self) -> "MLP":
        return MLP([W.copy() for W in self.Ws], [b.copy() for b in self.bs])

    @property
    def n_layers(self) -> int:
        return len(self.Ws)

    @property
    def widths(self) -> list[int]:
        return [self.Ws[0].shape[1]] + [W.shape[0] for W in self.Ws]

    @property
    def n_weights(self) -> int:
        return int(sum(W.size for W in self.Ws) + sum(b.size for b in self.bs))

    def forward(self, X: np.ndarray, return_hidden: bool = False):
        h = np.asarray(X, dtype=float)
        hidden = []
        for l, (W, b) in enumerate(zip(self.Ws, self.bs)):
            h = h @ W.T + b
            if l < len(self.Ws) - 1:
                h = np.maximum(h, 0.0)
                hidden.append(h)
        return (h, hidden) if return_hidden else h

    def round_(self, decimals: int = DECIMALS) -> "MLP":
        self.Ws = [np.round(W, decimals) + 0.0 for W in self.Ws]  # +0.0 removes -0.0
        self.bs = [np.round(b, decimals) + 0.0 for b in self.bs]
        return self

    def to_json(self) -> dict:
        return {"Ws": [W.tolist() for W in self.Ws], "bs": [b.tolist() for b in self.bs]}

    @staticmethod
    def from_json(d: dict) -> "MLP":
        return MLP(
            [np.array(W, dtype=float).reshape(len(W), -1) for W in d["Ws"]],
            [np.array(b, dtype=float) for b in d["bs"]],
        )

    def equals(self, other: "MLP") -> bool:
        return (
            len(self.Ws) == len(other.Ws)
            and all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(self.Ws, other.Ws))
            and all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(self.bs, other.bs))
        )


# ----------------------------------------------------------------------------
# Task
# ----------------------------------------------------------------------------


@dataclass
class Task:
    family: str
    params: dict
    n_in: int
    n_out: int
    kind: str  # "binary": 1 output, y > 0.5 | "class": argmax | "reg": |y - t| <= tol
    input_desc: str
    spec: str
    output_desc: str
    X: np.ndarray  # (N, n_in) scoring inputs (exhaustive when the domain is small)
    Y: np.ndarray  # binary: (N,) {0,1}; class: (N,) int; reg: (N, n_out)
    tol: float = 0.5
    exhaustive: bool = True

    def correct(self, net: MLP) -> np.ndarray:
        out = net.forward(self.X)
        if self.kind == "binary":
            return (out[:, 0] > 0.5) == (self.Y > 0.5)
        if self.kind == "class":
            return np.argmax(out, axis=1) == self.Y
        return np.all(np.abs(out - self.Y) <= self.tol, axis=1)

    def accuracy(self, net: MLP) -> float:
        return float(np.mean(self.correct(net)))


def _bool_domain(n: int, dseed: int) -> tuple[np.ndarray, bool]:
    if n <= 10:
        return np.array(list(itertools.product([0, 1], repeat=n)), dtype=float), True
    rng = np.random.default_rng(dseed)
    return rng.integers(0, 2, size=(1024, n)).astype(float), False


def _int_domain(n: int, dseed: int, hi: int = 9) -> tuple[np.ndarray, bool]:
    if (hi + 1) ** n <= 2000:
        return np.array(list(itertools.product(range(hi + 1), repeat=n)), dtype=float), True
    rng = np.random.default_rng(dseed)
    return rng.integers(0, hi + 1, size=(1000, n)).astype(float), False


BOOL_DESC = "{n} binary inputs x0..x{m}, each 0 or 1."
INT_DESC = "{n} inputs x0..x{m}, each an integer from 0 to 9."
BINARY_OUT = "1 output y. It counts as correct when (y > 0.5) agrees with the target (1 = true, 0 = false)."
REG_OUT = "{k} output(s). An output counts as correct when it is within 0.5 of the target value."
CLASS_OUT = "{k} outputs y[0]..y[{m}]. The network's answer is the index of the largest output."


def _xs(idx) -> str:
    return ", ".join(f"x{i}" for i in idx)


def _lin_str(coefs: dict) -> str:
    """Human-readable linear form such as 'x0 + 2*x3 - x4'."""
    parts = []
    for k in sorted(coefs, key=int):
        c = coefs[k]
        mag = "" if abs(c) == 1 else f"{abs(c)}*"
        term = f"{mag}x{k}"
        if not parts:
            parts.append(term if c > 0 else f"-{term}")
        else:
            parts.append(f"+ {term}" if c > 0 else f"- {term}")
    return " ".join(parts)


def _lin_row(coefs: dict, n: int) -> np.ndarray:
    row = np.zeros(n)
    for k, c in coefs.items():
        row[int(k)] = c
    return row


def _mlp(layers: list[tuple[list, list]]) -> MLP:
    return MLP(
        [np.array(W, dtype=float).reshape(len(b), -1) for W, b in layers],
        [np.array(b, dtype=float) for _, b in layers],
    )


# ----------------------------------------------------------------------------
# Families. Gadgets used throughout, for integer-valued z:
#   AND of k literals       relu(sum(literals) - (k-1))
#   [z >= t]                relu(z - t + 1) - relu(z - t)
#   thermometer             t_j = relu(z - j)
# ----------------------------------------------------------------------------


def _choose(rng, n, k) -> list[int]:
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))


# --- boolean formula (DNF) ---------------------------------------------------


def dnf_sample(rng, size):
    n = int(rng.integers(4, 7)) + size
    n_cl = int(rng.integers(2, 4)) + (size >= 2)
    clauses = []
    for _ in range(n_cl):
        k = int(rng.integers(2, 4))
        clauses.append([[v, int(rng.integers(0, 2))] for v in _choose(rng, n, k)])
    return {
        "n": n,
        "clauses": clauses,
        "negate": bool(rng.integers(0, 2)),
        "deep": bool(rng.random() < 0.35),
        "dseed": int(rng.integers(1 << 30)),
    }


def dnf_build(p):
    n = p["n"]
    X, ex = _bool_domain(n, p["dseed"])
    W1, b1, strs = [], [], []
    fired = np.zeros(len(X))
    for cl in p["clauses"]:
        row = np.zeros(n)
        nneg = 0
        ok = np.ones(len(X), dtype=bool)
        lits = []
        for v, neg in cl:
            row[v] = -1.0 if neg else 1.0
            nneg += neg
            ok &= (X[:, v] == 0) if neg else (X[:, v] == 1)
            lits.append(f"NOT x{v}" if neg else f"x{v}")
        W1.append(row)
        b1.append(nneg - (len(cl) - 1))
        fired += ok
        strs.append("(" + " AND ".join(lits) + ")")
    any_true = fired > 0
    Y = (~any_true if p["negate"] else any_true).astype(float)
    k = len(W1)
    if p["deep"]:
        l2 = ([[1.0] * k, [1.0] * k], [0.0, -1.0])  # o1 = relu(s), o2 = relu(s - 1)
        l3 = ([[-1.0, 1.0]], [1.0]) if p["negate"] else ([[1.0, -1.0]], [0.0])
        net = _mlp([(W1, b1), l2, l3])
    else:
        net = _mlp([(W1, b1), (([[-1.0] * k], [1.0]) if p["negate"] else ([[1.0] * k], [0.0]))])
    formula = " OR ".join(strs)
    spec = (
        f"Output 0 if {formula} is true; otherwise output 1."
        if p["negate"]
        else f"Output 1 if {formula} is true; otherwise output 0."
    )
    return Task("dnf", p, n, 1, "binary", BOOL_DESC.format(n=n, m=n - 1), spec, BINARY_OUT, X, Y, exhaustive=ex), net


# --- parity -------------------------------------------------------------------


def parity_sample(rng, size):
    n = int(rng.integers(4, 7)) + size
    k = int(rng.integers(2, 5)) + (size >= 2)
    return {"n": n, "S": _choose(rng, n, min(k, n)), "odd": bool(rng.integers(0, 2)), "dseed": int(rng.integers(1 << 30))}


def parity_build(p):
    n, S = p["n"], p["S"]
    X, ex = _bool_domain(n, p["dseed"])
    s = X[:, S].sum(1)
    par = (s % 2 == 1)
    Y = (par if p["odd"] else ~par).astype(float)
    row = np.zeros(n)
    row[S] = 1.0
    k = len(S)
    W1 = [row.copy() for _ in range(k)]
    b1 = [-float(j) for j in range(k)]
    out = [1.0] + [(-2.0 if j % 2 else 2.0) for j in range(1, k)]
    if p["odd"]:
        net = _mlp([(W1, b1), ([out], [0.0])])
    else:
        net = _mlp([(W1, b1), ([[-c for c in out]], [1.0])])
    word = "odd" if p["odd"] else "even"
    spec = f"Output 1 if an {word} number of the inputs {{{_xs(S)}}} are 1; otherwise output 0."
    return Task("parity", p, n, 1, "binary", BOOL_DESC.format(n=n, m=n - 1), spec, BINARY_OUT, X, Y, exhaustive=ex), net


# --- threshold / k-of-n -------------------------------------------------------


def threshold_sample(rng, size):
    n = int(rng.integers(4, 7)) + size
    m = int(rng.integers(3, min(n, 5 + size) + 1))
    S = _choose(rng, n, m)
    weighted = bool(rng.random() < 0.4)
    coefs = {str(v): (int(rng.integers(1, 3)) if weighted else 1) for v in S}
    total = sum(coefs.values())
    mode = str(rng.choice(["atleast", "exactly", "atmost"]))
    k = int(rng.integers(1, total)) if mode != "exactly" else int(rng.integers(1, total))
    return {"n": n, "coefs": coefs, "mode": mode, "k": k, "dseed": int(rng.integers(1 << 30))}


def threshold_build(p):
    n, k, mode = p["n"], p["k"], p["mode"]
    X, ex = _bool_domain(n, p["dseed"])
    row = _lin_row(p["coefs"], n)
    z = X @ row
    lin = _lin_str(p["coefs"])
    if mode == "atleast":
        Y = z >= k
        net = _mlp([([row, row], [-(k - 1.0), -float(k)]), ([[1.0, -1.0]], [0.0])])
        spec = f"Output 1 if {lin} >= {k}; otherwise output 0."
    elif mode == "atmost":
        Y = z <= k
        net = _mlp([([row, row], [-float(k), -(k + 1.0)]), ([[-1.0, 1.0]], [1.0])])
        spec = f"Output 1 if {lin} <= {k}; otherwise output 0."
    else:
        Y = z == k
        net = _mlp([([row, row, row], [-(k - 1.0), -float(k), -(k + 1.0)]), ([[1.0, -2.0, 1.0]], [0.0])])
        spec = f"Output 1 if {lin} = {k} exactly; otherwise output 0."
    return Task("threshold", p, n, 1, "binary", BOOL_DESC.format(n=n, m=n - 1), spec, BINARY_OUT, X, Y.astype(float), exhaustive=ex), net


# --- comparison ---------------------------------------------------------------


def _sample_predicate(rng, n):
    form = int(rng.integers(0, 4))
    if form == 0:  # xi > xj
        i, j = (int(v) for v in rng.choice(n, 2, replace=False))
        return {"coefs": {str(i): 1, str(j): -1}, "theta": 1, "text": f"x{i} > x{j}"}
    if form == 1:  # xi >= xj + c
        i, j = (int(v) for v in rng.choice(n, 2, replace=False))
        c = int(rng.integers(1, 5))
        return {"coefs": {str(i): 1, str(j): -1}, "theta": c, "text": f"x{i} >= x{j} + {c}"}
    if form == 2:  # xi + xj > xk
        i, j, k = (int(v) for v in rng.choice(n, 3, replace=False))
        return {"coefs": {str(i): 1, str(j): 1, str(k): -1}, "theta": 1, "text": f"x{i} + x{j} > x{k}"}
    i = int(rng.integers(n))
    c = int(rng.integers(3, 8))
    return {"coefs": {str(i): 1}, "theta": c, "text": f"x{i} >= {c}"}


def compare_sample(rng, size):
    n = int(rng.integers(3, 6)) + size
    two = bool(rng.random() < 0.4)
    return {"n": n, "preds": [_sample_predicate(rng, n) for _ in range(2 if two else 1)], "dseed": int(rng.integers(1 << 30))}


def compare_build(p):
    n = p["n"]
    X, ex = _int_domain(n, p["dseed"])
    W1, b1 = [], []
    Y = np.ones(len(X), dtype=bool)
    for pr in p["preds"]:
        row = _lin_row(pr["coefs"], n)
        W1 += [row, row]
        b1 += [-(pr["theta"] - 1.0), -float(pr["theta"])]
        Y &= (X @ row) >= pr["theta"]
    if len(p["preds"]) == 1:
        net = _mlp([(W1, b1), ([[1.0, -1.0]], [0.0])])
        spec = f"Output 1 if {p['preds'][0]['text']}; otherwise output 0."
    else:
        net = _mlp([(W1, b1), ([[1.0, -1.0, 1.0, -1.0]], [-1.0]), ([[1.0]], [0.0])])
        spec = f"Output 1 if ({p['preds'][0]['text']}) AND ({p['preds'][1]['text']}); otherwise output 0."
    return Task("compare", p, n, 1, "binary", INT_DESC.format(n=n, m=n - 1), spec, BINARY_OUT, X, Y.astype(float), exhaustive=ex), net


# --- min / max / sort ----------------------------------------------------------


def minmax_sample(rng, size):
    n = int(rng.integers(3, 6)) + size
    mode = str(rng.choice(["max2", "min2", "sort2", "max3", "min3"]))
    k = 3 if mode.endswith("3") else 2
    return {"n": n, "mode": mode, "vars": [int(v) for v in rng.choice(n, k, replace=False)], "dseed": int(rng.integers(1 << 30))}


def minmax_build(p):
    n, mode, v = p["n"], p["mode"], p["vars"]
    X, ex = _int_domain(n, p["dseed"])

    def e(i, c=1.0):
        r = np.zeros(n)
        r[i] = c
        return r

    a, b = v[0], v[1]
    if mode == "max2":  # b + relu(a - b)
        net = _mlp([([e(b), e(a) - e(b)], [0.0, 0.0]), ([[1.0, 1.0]], [0.0])])
        Y = np.maximum(X[:, a], X[:, b])[:, None]
        spec = f"Output max(x{a}, x{b})."
    elif mode == "min2":  # a - relu(a - b)
        net = _mlp([([e(a), e(a) - e(b)], [0.0, 0.0]), ([[1.0, -1.0]], [0.0])])
        Y = np.minimum(X[:, a], X[:, b])[:, None]
        spec = f"Output min(x{a}, x{b})."
    elif mode == "sort2":
        net = _mlp([([e(a), e(b), e(a) - e(b)], [0.0, 0.0, 0.0]), ([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]], [0.0, 0.0])])
        Y = np.stack([np.minimum(X[:, a], X[:, b]), np.maximum(X[:, a], X[:, b])], 1)
        spec = f"Output two values: y[0] = min(x{a}, x{b}) and y[1] = max(x{a}, x{b})."
    elif mode == "max3":  # m = max(a,b) = relu(b) + relu(a-b); max(m,c) = relu(c) + relu(m - c)
        c = v[2]
        net = _mlp([
            ([e(b), e(a) - e(b), e(c)], [0.0, 0.0, 0.0]),
            ([[1.0, 1.0, -1.0], [0.0, 0.0, 1.0]], [0.0, 0.0]),
            ([[1.0, 1.0]], [0.0]),
        ])
        Y = np.max(X[:, [a, b, c]], axis=1)[:, None]
        spec = f"Output max(x{a}, x{b}, x{c})."
    else:  # min3: m = min(a,b) = relu(a) - relu(a-b); min(m,c) = m - relu(m - c)
        c = v[2]
        net = _mlp([
            ([e(a), e(a) - e(b), e(c)], [0.0, 0.0, 0.0]),
            ([[1.0, -1.0, 0.0], [1.0, -1.0, -1.0]], [0.0, 0.0]),
            ([[1.0, -1.0]], [0.0]),
        ])
        Y = np.min(X[:, [a, b, c]], axis=1)[:, None]
        spec = f"Output min(x{a}, x{b}, x{c})."
    k = Y.shape[1]
    return Task("minmax", p, n, k, "reg", INT_DESC.format(n=n, m=n - 1), spec, REG_OUT.format(k=k), X, Y.astype(float), exhaustive=ex), net


# --- arithmetic / piecewise linear ---------------------------------------------


def arith_sample(rng, size):
    n = int(rng.integers(3, 6)) + size
    mode = str(rng.choice(["absdiff", "clip", "satsum", "posdiff", "deadzone"]))
    a, b = (int(v) for v in rng.choice(n, 2, replace=False))
    lo = int(rng.integers(2, 7))
    hi = lo + int(rng.integers(3, 8))
    return {"n": n, "mode": mode, "a": a, "b": b, "lo": lo, "hi": hi, "dseed": int(rng.integers(1 << 30))}


def arith_build(p):
    n, mode, a, b, lo, hi = p["n"], p["mode"], p["a"], p["b"], p["lo"], p["hi"]
    X, ex = _int_domain(n, p["dseed"])
    ea, eb = np.zeros(n), np.zeros(n)
    ea[a], eb[b] = 1.0, 1.0
    xa, xb = X[:, a], X[:, b]
    if mode == "absdiff":
        net = _mlp([([ea - eb, eb - ea], [0.0, 0.0]), ([[1.0, 1.0]], [0.0])])
        Y = np.abs(xa - xb)
        spec = f"Output |x{a} - x{b}|."
    elif mode == "clip":  # lo + relu(s - lo) - relu(s - hi)
        net = _mlp([([ea + eb, ea + eb], [-float(lo), -float(hi)]), ([[1.0, -1.0]], [float(lo)])])
        Y = np.clip(xa + xb, lo, hi)
        spec = f"Output x{a} + x{b} clipped to the range [{lo}, {hi}]."
    elif mode == "satsum":  # s - relu(s - hi)
        net = _mlp([([ea + eb, ea + eb], [0.0, -float(hi)]), ([[1.0, -1.0]], [0.0])])
        Y = np.minimum(xa + xb, hi)
        spec = f"Output min(x{a} + x{b}, {hi})."
    elif mode == "posdiff":  # relu(a - b) + lo
        net = _mlp([([ea - eb], [0.0]), ([[1.0]], [float(lo)])])
        Y = np.maximum(xa - xb, 0) + lo
        spec = f"Output max(x{a} - x{b}, 0) + {lo}."
    else:  # deadzone on d = a - b: relu(d - lo) - relu(-d - lo)
        lo = min(lo, 4)
        net = _mlp([([ea - eb, eb - ea], [-float(lo), -float(lo)]), ([[1.0, -1.0]], [0.0])])
        d = xa - xb
        Y = np.where(d > lo, d - lo, np.where(d < -lo, d + lo, 0.0))
        spec = (
            f"Let d = x{a} - x{b}. Output d - {lo} if d > {lo}, d + {lo} if d < -{lo}, and 0 otherwise."
        )
    return Task("arith", p, n, 1, "reg", INT_DESC.format(n=n, m=n - 1), spec, REG_OUT.format(k=1), X, Y[:, None].astype(float), exhaustive=ex), net


# --- argmax over group scores ---------------------------------------------------


def argmax_sample(rng, size):
    k = int(rng.integers(3, 5))
    n = k + int(rng.integers(1, 3)) + size
    order = [int(v) for v in rng.permutation(n)]
    groups = [[order[c]] for c in range(k)]
    for extra in order[k : k + int(rng.integers(0, n - k + 1))]:
        groups[int(rng.integers(k))].append(extra)
    return {"n": n, "groups": [sorted(g) for g in groups], "dseed": int(rng.integers(1 << 30))}


def argmax_build(p):
    n, groups = p["n"], p["groups"]
    X, _ = _int_domain(n, p["dseed"])
    used = sorted({i for g in groups for i in g})
    W1 = [np.eye(n)[i] for i in used]
    W2 = [[1.0 if i in g else 0.0 for i in used] for g in groups]
    net = _mlp([(W1, [0.0] * len(used)), (W2, [0.0] * len(groups))])
    scores = np.stack([X[:, g].sum(1) for g in groups], 1)
    top = np.sort(scores, axis=1)
    keep = top[:, -1] > top[:, -2]  # drop ties so the target is unambiguous
    X, scores = X[keep], scores[keep]
    k = len(groups)
    desc = "; ".join(f"s{c} = " + " + ".join(f"x{i}" for i in g) for c, g in enumerate(groups))
    spec = f"Compute {k} scores: {desc}. The answer is the index of the largest score (inputs with tied top scores are never tested)."
    return Task("argmax", p, n, k, "class", INT_DESC.format(n=n, m=n - 1), spec, CLASS_OUT.format(k=k, m=k - 1), X, np.argmax(scores, 1), exhaustive=False), net


# --- counting (thermometer code) -------------------------------------------------


def count_sample(rng, size):
    n = int(rng.integers(4, 7)) + size
    m = int(rng.integers(2, 5)) + (size >= 2)
    return {"n": n, "S": _choose(rng, n, min(m, n)), "dseed": int(rng.integers(1 << 30))}


def count_build(p):
    n, S = p["n"], p["S"]
    m = len(S)
    X, ex = _bool_domain(n, p["dseed"])
    row = np.zeros(n)
    row[S] = 1.0
    js = list(range(-1, m))  # t_j = relu(s - j), j = -1..m-1
    W1 = [row.copy() for _ in js]
    b1 = [-float(j) for j in js]
    W2 = np.zeros((m + 1, len(js)))
    for c in range(m + 1):  # logit_c = t_{c-1} - 2 t_c + t_{c+1}: a bump that is 1 only at s = c
        for j, coef in ((c - 1, 1.0), (c, -2.0), (c + 1, 1.0)):
            if j in js:
                W2[c, js.index(j)] = coef
    net = _mlp([(W1, b1), (W2.tolist(), [0.0] * (m + 1))])
    spec = f"Count how many of the inputs {{{_xs(S)}}} are 1. The answer is that count (0 to {m})."
    return Task("count", p, n, m + 1, "class", BOOL_DESC.format(n=n, m=n - 1), spec, CLASS_OUT.format(k=m + 1, m=m), X, X[:, S].sum(1).astype(int), exhaustive=ex), net


# --- equality of bit pairs --------------------------------------------------------


def equality_sample(rng, size):
    k = int(rng.integers(2, 4)) + (size >= 2)
    n = 2 * k + int(rng.integers(0, 2)) + size
    idx = [int(v) for v in rng.permutation(n)[: 2 * k]]
    return {
        "n": n, "pairs": [[idx[2 * i], idx[2 * i + 1]] for i in range(k)],
        "m": int(rng.integers(0, k)),  # tolerated number of mismatching pairs
        "equal": bool(rng.integers(0, 2)), "dseed": int(rng.integers(1 << 30)),
    }


def equality_build(p):
    """Pairwise equality / Hamming distance: d = number of pairs that differ, via XOR gadgets."""
    n, m = p["n"], p["m"]
    X, ex = _bool_domain(n, p["dseed"])
    W1 = []
    d = np.zeros(len(X))
    for i, j in p["pairs"]:
        r = np.zeros(n)
        r[i], r[j] = 1.0, -1.0
        W1 += [r, -r]  # relu(xi - xj) + relu(xj - xi) = [xi != xj]
        d += X[:, i] != X[:, j]
    k = len(W1)
    plist = ", ".join(f"(x{i}, x{j})" for i, j in p["pairs"])
    if m == 0:
        pairs = " and ".join(f"x{i} = x{j}" for i, j in p["pairs"])
        if p["equal"]:
            net = _mlp([(W1, [0.0] * k), ([[-1.0] * k], [1.0])])
            spec = f"Output 1 if {pairs} (every listed pair matches); otherwise output 0."
        else:
            net = _mlp([(W1, [0.0] * k), ([[1.0] * k], [0.0])])
            spec = f"Output 0 if {pairs} (every listed pair matches); otherwise output 1."
    else:  # [d > m] = relu(d - m) - relu(d - m - 1)
        l2 = ([[1.0] * k, [1.0] * k], [-float(m), -(m + 1.0)])
        if p["equal"]:
            net = _mlp([(W1, [0.0] * k), l2, ([[-1.0, 1.0]], [1.0])])
            spec = f"Compare the input pairs {plist}. Output 1 if at most {m} of the pairs differ; otherwise output 0."
        else:
            net = _mlp([(W1, [0.0] * k), l2, ([[1.0, -1.0]], [0.0])])
            spec = f"Compare the input pairs {plist}. Output 1 if more than {m} of the pairs differ; otherwise output 0."
    Y = (d <= m) if p["equal"] else (d > m)
    return Task("equality", p, n, 1, "binary", BOOL_DESC.format(n=n, m=n - 1), spec, BINARY_OUT, X, Y.astype(float), exhaustive=ex), net


# --- multiplexer (routing) ---------------------------------------------------------


def mux_sample(rng, size):
    nc = 1 if (size == 0 and rng.random() < 0.5) else 2
    nd = 2**nc
    n = nc + nd + int(rng.integers(0, 2)) + size
    idx = [int(v) for v in rng.permutation(n)]
    return {"n": n, "ctrl": idx[:nc], "data": idx[nc : nc + nd], "dseed": int(rng.integers(1 << 30))}


def mux_build(p):
    n, ctrl, data = p["n"], p["ctrl"], p["data"]
    nc = len(ctrl)
    X, ex = _bool_domain(n, p["dseed"])
    sel = np.zeros(len(X), dtype=int)
    for c in ctrl:  # ctrl[0] is the most significant bit
        sel = sel * 2 + X[:, c].astype(int)
    Y = X[np.arange(len(X)), np.array(data)[sel]]
    W1, b1 = [], []
    for i, dvar in enumerate(data):
        row = np.zeros(n)
        row[dvar] = 1.0
        nneg = 0
        for pos, c in enumerate(ctrl):
            bit = (i >> (nc - 1 - pos)) & 1
            row[c] = 1.0 if bit else -1.0
            nneg += 1 - bit
        W1.append(row)
        b1.append(nneg - float(nc))
    net = _mlp([(W1, b1), ([[1.0] * len(data)], [0.0])])
    if nc == 1:
        sel_txt = f"x{ctrl[0]} is a selector bit"
    else:
        sel_txt = f"x{ctrl[0]} (high bit) and x{ctrl[1]} (low bit) form a 2-bit selector s = 2*x{ctrl[0]} + x{ctrl[1]}"
    routes = ", ".join(f"s={i} -> x{d}" for i, d in enumerate(data))
    spec = f"Multiplexer: {sel_txt}. Output the data input chosen by the selector: {routes}."
    return Task("mux", p, n, 1, "binary", BOOL_DESC.format(n=n, m=n - 1), spec, BINARY_OUT, X, Y.astype(float), exhaustive=ex), net


# --- interval membership -------------------------------------------------------------


def _sample_interval(rng, n):
    if rng.random() < 0.5:
        i = int(rng.integers(n))
        coefs, top = {str(i): 1}, 9
    else:
        i, j = (int(v) for v in rng.choice(n, 2, replace=False))
        coefs, top = {str(i): 1, str(j): 1}, 18
    lo = int(rng.integers(1, top - 3))
    hi = int(rng.integers(lo + 1, min(top - 1, lo + top // 2) + 1))
    return {"coefs": coefs, "lo": lo, "hi": hi}


def interval_sample(rng, size):
    n = int(rng.integers(3, 6)) + size
    box = bool(rng.random() < 0.35)
    return {"n": n, "ivs": [_sample_interval(rng, n) for _ in range(2 if box else 1)], "inside": bool(box or rng.random() < 0.6), "dseed": int(rng.integers(1 << 30))}


def interval_build(p):
    n = p["n"]
    X, ex = _int_domain(n, p["dseed"])
    W1, b1, txt = [], [], []
    inside = np.ones(len(X), dtype=bool)
    for iv in p["ivs"]:
        row = _lin_row(iv["coefs"], n)
        lo, hi = iv["lo"], iv["hi"]
        W1 += [row] * 4  # [z >= lo] - [z >= hi + 1]
        b1 += [-(lo - 1.0), -float(lo), -float(hi), -(hi + 1.0)]
        z = X @ row
        inside &= (z >= lo) & (z <= hi)
        txt.append(f"{lo} <= {_lin_str(iv['coefs'])} <= {hi}")
    unit = [1.0, -1.0, -1.0, 1.0]
    if len(p["ivs"]) == 1:
        if p["inside"]:
            net = _mlp([(W1, b1), ([unit], [0.0])])
            Y, spec = inside, f"Output 1 if {txt[0]}; otherwise output 0."
        else:
            net = _mlp([(W1, b1), ([[-u for u in unit]], [1.0])])
            Y, spec = ~inside, f"Output 0 if {txt[0]}; otherwise output 1."
    else:
        net = _mlp([(W1, b1), ([unit + unit], [-1.0]), ([[1.0]], [0.0])])
        Y, spec = inside, f"Output 1 if ({txt[0]}) AND ({txt[1]}); otherwise output 0."
    return Task("interval", p, n, 1, "binary", INT_DESC.format(n=n, m=n - 1), spec, BINARY_OUT, X, Y.astype(float), exhaustive=ex), net


FAMILIES: dict[str, tuple[Callable, Callable]] = {
    "dnf": (dnf_sample, dnf_build),
    "parity": (parity_sample, parity_build),
    "threshold": (threshold_sample, threshold_build),
    "compare": (compare_sample, compare_build),
    "minmax": (minmax_sample, minmax_build),
    "arith": (arith_sample, arith_build),
    "argmax": (argmax_sample, argmax_build),
    "count": (count_sample, count_build),
    "equality": (equality_sample, equality_build),
    "mux": (mux_sample, mux_build),
    "interval": (interval_sample, interval_build),
}
BOOL_FAMILIES = {"dnf", "parity", "threshold", "count", "equality", "mux"}


def build_task(family: str, params: dict) -> tuple[Task, MLP]:
    return FAMILIES[family][1](params)


def sample_task(family: str, rng: np.random.Generator, size: int = 0, max_tries: int = 200) -> tuple[Task, MLP]:
    """Sample a task instance whose labels are not degenerate."""
    for _ in range(max_tries):
        params = FAMILIES[family][0](rng, size)
        task, net = build_task(family, params)
        if len(task.X) < 12:
            continue
        if task.kind == "binary":
            frac = float(np.mean(task.Y))
            if not 0.2 <= frac <= 0.8:
                continue
        elif task.kind == "class":
            counts = np.bincount(task.Y, minlength=task.n_out) / len(task.Y)
            if counts.min() < 0.04 or counts.max() > 0.8:
                continue
        assert task.accuracy(net) == 1.0, (family, params)
        return task, net
    raise RuntimeError(f"could not sample a balanced task for {family}")


# ----------------------------------------------------------------------------
# Function-preserving symmetries. ``l`` indexes a hidden layer: its incoming
# weights are Ws[l] / bs[l] and its outgoing weights are the columns of Ws[l+1].
# ----------------------------------------------------------------------------


def permute_hidden(net: MLP, l: int, perm: np.ndarray) -> MLP:
    net = net.copy()
    net.Ws[l], net.bs[l] = net.Ws[l][perm], net.bs[l][perm]
    net.Ws[l + 1] = net.Ws[l + 1][:, perm]
    return net


def rescale_hidden(net: MLP, l: int, c: np.ndarray) -> MLP:
    """relu(c * z) / c == relu(z) for c > 0."""
    assert np.all(c > 0)
    net = net.copy()
    net.Ws[l] = net.Ws[l] * c[:, None]
    net.bs[l] = net.bs[l] * c
    net.Ws[l + 1] = net.Ws[l + 1] / c[None, :]
    return net


def split_neuron(net: MLP, l: int, j: int, alpha: float) -> MLP:
    """Duplicate neuron j; the copies share its outgoing weights alpha : 1 - alpha."""
    net = net.copy()
    net.Ws[l] = np.vstack([net.Ws[l], net.Ws[l][j : j + 1]])
    net.bs[l] = np.append(net.bs[l], net.bs[l][j])
    col = net.Ws[l + 1][:, j].copy()
    net.Ws[l + 1][:, j] = alpha * col
    net.Ws[l + 1] = np.hstack([net.Ws[l + 1], ((1 - alpha) * col)[:, None]])
    return net


def add_neuron(net: MLP, l: int, row: np.ndarray, bias: float, outcol: np.ndarray) -> MLP:
    net = net.copy()
    net.Ws[l] = np.vstack([net.Ws[l], row[None, :]])
    net.bs[l] = np.append(net.bs[l], bias)
    net.Ws[l + 1] = np.hstack([net.Ws[l + 1], outcol[:, None]])
    return net


def _layer_inputs(net: MLP, l: int, X: np.ndarray) -> np.ndarray:
    if l == 0:
        return X
    return net.forward(X, return_hidden=True)[1][l - 1]


def _random_row(rng, like: np.ndarray) -> tuple[np.ndarray, float]:
    """A plausible incoming row: resample an existing row's nonzeros onto new positions."""
    fan_in = like.shape[1]
    src = like[int(rng.integers(like.shape[0]))]
    vals = src[np.abs(src) > 1e-9]
    if len(vals) == 0:
        vals = np.array([1.0])
    k = int(np.clip(len(vals) + rng.integers(-1, 2), 1, fan_in))
    row = np.zeros(fan_in)
    pos = rng.choice(fan_in, size=k, replace=False)
    row[pos] = rng.choice(vals, size=k) * rng.choice([-1.0, 1.0], size=k)
    return row, float(rng.integers(-2, 3))


def _random_outcol(rng, W_next: np.ndarray) -> np.ndarray:
    vals = W_next[np.abs(W_next) > 1e-9]
    if len(vals) == 0:
        vals = np.array([1.0])
    col = rng.choice(vals, size=W_next.shape[0]) * rng.choice([-1.0, 1.0], size=W_next.shape[0])
    if W_next.shape[0] > 2:  # keep some sparsity in wide fan-outs, but never an all-zero column
        mask = rng.random(W_next.shape[0]) < 0.6
        mask[int(rng.integers(len(mask)))] = True
        col = col * mask
    return col


def add_dead_neuron(net: MLP, l: int, X: np.ndarray, rng) -> MLP:
    """A neuron that never fires on the input domain but has ordinary-looking weights."""
    row, _ = _random_row(rng, net.Ws[l])
    pre = _layer_inputs(net, l, X) @ row
    bias = -(float(pre.max()) + float(rng.integers(1, 3)))
    return add_neuron(net, l, row, bias, _random_outcol(rng, net.Ws[l + 1]))


def add_cancel_pair(net: MLP, l: int, rng) -> MLP:
    """Two neurons with identical inputs and opposite outputs: exactly zero net effect."""
    row, bias = _random_row(rng, net.Ws[l])
    col = _random_outcol(rng, net.Ws[l + 1])
    return add_neuron(add_neuron(net, l, row, bias, col), l, row, bias, -col)


@dataclass
class ObfuscationConfig:
    n_distract: tuple[int, int] = (2, 4)  # extra neurons per hidden layer (inclusive range)
    p_split: float = 0.5
    scale: tuple[float, float] = (0.5, 2.0)
    noise: float = 0.01
    min_clean_acc: float = 0.99


def obfuscate(net: MLP, task: Task, rng: np.random.Generator, cfg: ObfuscationConfig) -> MLP:
    """Apply random symmetries, then noise and rounding. Returns the *clean* network of an
    episode; raises if noise/rounding pushed accuracy below ``cfg.min_clean_acc``."""
    net = net.copy()
    for l in range(net.n_layers - 1):
        budget = int(rng.integers(cfg.n_distract[0], cfg.n_distract[1] + 1))
        while budget > 0:
            r = rng.random()
            if r < cfg.p_split / 2 or budget == 1 and r < 0.5:
                net = split_neuron(net, l, int(rng.integers(net.Ws[l].shape[0])), float(rng.uniform(0.3, 0.7)))
                budget -= 1
            elif r < 0.6 or budget == 1:
                net = add_dead_neuron(net, l, task.X, rng)
                budget -= 1
            else:
                net = add_cancel_pair(net, l, rng)
                budget -= 2
        width = net.Ws[l].shape[0]
        net = permute_hidden(net, l, rng.permutation(width))
        net = rescale_hidden(net, l, np.exp(rng.uniform(np.log(cfg.scale[0]), np.log(cfg.scale[1]), size=width)))
    exact = net
    for attempt in range(6):
        sigma = cfg.noise * (0.5**attempt)
        noisy = exact.copy()
        # Integer-valued inputs amplify weight noise roughly tenfold, so damp it there.
        s_in = sigma * (0.3 if task.family not in BOOL_FAMILIES else 1.0)
        for i in range(noisy.n_layers):
            s = s_in if i == 0 else sigma
            noisy.Ws[i] = noisy.Ws[i] + rng.normal(0, s, size=noisy.Ws[i].shape)
            noisy.bs[i] = noisy.bs[i] + rng.normal(0, s, size=noisy.bs[i].shape)
        noisy.round_()
        if task.accuracy(noisy) >= cfg.min_clean_acc:
            return noisy
    raise RuntimeError("obfuscation could not keep the network accurate")
