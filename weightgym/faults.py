"""Planted faults. Each fault function returns ``(faulty_net, inverse_edit, info)``
where ``inverse_edit`` is a valid edit in the DSL of ``edits.py`` that undoes it
(up to rounding), or ``None`` when the fault does not apply to this network.

Faults are designed to be invisible to marginal weight statistics wherever
possible: sign flips and swaps leave the multiset of magnitudes unchanged, and
scale factors stay inside the range that the random per-neuron rescaling of
``nets.obfuscate`` already produces.
"""

from __future__ import annotations

import numpy as np

from .nets import DECIMALS, MLP, Task

SCALE_FACTORS = (0.3, 0.4, 2.5, 3.0)


def _hidden(net: MLP, rng) -> tuple[int, int]:
    l = int(rng.integers(net.n_layers - 1))
    return l, int(rng.integers(net.Ws[l].shape[0]))


def _r(x):
    return np.round(x, DECIMALS) + 0.0


def sign_flip_out(net: MLP, task: Task, rng):
    l, j = _hidden(net, rng)
    f = net.copy()
    f.Ws[l + 1][:, j] *= -1.0
    f.round_()
    return f, [{"op": "scale", "tensor": f"W{l + 2}", "axis": "col", "index": j, "factor": -1.0}], {"layer": l, "neuron": j}


def sign_flip_in(net: MLP, task: Task, rng):
    l, j = _hidden(net, rng)
    f = net.copy()
    f.Ws[l][j, :] *= -1.0
    f.round_()
    return f, [{"op": "scale", "tensor": f"W{l + 1}", "axis": "row", "index": j, "factor": -1.0}], {"layer": l, "neuron": j}


def swap_out_cols(net: MLP, task: Task, rng):
    l = int(rng.integers(net.n_layers - 1))
    width = net.Ws[l].shape[0]
    if width < 2:
        return None
    i, j = (int(v) for v in rng.choice(width, 2, replace=False))
    f = net.copy()
    f.Ws[l + 1][:, [i, j]] = f.Ws[l + 1][:, [j, i]]
    return f, [{"op": "swap", "tensor": f"W{l + 2}", "axis": "col", "i": i, "j": j}], {"layer": l, "neurons": [i, j]}


def scale_out_col(net: MLP, task: Task, rng):
    l, j = _hidden(net, rng)
    factor = float(rng.choice(SCALE_FACTORS))
    f = net.copy()
    f.Ws[l + 1][:, j] = _r(f.Ws[l + 1][:, j] * factor)
    return f, [{"op": "scale", "tensor": f"W{l + 2}", "axis": "col", "index": j, "factor": round(1 / factor, 4)}], {"layer": l, "neuron": j, "factor": factor}


def scale_in_row(net: MLP, task: Task, rng):
    l, j = _hidden(net, rng)
    factor = float(rng.choice(SCALE_FACTORS))
    f = net.copy()
    f.Ws[l][j, :] = _r(f.Ws[l][j, :] * factor)
    return f, [{"op": "scale", "tensor": f"W{l + 1}", "axis": "row", "index": j, "factor": round(1 / factor, 4)}], {"layer": l, "neuron": j, "factor": factor}


def bias_shift(net: MLP, task: Task, rng):
    """Shift one hidden bias so the neuron dies, is always on, or fires at the wrong threshold."""
    l, j = _hidden(net, rng)
    row = np.abs(net.Ws[l][j])
    unit = float(row[row > 1e-9].mean()) if np.any(row > 1e-9) else 1.0
    delta = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 4.0) * unit)
    f = net.copy()
    old = float(f.bs[l][j])
    f.bs[l][j] = _r(old + delta)
    if f.bs[l][j] == old:
        return None
    return f, [{"op": "set", "tensor": f"b{l + 1}", "index": j, "value": old}], {"layer": l, "neuron": j, "delta": float(f.bs[l][j] - old)}


def weight_corrupt(net: MLP, task: Task, rng):
    """Corrupt a single weight: zero it, negate it, or overwrite it with a neighbour's value."""
    l = int(rng.integers(net.n_layers))
    W = net.Ws[l]
    r, c = int(rng.integers(W.shape[0])), int(rng.integers(W.shape[1]))
    old = float(W[r, c])
    others = W[r][np.abs(W[r] - old) >= 0.25]
    mode = str(rng.choice(["zero", "negate", "copy"]))
    if mode == "copy" and len(others):
        new = float(rng.choice(others))
    elif mode == "negate":
        new = -old
    else:
        new = 0.0
    if abs(new - old) < 0.25:
        return None
    f = net.copy()
    f.Ws[l][r, c] = new
    return f, [{"op": "set", "tensor": f"W{l + 1}", "index": [r, c], "value": old}], {"layer": l, "index": [r, c], "mode": mode}


def label_swap(net: MLP, task: Task, rng):
    """Swap two output units (rows of the last layer together with their biases)."""
    n_out = net.Ws[-1].shape[0]
    if n_out < 2:
        return None
    i, j = (int(v) for v in rng.choice(n_out, 2, replace=False))
    f = net.copy()
    f.Ws[-1][[i, j]] = f.Ws[-1][[j, i]]
    f.bs[-1][[i, j]] = f.bs[-1][[j, i]]
    L = net.n_layers
    inv = [
        {"op": "swap", "tensor": f"W{L}", "axis": "row", "i": i, "j": j},
        {"op": "swap", "tensor": f"b{L}", "i": i, "j": j},
    ]
    return f, inv, {"outputs": [i, j]}


def spurious_direction(net: MLP, task: Task, rng):
    """Add multiples of one shared direction to the outgoing weights of 2-3 neurons
    (a small-network analogue of a spurious direction written by several MLP neurons)."""
    l = int(rng.integers(net.n_layers - 1))
    W = net.Ws[l + 1]
    fan_out, width = W.shape
    if width < 2:
        return None
    m = int(rng.integers(2, min(3, width) + 1))
    neurons = [int(v) for v in rng.choice(width, m, replace=False)]
    if fan_out == 1:
        v = np.array([1.0])
    else:
        v = rng.choice([-1.0, -0.5, 0.5, 1.0], size=fan_out) * (rng.random(fan_out) < 0.7)
        if not np.any(v):
            v[int(rng.integers(fan_out))] = 1.0
    nz = np.abs(W[np.abs(W) > 1e-9])
    unit = float(np.median(nz)) if len(nz) else 1.0
    f = net.copy()
    inv = []
    for j in neurons:
        delta = _r(float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5) * unit) * v)
        if not np.any(delta):
            return None
        f.Ws[l + 1][:, j] = _r(f.Ws[l + 1][:, j] + delta)
        inv.append({"op": "add", "tensor": f"W{l + 2}", "axis": "col", "index": j, "vector": (-delta + 0.0).tolist()})
    return f, inv, {"layer": l, "neurons": neurons}


FAULTS = {
    "sign_flip_out": sign_flip_out,
    "sign_flip_in": sign_flip_in,
    "swap_out_cols": swap_out_cols,
    "scale_out_col": scale_out_col,
    "bias_shift": bias_shift,
    "weight_corrupt": weight_corrupt,
    "label_swap": label_swap,
    "scale_in_row": scale_in_row,
    "spurious_direction": spurious_direction,
}


def applicable_faults(net: MLP, kinds) -> list[str]:
    return [k for k in kinds if k != "label_swap" or net.Ws[-1].shape[0] >= 2]


def plant_fault(net: MLP, task: Task, kind: str, rng, min_drop: float = 0.15, tries: int = 60):
    """Try positions for ``kind`` until one drops accuracy by at least ``min_drop`` *and* its
    recorded inverse edit restores the network. Returns (faulty, inverse, info, acc_faulty) or None."""
    from .edits import apply_edit, validate_edit

    acc_clean = task.accuracy(net)
    for _ in range(tries):
        out = FAULTS[kind](net, task, rng)
        if out is None:
            continue
        faulty, inv, info = out
        acc = task.accuracy(faulty)
        if acc_clean - acc < min_drop:
            continue
        restored = task.accuracy(apply_edit(faulty, validate_edit(inv, faulty)))
        if (restored - acc) / (acc_clean - acc) < 0.999:
            continue
        return faulty, inv, info, acc
    return None
