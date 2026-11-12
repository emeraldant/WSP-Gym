"""A small strict edit DSL for repairing an MLP.

An edit is a JSON list of at most ``MAX_OPS`` operations. Each operation touches
one row, one column, or one scalar of a single tensor. Tensors are named
``W1..WL`` (weight matrices, shape fan_out x fan_in) and ``b1..bL`` (biases).

    {"op": "scale", "tensor": "W2", "axis": "col", "index": 4, "factor": -1}
    {"op": "scale", "tensor": "b1", "index": 4, "factor": 0.5}
    {"op": "swap",  "tensor": "W2", "axis": "col", "i": 1, "j": 5}
    {"op": "swap",  "tensor": "b2", "i": 0, "j": 2}
    {"op": "set",   "tensor": "W1", "index": [3, 0], "value": 0.5}
    {"op": "set",   "tensor": "b1", "index": 3, "value": -1.5}
    {"op": "add",   "tensor": "W2", "axis": "col", "index": 2, "vector": [0.5, -0.5]}
    {"op": "add",   "tensor": "b1", "index": 3, "value": 0.25}

The empty list ``[]`` means "no fault, leave the network alone".
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

from .nets import MLP

MAX_OPS = 3
MAX_ABS = 1000.0
OPS = ("scale", "swap", "set", "add")


class EditError(ValueError):
    """The answer is not a valid edit (bad JSON, bad op, or over budget)."""


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise EditError(f"{what} must be a number")
    v = float(v)
    if not math.isfinite(v) or abs(v) > MAX_ABS:
        raise EditError(f"{what} must be finite with magnitude <= {MAX_ABS:g}")
    return v


def _idx(v, size: int, what: str) -> int:
    if not _is_int(v) or not 0 <= v < size:
        raise EditError(f"{what} must be an integer in [0, {size - 1}]")
    return int(v)


def _keys(op: dict, required: set[str]) -> None:
    extra, missing = set(op) - required, required - set(op)
    if missing:
        raise EditError(f"op {op.get('op')!r} is missing {sorted(missing)}")
    if extra:
        raise EditError(f"op {op.get('op')!r} has unexpected keys {sorted(extra)}")


def _tensor(net: MLP, name) -> tuple[str, int]:
    m = re.fullmatch(r"([Wb])([1-9]\d*)", name) if isinstance(name, str) else None
    if not m or int(m.group(2)) > net.n_layers:
        raise EditError(f"unknown tensor {name!r}")
    return m.group(1), int(m.group(2)) - 1


def validate_edit(ops, net: MLP) -> list[dict]:
    """Check an edit against ``net`` and return it in normalised form."""
    if not isinstance(ops, list):
        raise EditError("edit must be a JSON list")
    if len(ops) > MAX_OPS:
        raise EditError(f"over budget: {len(ops)} ops > {MAX_OPS}")
    out = []
    for op in ops:
        if not isinstance(op, dict) or op.get("op") not in OPS:
            raise EditError(f"each op must be an object with op in {OPS}")
        kind, name = op["op"], op.get("tensor")
        tk, l = _tensor(net, name)
        W, b = net.Ws[l], net.bs[l]
        if tk == "W" and kind != "set":
            if op.get("axis") not in ("row", "col"):
                raise EditError("weight-matrix ops need axis 'row' or 'col'")
            n_lines = W.shape[0] if op["axis"] == "row" else W.shape[1]
            line_len = W.shape[1] if op["axis"] == "row" else W.shape[0]
        if kind == "scale":
            if tk == "W":
                _keys(op, {"op", "tensor", "axis", "index", "factor"})
                norm = {"op": kind, "tensor": name, "axis": op["axis"], "index": _idx(op["index"], n_lines, "index")}
            else:
                _keys(op, {"op", "tensor", "index", "factor"})
                norm = {"op": kind, "tensor": name, "index": _idx(op["index"], len(b), "index")}
            norm["factor"] = _num(op["factor"], "factor")
        elif kind == "swap":
            if tk == "W":
                _keys(op, {"op", "tensor", "axis", "i", "j"})
                norm = {"op": kind, "tensor": name, "axis": op["axis"], "i": _idx(op["i"], n_lines, "i"), "j": _idx(op["j"], n_lines, "j")}
            else:
                _keys(op, {"op", "tensor", "i", "j"})
                norm = {"op": kind, "tensor": name, "i": _idx(op["i"], len(b), "i"), "j": _idx(op["j"], len(b), "j")}
        elif kind == "set":
            _keys(op, {"op", "tensor", "index", "value"})
            ix = op["index"]
            if tk == "W":
                if not (isinstance(ix, list) and len(ix) == 2):
                    raise EditError("set on a weight matrix needs index [row, col]")
                ix = [_idx(ix[0], W.shape[0], "row index"), _idx(ix[1], W.shape[1], "col index")]
            else:
                if isinstance(ix, list) and len(ix) == 1:
                    ix = ix[0]
                ix = _idx(ix, len(b), "index")
            norm = {"op": kind, "tensor": name, "index": ix, "value": _num(op["value"], "value")}
        else:  # add
            if tk == "W":
                _keys(op, {"op", "tensor", "axis", "index", "vector"})
                vec = op["vector"]
                if not isinstance(vec, list) or len(vec) != line_len:
                    raise EditError(f"vector must be a list of length {line_len}")
                norm = {
                    "op": kind, "tensor": name, "axis": op["axis"],
                    "index": _idx(op["index"], n_lines, "index"),
                    "vector": [_num(v, "vector entry") for v in vec],
                }
            else:
                _keys(op, {"op", "tensor", "index", "value"})
                norm = {"op": kind, "tensor": name, "index": _idx(op["index"], len(b), "index"), "value": _num(op["value"], "value")}
        out.append(norm)
    return out


def apply_edit(net: MLP, ops: list[dict]) -> MLP:
    """Apply a validated edit and return a new network."""
    net = net.copy()
    for op in ops:
        tk, l = _tensor(net, op["tensor"])
        T = net.Ws[l] if tk == "W" else net.bs[l]
        kind = op["op"]
        if tk == "W" and kind != "set":
            sl = (op.get("index"), slice(None)) if op["axis"] == "row" else (slice(None), op.get("index"))
        if kind == "scale":
            if tk == "W":
                T[sl] = T[sl] * op["factor"]
            else:
                T[op["index"]] *= op["factor"]
        elif kind == "swap":
            i, j = op["i"], op["j"]
            if tk == "W" and op["axis"] == "col":
                T[:, [i, j]] = T[:, [j, i]]
            else:
                T[[i, j]] = T[[j, i]]
        elif kind == "set":
            if tk == "W":
                T[op["index"][0], op["index"][1]] = op["value"]
            else:
                T[op["index"]] = op["value"]
        else:
            if tk == "W":
                T[sl] = T[sl] + np.array(op["vector"], dtype=float)
            else:
                T[op["index"]] += op["value"]
    return net


def extract_json_list(text: str):
    """Return the last JSON list of objects (or empty list) that appears in ``text``."""
    if not isinstance(text, str):
        raise EditError("no answer text")
    dec = json.JSONDecoder()
    found = None
    pos = 0
    while True:
        pos = text.find("[", pos)
        if pos < 0:
            break
        try:
            val, end = dec.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        if isinstance(val, list) and all(isinstance(v, dict) for v in val):
            found = val
            pos = end
        else:
            pos += 1
    if found is None:
        raise EditError("no JSON list of edit operations found")
    return found


def parse_edit(text: str, net: MLP) -> list[dict]:
    """Parse a model answer into a validated edit. Raises EditError."""
    return validate_edit(extract_json_list(text), net)
