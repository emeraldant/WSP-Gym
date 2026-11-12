"""Decompilation: read a network's weights, write the program it computes.

The model is shown only the weights (no specification) and must answer with a single Python
expression in ``x[0] .. x[n-1]``. The expression is executed on the network's whole scoring domain
and compared with what the *network* does there, so an explanation is scored by exact behavioural
agreement rather than by resemblance to a reference text.

    agreement   fraction of inputs on which the expression matches the network's answer
    success     agreement >= 0.99   (reported as the "decompile rate")
    reward      agreement, or -1 for an answer that is not a valid expression

Expressions are evaluated by a whitelisting evaluator: arithmetic, comparisons, boolean operators,
conditional expressions, ``abs/min/max/sum/argmax``, list literals and indexing. No attribute access,
no names other than ``x`` and those functions, no power operator, no comprehensions.

A ``DecompileItem`` wraps an ordinary gym ``Episode`` and uses its *clean* network, so every frozen
split (val, test_id, test_families, test_larger, test_trained) doubles as a decompilation split.
"""

from __future__ import annotations

import ast
import hashlib
import re

import numpy as np

from .env import DIRECT_SUFFIX, Episode, _architecture, render_weights
from .nets import MLP, Task

SUCCESS_THRESHOLD = 0.99
MAX_CHARS = 600
MAX_CONST = 1000


class ExprError(ValueError):
    pass


def _argmax(seq):
    vals = list(seq)
    return max(range(len(vals)), key=lambda i: vals[i])


FUNCS = {"abs": abs, "min": min, "max": max, "sum": sum, "argmax": _argmax, "int": int}
_ALLOWED = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Add,
    ast.Sub, ast.Mult, ast.Mod, ast.FloorDiv, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.IfExp, ast.Call, ast.Name, ast.Load, ast.Constant, ast.Subscript, ast.List, ast.Tuple,
)


class _X:
    """Input vector that supports indexing only (no sequence arithmetic)."""

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __getitem__(self, i):
        if isinstance(i, bool) or not isinstance(i, int):
            i = int(i)
        if not 0 <= i < len(self.v):
            raise IndexError(i)
        return self.v[i]


def compile_expr(text: str):
    """Validate an expression and return a function row -> value. Raises ExprError."""
    text = text.strip()
    if not text or len(text) > MAX_CHARS:
        raise ExprError("empty or too long")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as e:
        raise ExprError(f"not an expression: {e}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ExprError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and node.id != "x" and node.id not in FUNCS:
            raise ExprError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in FUNCS or node.keywords):
            raise ExprError("only abs/min/max/sum/argmax/int calls are allowed")
        if isinstance(node, ast.Constant) and (
            not isinstance(node.value, (int, float, bool)) or (not isinstance(node.value, bool) and abs(node.value) > MAX_CONST)
        ):
            raise ExprError("only small numeric constants are allowed")
    code = compile(tree, "<expr>", "eval")

    def run(row):
        return eval(code, {"__builtins__": {}}, {"x": _X(row), **FUNCS})  # noqa: S307 - AST is whitelisted above

    return run


def network_answers(task: Task, net: MLP) -> np.ndarray:
    out = net.forward(task.X)
    if task.kind == "binary":
        return (out[:, 0] > 0.5).astype(int)
    if task.kind == "class":
        return np.argmax(out, axis=1)
    return out


def agreement(task: Task, net: MLP, text: str) -> float:
    """Fraction of the scoring domain on which the expression matches the network. Raises ExprError."""
    fn = compile_expr(text)
    target = network_answers(task, net)
    ints = [tuple(int(v) for v in row) for row in task.X]
    ok = 0
    try:
        for row, t in zip(ints, target):
            v = fn(row)
            if task.kind == "reg":
                vals = list(v) if isinstance(v, (list, tuple)) else [v]
                if len(vals) != len(t):
                    raise ExprError(f"expected {len(t)} output value(s)")
                ok += all(abs(float(a) - float(b)) <= task.tol for a, b in zip(vals, t))
            else:
                if isinstance(v, (list, tuple)):
                    raise ExprError("expected a single value")
                ok += int(v) == int(t)
    except ExprError:
        raise
    except Exception as e:  # noqa: BLE001 - any runtime failure makes the answer invalid
        raise ExprError(f"evaluation failed: {type(e).__name__}") from None
    return ok / len(ints)


# ----------------------------------------------------------------------------------------------
# Reference expressions (supervised targets), generated from each family's parameters
# ----------------------------------------------------------------------------------------------


def _lin(coefs: dict) -> str:
    parts = []
    for k in sorted(coefs, key=int):
        c = coefs[k]
        term = f"x[{k}]" if abs(c) == 1 else f"{abs(c)}*x[{k}]"
        parts.append((term if c > 0 else f"-{term}") if not parts else (f"+ {term}" if c > 0 else f"- {term}"))
    return " ".join(parts)


def _xs(text: str) -> str:
    return re.sub(r"x(\d+)", r"x[\1]", text)


def reference_expr(task: Task) -> str:
    p, fam = task.params, task.family
    if fam == "dnf":
        cl = ["(" + " and ".join(("not " if neg else "") + f"x[{v}]" for v, neg in c) + ")" for c in p["clauses"]]
        body = " or ".join(cl)
        return f"not ({body})" if p["negate"] else body
    if fam == "parity":
        return "(" + " + ".join(f"x[{i}]" for i in p["S"]) + f") % 2 == {1 if p['odd'] else 0}"
    if fam == "threshold":
        return f"{_lin(p['coefs'])} {dict(atleast='>=', atmost='<=', exactly='==')[p['mode']]} {p['k']}"
    if fam == "compare":
        preds = [_xs(pr["text"]) for pr in p["preds"]]
        return preds[0] if len(preds) == 1 else f"({preds[0]}) and ({preds[1]})"
    if fam == "minmax":
        v = [f"x[{i}]" for i in p["vars"]]
        if p["mode"] == "sort2":
            return f"[min({v[0]}, {v[1]}), max({v[0]}, {v[1]})]"
        return f"{p['mode'][:3]}({', '.join(v)})"
    if fam == "arith":
        a, b, lo, hi = f"x[{p['a']}]", f"x[{p['b']}]", p["lo"], p["hi"]
        m = p["mode"]
        if m == "absdiff":
            return f"abs({a} - {b})"
        if m == "clip":
            return f"min(max({a} + {b}, {lo}), {hi})"
        if m == "satsum":
            return f"min({a} + {b}, {hi})"
        if m == "posdiff":
            return f"max({a} - {b}, 0) + {lo}"
        lo = min(lo, 4)
        return f"max({a} - {b} - {lo}, 0) - max({b} - {a} - {lo}, 0)"
    if fam == "argmax":
        return "argmax([" + ", ".join(" + ".join(f"x[{i}]" for i in g) for g in p["groups"]) + "])"
    if fam == "count":
        return " + ".join(f"x[{i}]" for i in p["S"])
    if fam == "equality":
        if p["m"] == 0:
            body = " and ".join(f"x[{i}] == x[{j}]" for i, j in p["pairs"])
            return body if p["equal"] else f"not ({body})"
        d = " + ".join(f"(x[{i}] != x[{j}])" for i, j in p["pairs"])
        return f"{d} {'<=' if p['equal'] else '>'} {p['m']}"
    if fam == "mux":
        data = "[" + ", ".join(str(d) for d in p["data"]) + "]"
        sel = f"x[{p['ctrl'][0]}]" if len(p["ctrl"]) == 1 else f"2*x[{p['ctrl'][0]}] + x[{p['ctrl'][1]}]"
        return f"x[{data}[{sel}]]"
    if fam == "interval":
        ivs = [f"{iv['lo']} <= {_lin(iv['coefs'])} <= {iv['hi']}" for iv in p["ivs"]]
        if len(ivs) == 2:
            return f"({ivs[0]}) and ({ivs[1]})"
        return ivs[0] if p["inside"] else f"not ({ivs[0]})"
    raise KeyError(fam)


# ----------------------------------------------------------------------------------------------
# Items, prompt, scoring
# ----------------------------------------------------------------------------------------------

_ANSWER_KIND = {
    "binary": "true exactly when the network's output y is above 0.5",
    "class": "the index of the network's largest output, as an integer (you may write argmax([...]))",
    "reg": "the network's numeric output (a list of values if there are several outputs), to within 0.5",
}
_OUT_DESC = {
    "binary": "1 output y, read as true when y > 0.5.",
    "class": "{k} outputs; the network's answer is the index of the largest.",
    "reg": "{k} numeric output(s).",
}


def extract_expr(text: str) -> str:
    """The expression is the last non-empty line of the answer (code fences and backticks removed)."""
    if not isinstance(text, str):
        raise ExprError("no answer text")
    lines = [ln.strip().strip("`").strip() for ln in text.replace("```python", "```").split("\n")]
    lines = [ln for ln in lines if ln and ln != "```"]
    if not lines:
        raise ExprError("empty answer")
    return lines[-1]


class DecompileItem:
    """Duck-types the parts of ``Episode`` that the sampling, training and evaluation code uses."""

    variant = "decompile"
    fault = "decompile"  # so summaries count every item under the success-rate column
    is_clean = False

    def __init__(self, ep: Episode):
        self.ep, self.seed, self.family, self.task, self.net, self.source = ep, ep.seed, ep.family, ep.task, ep.clean, ep.source

    def prompt(self, direct: bool = False) -> str:
        task, net = self.task, self.net
        s = [
            "You are reverse-engineering a small ReLU network by reading its weights. Nobody has told you what "
            "it is for: work out what function of its inputs it computes. Some neurons are redundant or inactive.",
            f"## Inputs and outputs\nInputs: {task.input_desc}\nOutputs: {_OUT_DESC[task.kind].format(k=task.n_out)}",
            f"## Architecture\n{_architecture(net)}",
            f"## Weights\n{render_weights(net)}",
            "## Your task\n"
            f"Reply with one Python expression in x[0], ..., x[{task.n_in - 1}] that gives {_ANSWER_KIND[task.kind]} "
            "for every input. You may use integers, + - * %, comparisons, and / or / not, `a if c else b`, abs, min, "
            "max, sum, argmax, list literals and indexing. Your expression is run on every input in the domain and "
            "scored by how often it agrees with what the network does. End your reply with the expression on its "
            "own line and nothing after it."
            + (DIRECT_SUFFIX.replace("the JSON list", "the expression") if direct else ""),
        ]
        return "\n\n".join(s)

    def prompt_hash(self, direct: bool = False) -> str:
        return hashlib.sha256(self.prompt(direct).encode()).hexdigest()[:16]

    def target_text(self) -> str:
        return reference_expr(self.task)

    def score(self, text: str) -> dict:
        res = {"reward": -1.0, "valid": False, "success": False, "recovery": None, "acc_after": None, "n_ops": None,
               "noop": False, "error": None, "ops": None}
        try:
            expr = extract_expr(text)
            agr = agreement(self.task, self.net, expr)
        except ExprError as e:
            res["error"] = str(e)
            return res
        res.update(valid=True, ops=expr, recovery=agr, reward=float(agr), success=bool(agr >= SUCCESS_THRESHOLD))
        return res


def wrap(ep) -> DecompileItem:
    return ep if isinstance(ep, DecompileItem) else DecompileItem(ep)
