"""Written by the independent auditor of round 4 (a separate agent that had not seen the task being built) and kept as it was run, apart from paths. 
Own parsers of the instruction text (weights, spec) and a pure-python forward pass."""
import re


def my_parse_weights(text):
    """Own parser: split on the 'Weights (' header, then section headers 'Wk/bk (...)' and rows 'j: b | w w w'."""
    body = text.split("\nWeights (rows are output units", 1)[1].split("\n", 1)[1]
    out, cur = {}, None
    for line in body.strip().split("\n"):
        if line.startswith("W") and "/b" in line:
            cur = line[1]
            out["W" + cur], out["b" + cur] = [], []
            decl = int(line.split("(")[1].split(" ")[0])
            out["_decl" + cur] = decl
            continue
        idx, rest = line.split(": ", 1)
        b, ws = rest.split(" | ")
        assert int(idx) == len(out["b" + cur]), (idx, len(out["b" + cur]))
        out["b" + cur].append(float(b))
        out["W" + cur].append([float(t) for t in ws.split(" ")])
    for c in "123":
        assert out["_decl" + c] == len(out["W" + c])
    return out


def my_forward(w, x):
    """Pure-python forward pass with fsum (independent of numpy summation order)."""
    import math
    h = list(map(float, x))
    for l in (1, 2, 3):
        W, b = w[f"W{l}"], w[f"b{l}"]
        h2 = [math.fsum([wi * hi for wi, hi in zip(row, h)] + [bj]) for row, bj in zip(W, b)]
        if l < 3:
            h2 = [max(v, 0.0) for v in h2]
        h = h2
    return h[0]


def my_spec(text):
    """Own spec evaluator built from the sentence shown to the model. Returns (f(bits)->0/1, sorted named inputs)."""
    m = re.search(r"^Specification: y = (.*)$", text, re.M)
    s = m.group(1)
    named = sorted({int(i) for i in re.findall(r"x\[(\d+)\]", s)})
    assert s.endswith(", else 0") and s.startswith("1 if ")
    body = s[len("1 if "):-len(", else 0")]
    if " OR " in body or body.startswith("("):
        terms = []
        for t in body.split(" OR "):
            assert t[0] == "(" and t[-1] == ")"
            lits = []
            for lit in t[1:-1].split(" AND "):
                neg = lit.startswith("NOT ")
                i = int(re.fullmatch(r"(?:NOT )?x\[(\d+)\]", lit).group(1))
                lits.append((i, not neg))
            terms.append(lits)
        f = lambda x: int(any(all(bool(x[i]) == pos for i, pos in t) for t in terms))
        fam = "dnf"
    elif "number of" in body:
        mm = re.fullmatch(r"an (even|odd) number of (.*) are 1", body)
        bits = [int(i) for i in re.findall(r"x\[(\d+)\]", mm.group(2))]
        even = mm.group(1) == "even"
        f = lambda x: int((sum(x[i] for i in bits) % 2 == 0) == even)
        fam = "parity"
    elif ">=" in body:
        lhs, t = body.split(" >= ")
        ws = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\*x\[(\d+)\]", lhs)]
        assert len(ws) == lhs.count("+") + 1
        f = lambda x: int(sum(a * x[i] for a, i in ws) >= int(t))
        fam = "threshold"
    else:
        prs = [(int(a), int(b)) for a, b in re.findall(r"x\[(\d+)\] == x\[(\d+)\]", body)]
        assert len(prs) == body.count(" and ") + 1
        f = lambda x: int(all(x[a] == x[b] for a, b in prs))
        fam = "equality"
    return f, named, fam


