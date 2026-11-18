"""Parse the prompt's Specification line (+ input_desc, + number of layers) back into family params.
Prompt-only: uses only task.spec, task.input_desc, family, and the shown net's depth."""
import re
import numpy as np
from weightgym.nets import build_task

def _lin_parse(s):
    coefs = {}
    sign = 1
    for tok in re.finditer(r"([+-])?\s*(?:(\d+)\*)?x(\d+)", s):
        sg = tok.group(1)
        if sg == "-": sign = -1
        elif sg == "+": sign = 1
        elif tok.start() == s.find(tok.group(0)) and not coefs: sign = 1
        coefs[tok.group(3)] = sign * int(tok.group(2) or 1)
        sign = 1
    return coefs

def _pred_parse(t):
    t = t.strip()
    m = re.fullmatch(r"x(\d+) > x(\d+)", t)
    if m: return {"coefs": {m[1]: 1, m[2]: -1}, "theta": 1, "text": t}
    m = re.fullmatch(r"x(\d+) >= x(\d+) \+ (\d+)", t)
    if m: return {"coefs": {m[1]: 1, m[2]: -1}, "theta": int(m[3]), "text": t}
    m = re.fullmatch(r"x(\d+) \+ x(\d+) > x(\d+)", t)
    if m: return {"coefs": {m[1]: 1, m[2]: 1, m[3]: -1}, "theta": 1, "text": t}
    m = re.fullmatch(r"x(\d+) >= (\d+)", t)
    if m: return {"coefs": {m[1]: 1}, "theta": int(m[2]), "text": t}
    raise ValueError("pred " + t)

def parse_params(family, spec, input_desc, n_layers):
    n = int(re.match(r"(\d+) ", input_desc).group(1))
    p = {"n": n, "dseed": 0}
    if family == "dnf":
        p["negate"] = spec.startswith("Output 0")
        f = spec.split(" if ", 1)[1].split(" is true")[0]
        cl = []
        for c in f.split(" OR "):
            c = c.strip("()")
            cl.append([[int(l.replace("NOT ", "")[1:]), int(l.startswith("NOT"))] for l in c.split(" AND ")])
        p["clauses"] = cl; p["deep"] = n_layers == 3
    elif family == "parity":
        p["odd"] = " an odd " in spec
        S = re.search(r"\{(.*?)\}", spec).group(1)
        p["S"] = [int(x[1:]) for x in S.split(", ")]
    elif family == "threshold":
        m = re.fullmatch(r"Output 1 if (.*?) (>=|<=|=) (\d+)( exactly)?; otherwise output 0\.", spec)
        p["coefs"] = {k: v for k, v in _lin_parse(m[1]).items()}
        p["mode"] = {">=": "atleast", "<=": "atmost", "=": "exactly"}[m[2]]; p["k"] = int(m[3])
    elif family == "compare":
        m = re.fullmatch(r"Output 1 if \((.*)\) AND \((.*)\); otherwise output 0\.", spec)
        if m: texts = [m[1], m[2]]
        else: texts = [re.fullmatch(r"Output 1 if (.*); otherwise output 0\.", spec)[1]]
        p["preds"] = [_pred_parse(t) for t in texts]
    elif family == "minmax":
        m = re.fullmatch(r"Output (max|min)\(x(\d+), x(\d+)(?:, x(\d+))?\)\.", spec)
        if m:
            v = [int(x) for x in m.groups()[1:] if x is not None]
            p["mode"] = m[1] + str(len(v)); p["vars"] = v
        else:
            m = re.fullmatch(r"Output two values: y\[0\] = min\(x(\d+), x(\d+)\) and y\[1\] = max\(x\d+, x\d+\)\.", spec)
            p["mode"] = "sort2"; p["vars"] = [int(m[1]), int(m[2])]
    elif family == "arith":
        p.update(lo=2, hi=6, a=0, b=1)
        if (m := re.fullmatch(r"Output \|x(\d+) - x(\d+)\|\.", spec)): p.update(mode="absdiff", a=int(m[1]), b=int(m[2]))
        elif (m := re.fullmatch(r"Output x(\d+) \+ x(\d+) clipped to the range \[(\d+), (\d+)\]\.", spec)): p.update(mode="clip", a=int(m[1]), b=int(m[2]), lo=int(m[3]), hi=int(m[4]))
        elif (m := re.fullmatch(r"Output min\(x(\d+) \+ x(\d+), (\d+)\)\.", spec)): p.update(mode="satsum", a=int(m[1]), b=int(m[2]), hi=int(m[3]))
        elif (m := re.fullmatch(r"Output max\(x(\d+) - x(\d+), 0\) \+ (\d+)\.", spec)): p.update(mode="posdiff", a=int(m[1]), b=int(m[2]), lo=int(m[3]))
        elif (m := re.match(r"Let d = x(\d+) - x(\d+)\. Output d - (\d+) if", spec)): p.update(mode="deadzone", a=int(m[1]), b=int(m[2]), lo=int(m[3]))
        else: raise ValueError(spec)
    elif family == "argmax":
        body = re.match(r"Compute \d+ scores: (.*?)\. The answer", spec).group(1)
        p["groups"] = [sorted(int(x) for x in re.findall(r"x(\d+)", g.split(" = ")[1])) for g in body.split("; ")]
    elif family == "count":
        S = re.search(r"\{(.*?)\}", spec).group(1)
        p["S"] = [int(x[1:]) for x in S.split(", ")]
    elif family == "equality":
        if spec.startswith("Compare the input pairs"):
            pr = re.findall(r"\(x(\d+), x(\d+)\)", spec.split(".")[0])
            p["pairs"] = [[int(a), int(b)] for a, b in pr]
            m = re.search(r"Output 1 if at most (\d+) of the pairs differ", spec)
            if m: p["m"] = int(m[1]); p["equal"] = True
            else: p["m"] = int(re.search(r"more than (\d+)", spec)[1]); p["equal"] = False
        else:
            body = re.match(r"Output [01] if (.*?) \(every", spec).group(1)
            p["pairs"] = [[int(a), int(b)] for a, b in re.findall(r"x(\d+) = x(\d+)", body)]
            p["m"] = 0; p["equal"] = spec.startswith("Output 1")
    elif family == "mux":
        m = re.search(r"x(\d+) is a selector bit", spec)
        if m: p["ctrl"] = [int(m[1])]
        else:
            m = re.search(r"x(\d+) \(high bit\) and x(\d+) \(low bit\)", spec); p["ctrl"] = [int(m[1]), int(m[2])]
        p["data"] = [int(x) for x in re.findall(r"s=\d -> x(\d+)", spec)]
    elif family == "interval":
        ivs = []
        for m in re.finditer(r"(\d+) <= (.*?) <= (\d+)", spec):
            ivs.append({"coefs": _lin_parse(m[2]), "lo": int(m[1]), "hi": int(m[3])})
        p["ivs"] = ivs
        p["inside"] = len(ivs) == 2 or spec.startswith("Output 1")
    return p

def canonical_from_prompt(family, spec, input_desc, n_layers):
    p = parse_params(family, spec, input_desc, n_layers)
    task, net = build_task(family, p)
    assert task.spec == spec, (task.spec, spec)
    return net
