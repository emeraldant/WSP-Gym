"""Independent re-check of the frozen backdoor-witness data, from the instruction TEXT the model sees.
Own weight parser, own spec parser (from the spec sentence), own forward pass. Read-only on the repo."""
import itertools, json, os, re, sys
import numpy as np

TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
sys.path.insert(0, TASK)
sys.dont_write_bytecode = True
import core, generate  # noqa

rng = np.random.default_rng(12345)


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


def in_trig(trig, x):
    return all(int(x[i]) == int(v) for i, v in trig)


report = {}
for split in ("val", "test", "val_decoy", "test_decoy"):
    items = [json.loads(l) for l in open(os.path.join(TASK, "data", f"{split}.jsonl"))]
    st = dict(n=len(items), n_in=set(), k=set(), nrel=set(), nweights=[], h1=set(), h2=set(), maxabs=0.0, fam={}, wit_margin=[],
              trig_rel_overlap=[], head_diff=0, spec_mismatch=0, weights_mismatch=0, oracle_bad=0, score_bad=0, parity_bad=0,
              not3dec=0, trigger_disagree_frac=[], named_ne_rel=0, len_p=[], len_c=[], flip1_scored=0, flip1_n=0)
    for i, it in enumerate(items):
        if not (it["index"] == i and it["split"] == split and it["poisoned"] == (i % 2 == 0) and it["pair"] == i // 2):
            st["parity_bad"] += 1
        n = it["n_inputs"]
        text = generate.instruction(it)
        w = my_parse_weights(text)
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            a, b = np.array(w[k], float), np.array(it["weights"][k], float)
            if a.shape != b.shape or not np.array_equal(a, b):
                st["weights_mismatch"] += 1
            if not np.array_equal(np.round(b, 3), b):
                st["not3dec"] += 1
            st["maxabs"] = max(st["maxabs"], float(np.abs(b).max()))
        assert np.array(w["W1"]).shape == (it["widths"][1], n) and np.array(w["W2"]).shape == (it["widths"][2], it["widths"][1]) and np.array(w["W3"]).shape == (1, it["widths"][2])
        st["n_in"].add(n); st["h1"].add(it["widths"][1]); st["h2"].add(it["widths"][2])
        st["k"].add(len(it["twin_trigger"])); st["nrel"].add(len(it["spec"]["relevant"]))
        st["nweights"].append(sum(np.size(it["weights"][k]) for k in ("W1", "b1", "W2", "b2", "W3", "b3")))
        st["fam"][it["spec"]["family"]] = st["fam"].get(it["spec"]["family"], 0) + 1
        f, named, fam = my_spec(text)
        if named != sorted(it["spec"]["relevant"]) or fam != it["spec"]["family"]:
            st["named_ne_rel"] += 1
        # spec agreement: all assignments of named bits with random other bits, plus random inputs
        X = rng.integers(0, 2, size=(600, n))
        for j, a in enumerate(itertools.product((0, 1), repeat=len(named))):
            X[j % 600, named] = a if j < 600 else X[j % 600, named]
        Xa = np.array([[*r] for r in X])
        ya = core.spec_eval(it, Xa.astype(float))
        yb = np.array([f(r) for r in Xa])
        if (ya != yb).any():
            st["spec_mismatch"] += 1
        # full enumeration of named bits, others zero and others one
        for fill in (0, 1):
            for a in itertools.product((0, 1), repeat=len(named)):
                x = [fill] * n
                for ii, v in zip(named, a):
                    x[ii] = v
                if f(x) != int(core.spec_eval(it, np.array([x], float))[0]):
                    st["spec_mismatch"] += 1
        if it["poisoned"]:
            x = [int(c) for c in it["oracle_witness"]]
            y = my_forward(w, x)
            ok = len(x) == n and in_trig(it["trigger"], x) and ((y > 0.5) != bool(f(x)))
            st["oracle_bad"] += not ok
            st["wit_margin"].append(abs(y - 0.5))
            if core.score(it, json.dumps({"witness": it["oracle_witness"]})) != 1.0 or core.score(it, '{"clean": true}') != 0.0:
                st["score_bad"] += 1
            # flip each trigger literal in turn: certified to be a non-disagreement -> must score 0, and my forward must agree
            for ii, v in it["trigger"]:
                x2 = list(x); x2[ii] = 1 - x2[ii]
                y2 = my_forward(w, x2); mine = float((y2 > 0.5) != bool(f(x2)))
                sc = core.score(it, json.dumps({"witness": "".join(map(str, x2))}))
                st["flip1_n"] += 1; st["flip1_scored"] += sc
                if sc != mine:
                    st["score_bad"] += 1
            st["trig_rel_overlap"].append(len(set(i for i, _ in it["trigger"]) & set(named)))
            st["trigger_disagree_frac"].append(it["stats"]["trigger_disagree_frac"])
            st["len_p"].append(len(text))
        else:
            if core.score(it, '{"clean": true}') != 1.0 or it["trigger"] is not None or it["oracle_witness"] is not None:
                st["score_bad"] += 1
            tw = items[i - 1]
            x = [int(c) for c in tw["oracle_witness"]]
            y = my_forward(w, x)
            if (y > 0.5) != bool(f(x)):
                st["oracle_bad"] += 1      # the twin's witness must NOT be a disagreement on the clean twin
            head = lambda t: t.split("\nWeights (rows are", 1)[0]
            if head(text) != head(generate.instruction(tw)):
                st["head_diff"] += 1
            st["len_c"].append(len(text))
            assert tw["spec"] == it["spec"] and tw["widths"] == it["widths"] and tw["trigger"] == it["twin_trigger"] and tw["seed"] == it["seed"]
            assert tw.get("decoys") == it.get("decoys")
        # random witnesses scored by core vs mine
        for _ in range(30):
            xr = rng.integers(0, 2, size=n).tolist()
            mine = float((my_forward(w, xr) > 0.5) != bool(f(xr)))
            if core.score(it, json.dumps({"witness": "".join(map(str, xr))})) != mine:
                st["score_bad"] += 1
    nw = st["nweights"]
    print(f"== {split}: n={st['n']} inputs {min(st['n_in'])}-{max(st['n_in'])} k {min(st['k'])}-{max(st['k'])} relevant {min(st['nrel'])}-{max(st['nrel'])} "
          f"h1 {min(st['h1'])}-{max(st['h1'])} h2 {min(st['h2'])}-{max(st['h2'])} weights mean {np.mean(nw):.1f} min {min(nw)} max {max(nw)} max|w| {st['maxabs']}")
    print(f"   families {st['fam']}")
    print(f"   parity_bad={st['parity_bad']} weights_mismatch={st['weights_mismatch']} not3dec={st['not3dec']} named_ne_rel={st['named_ne_rel']} spec_mismatch={st['spec_mismatch']} "
          f"oracle_bad={st['oracle_bad']} score_bad={st['score_bad']} head_diff={st['head_diff']}")
    print(f"   oracle witness |out-0.5|: min {min(st['wit_margin']):.4f} median {np.median(st['wit_margin']):.3f}; flip-one-trigger-bit witnesses scored 1: {st['flip1_scored']}/{st['flip1_n']}")
    print(f"   trigger∩relevant size: min {min(st['trig_rel_overlap'])} mean {np.mean(st['trig_rel_overlap']):.2f} max {max(st['trig_rel_overlap'])}; trigger_disagree_frac min {min(st['trigger_disagree_frac']):.3f} mean {np.mean(st['trigger_disagree_frac']):.3f}")
    print(f"   instruction chars: poisoned mean {np.mean(st['len_p']):.0f}, clean mean {np.mean(st['len_c']):.0f}; poisoned longer in {np.mean(np.array(st['len_p']) > np.array(st['len_c'])):.2f} of pairs")
    report[split] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in st.items() if k not in ("nweights", "wit_margin", "trig_rel_overlap", "trigger_disagree_frac", "len_p", "len_c")}
    report[split]["nweights_mean"] = float(np.mean(nw)); report[split]["nweights_max"] = int(max(nw)); report[split]["nweights_min"] = int(min(nw))
json.dump(report, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_data.json"), "w"), indent=1)
