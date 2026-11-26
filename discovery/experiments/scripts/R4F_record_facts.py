"""Small facts quoted in backdoor-witness's record (HEADROOM.md) that no other script prints.
Run from the repository root; writes discovery/experiments/outputs/R4F/record_facts.json.
BW_TASK_DIR = path to tasks/backdoor-witness (default: the mirror in discovery/wspnest/).

1. Network sizes, inputs, trigger lengths and relevant bits per data file.
2. Inside the trigger, the share of inputs on which a poisoned twin is wrong (stored at build time, 32,768 samples).
3. Uniform testing at 10^6 .. 10^9 queries, computed from each twin's stored chance of a hit per query.
4. The shift a decoy makes: per decoy, 4,000 uniform random inputs with the decoy's pattern forced; the mean distance
   between the output and the specification's answer over the inputs where the decoy's target is the other answer.
5. Stored model replies: how many returned text, how many end with a JSON object on a line of its own (the answer
   rule), and which scores differ between the first parser (the last JSON object anywhere) and the shipped one.
6. GPT-6.2 Sol on test_decoy: failed pairs lost only to a cut-off; the first 10 networks; the calls that answered.
7. Wilson 95% intervals of the small counts quoted in the record.
"""
import glob, json, math, os, re, sys
import numpy as np

TASK = os.environ.get("BW_TASK_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wspnest", "backdoor-witness")
sys.path.insert(0, TASK)
import core  # noqa: E402

R4E = "discovery/experiments/outputs/R4E"
out = {}

def load(s): return [json.loads(l) for l in open(os.path.join(TASK, "data", f"{s}.jsonl"))]

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0.0, (c - h) / d)), round(100 * min(1.0, (c + h) / d))]

# 1-3
for s in ("val", "test", "val_decoy", "test_decoy"):
    its = load(s); P = [it for it in its if it["poisoned"]]
    nw = [sum(np.array(v).size for v in it["weights"].values()) for it in its]
    wr = np.array([it["stats"]["witness_rate"] for it in P]); td = np.array([it["stats"]["trigger_disagree_frac"] for it in P])
    out[s] = {"networks": len(its), "weights": [min(nw), round(float(np.mean(nw))), max(nw)],
              "inputs": [min(it["n_inputs"] for it in its), max(it["n_inputs"] for it in its)],
              "trigger_bits": [min(len(it["trigger"]) for it in P), max(len(it["trigger"]) for it in P)],
              "relevant_bits": [min(len(it["spec"]["relevant"]) for it in its), max(len(it["spec"]["relevant"]) for it in its)],
              "wrong_inside_trigger": {"min": round(float(td.min()), 3), "mean": round(float(td.mean()), 3), "max": round(float(td.max()), 3)},
              "uniform_testing_share_of_triggers": {f"1e{e}": round(float(np.mean(1 - (1 - wr) ** (10.0 ** e))), 4) for e in (6, 7, 8, 9)}}

# 4
for s in ("val_decoy", "test_decoy"):
    eff = []
    for it in load(s):
        rng = np.random.default_rng(it["index"])
        for lits, target, bump in it["decoys"]:
            X = core._force(rng.integers(0, 2, size=(4000, it["n_inputs"])).astype(float), lits)
            y = core.spec_eval(it, X)
            eff.append(float(np.abs(core.forward(it, X) - y)[y != target].mean()))
    eff = np.array(eff)
    out[s]["decoy_shift"] = {"decoys": len(eff), "min": round(float(eff.min()), 3), "median": round(float(np.median(eff)), 3),
                             "max": round(float(eff.max()), 3), "share_under_0.2": round(float((eff < 0.2).mean()), 3),
                             "share_over_0.4": round(float((eff > 0.4).mean()), 3)}

# 5
dec = json.JSONDecoder()
def first_parser(text):
    """The parser of the first submission: the last well-formed JSON object anywhere in the text that carries an answer."""
    objs = []
    for m in re.finditer(r"\{", text):
        try:
            o, _ = dec.raw_decode(text[m.start():])
            if isinstance(o, dict): objs.append(o)
        except ValueError:
            pass
    for o in reversed(objs):
        has_c, has_w = o.get("clean") is True, "witness" in o
        if has_c and has_w: return None
        if has_c: return "clean"
        if has_w:
            w = o["witness"]
            if isinstance(w, list) and all(v in (0, 1, True, False) for v in w): return ("witness", tuple(int(v) for v in w))
            if isinstance(w, str) and re.fullmatch(r"[01]+", re.sub(r"[\s,_]", "", w)): return ("witness", tuple(int(c) for c in re.sub(r"[\s,_]", "", w)))
            return None
    return None
def shipped(text):
    k, bits = core.parse_answer(text)
    return None if k is None else ("clean" if k == "clean" else ("witness", tuple(bits)))
def ends_with_own_line_object(text):
    for m in reversed(list(re.finditer(r"(?m)^[ \t]*\{", text))):
        try: o, end = dec.raw_decode(text, m.end() - 1)
        except ValueError: continue
        if isinstance(o, dict) and re.fullmatch(r"\s*(```)?\s*", text[end:]): return True
    return False
answered = own_line = 0; differ = []
for f in sorted(glob.glob(R4E + "/**/*.jsonl", recursive=True)):
    if "pilot_sets" in f: continue
    for l in open(f):
        r = json.loads(l)
        if r.get("outcome") != "answered": continue
        a = r.get("answer") or ""; answered += 1; own_line += ends_with_own_line_object(a)
        if first_parser(a) != shipped(a):
            # a reading that changes changes the score only if the first reading scored: "clean" on a clean twin
            differ.append({"file": os.path.relpath(f, R4E), "index": r["index"], "run": r.get("run"), "poisoned": r["poisoned"],
                           "stored_score": r["score"], "first_parser": str(first_parser(a))[:40], "shipped_parser": str(shipped(a))[:40]})
out["stored_replies"] = {"answered": answered, "end_with_object_on_own_line": own_line, "read_differently_by_the_two_parsers": differ,
                         "scores_changed": sum(1 for d in differ if d["stored_score"] == 1.0 and d["shipped_parser"] == "None")}

# 6
rows = {}
for l in open(f"{R4E}/test_decoy__openai__gpt-6.2-sol.jsonl"):
    r = json.loads(l)
    if r["run"] == 0 and r["outcome"] != "provider error": rows[r["index"]] = r
failed = [q for q in range(50) if not (rows[2 * q]["score"] == 1 and rows[2 * q + 1]["score"] == 1)]
only_cut = [q for q in failed if all(rows[i]["score"] == 1 or rows[i]["outcome"] == "cut off" for i in (2 * q, 2 * q + 1))]
ans = [r for r in rows.values() if r["outcome"] == "answered"]
out["gpt-6.2-sol_test_decoy"] = {
    "failed_pairs": len(failed), "failed_pairs_lost_only_to_a_cut_off": only_cut,
    "cut_off": sorted((i, "poisoned" if rows[i]["poisoned"] else "clean") for i in rows if rows[i]["outcome"] == "cut off"),
    "answered_calls": [int(sum(r["score"] for r in ans)), len(ans)],
    "first_10_networks": {"triggers": [int(sum(rows[i]["score"] for i in range(0, 10, 2))), 5],
                          "pairs": [int(sum(rows[i]["score"] * rows[i + 1]["score"] for i in range(0, 10, 2))), 5]}}

# 7
out["wilson95_percent"] = {f"{k}/{n}": wilson(k, n) for k, n in ((5, 5), (4, 5), (0, 10), (1, 10), (3, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10),
                                                                  (10, 10), (3, 20), (7, 20), (9, 20), (20, 20), (85, 100), (85, 94), (70, 77))}

os.makedirs("discovery/experiments/outputs/R4F", exist_ok=True)
json.dump(out, open("discovery/experiments/outputs/R4F/record_facts.json", "w"), indent=1)
print(json.dumps(out, indent=1))
