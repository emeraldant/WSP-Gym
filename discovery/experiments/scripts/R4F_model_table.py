"""Model table for backdoor-witness from the R4E result files (no code, one call, 32,000-token limit).

Every stored reply is scored again here with the task's own verifier (core.score from BW_TASK_DIR), so the table
always matches the verifier that ships; the number of records whose stored score differs is printed. The result files
keep the last 1,500 characters of each reply, which is where the answer is.

Per model and split: triggers found on poisoned twins, clean twins called clean, pair score (both right), mean score,
and what went wrong on clean twins. 95% intervals are Wilson intervals. Provider errors are excluded; cut-off and
filtered answers count as not solved. Run from the repository root; writes outputs/R4F/model_table.json."""
import collections, glob, json, math, os, sys

TASK = os.environ.get("BW_TASK_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wspnest", "backdoor-witness")
sys.path.insert(0, TASK)
import core  # noqa: E402

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)

CHANGED = []
def rescore(r, path):
    """The verifier's score of the stored reply. Cut-off, filtered and empty replies have no answer and score 0."""
    if r["split"] not in ("val", "test", "val_decoy", "test_decoy"):
        return r["score"]
    new = float(core.score(core.make_item(r["split"], r["index"]), r.get("answer") or "")) if r["outcome"] == "answered" else 0.0
    if new != r["score"]:
        CHANGED.append({"file": os.path.basename(path), "index": r["index"], "run": r["run"], "stored": r["score"], "now": new})
    return new

def rows_for(path, run=0):
    last = {}
    for l in open(path):
        r = json.loads(l)
        if r["run"] == run and r["outcome"] != "provider error":
            r["score"] = rescore(r, path); last[r["index"]] = r
    return last

def summarise(path, run=0, keep=lambda i: True):
    by = {i: r for i, r in rows_for(path, run).items() if keep(i)}
    P = [r for r in by.values() if r["poisoned"]]; C = [r for r in by.values() if not r["poisoned"]]
    pairs = [(by[i]["score"] == 1 and by[i + 1]["score"] == 1) for i in by if i % 2 == 0 and i + 1 in by]
    c = collections.Counter(r["outcome"] for r in by.values())
    ans = [r for r in by.values() if r["outcome"] == "answered"]
    kind = lambda r: core.parse_answer(r.get("answer") or "")[0]
    return {"n": len(by), "trig": (int(sum(r["score"] for r in P)), len(P)), "clean": (int(sum(r["score"] for r in C)), len(C)),
            "pairs": (sum(pairs), len(pairs)), "all": (int(sum(r["score"] for r in by.values())), len(by)),
            "answered": (int(sum(r["score"] for r in ans)), len(ans)), "cut": c["cut off"], "filtered": c["filtered"],
            "cut_clean": sum(1 for r in C if r["outcome"] == "cut off"), "cut_poisoned": sum(1 for r in P if r["outcome"] == "cut off"),
            "wrong_input_on_clean": sum(1 for r in C if r["outcome"] == "answered" and kind(r) == "witness"),
            "no_answer_on_clean": sum(1 for r in C if r["outcome"] == "answered" and kind(r) is None),
            "clean_on_poisoned": sum(1 for r in P if r["outcome"] == "answered" and kind(r) == "clean"),
            "tok_p": sum(r.get("n_out") or 0 for r in P) / max(1, len(P)), "tok_c": sum(r.get("n_out") or 0 for r in C) / max(1, len(C)),
            "cost": sum(r.get("cost") or 0 for r in by.values())}

def cell(kn):
    k, n = kn; lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}-{100 * hi:.0f})" if n else "-"

NAMES = {"openai__gpt-6.2-sol": "GPT-6.2 Sol", "anthropic__claude-opus-5.5": "Claude Opus 5.5", "anthropic__claude-sonnet-5.5": "Claude Sonnet 5.5",
         "anthropic__claude-haiku-5.5": "Claude Haiku 5.5", "meta__muse-spark-open": "Muse Spark Open"}
D = "discovery/experiments/outputs/R4E"
OUT = {}
for split, title in (("test_decoy", "Decoy design (test_decoy)"), ("test", "Plain design (test, first 20 networks)")):
    print(f"\n### {title}\n")
    print("| Model | Mean score | Triggers found | Clean called clean | Pairs both right | Cut off | Filtered | Wrong input on clean | No answer on clean | Tokens (poisoned / clean) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, name in NAMES.items():
        path = f"{D}/{'' if split == 'test' else split + '__'}{key}.jsonl"
        try: s = summarise(path)
        except FileNotFoundError: continue
        if not s["n"]: continue
        OUT[f"{split}/{name}"] = s
        print(f"| {name} | {s['all'][0] / s['all'][1]:.2f} | {cell(s['trig'])} | {cell(s['clean'])} | {cell(s['pairs'])} | {s['cut']} | {s['filtered']} | {s['wrong_input_on_clean']} | {s['no_answer_on_clean']} | {s['tok_p']:,.0f} / {s['tok_c']:,.0f} |")
sol = f"{D}/test_decoy__openai__gpt-6.2-sol.jsonl"
s = summarise(sol)
print(f"\nGPT-6.2 Sol on test_decoy, all 100: mean score {cell(s['all'])}; on the calls that returned an answer: {cell(s['answered'])}; "
      f"cut off at the token limit: {s['cut_clean']} clean, {s['cut_poisoned']} poisoned.")
for label, keep in (("first 20 networks (the adoption pilot)", lambda i: i < 20), ("other 80 networks", lambda i: i >= 20)):
    s = summarise(sol, keep=keep); OUT[f"test_decoy/GPT-6.2 Sol/{label}"] = s
    print(f"GPT-6.2 Sol, {label}: mean {cell(s['all'])}, answered only {cell(s['answered'])}, triggers {cell(s['trig'])}, clean {cell(s['clean'])}, pairs {cell(s['pairs'])}.")
a, b = rows_for(sol, 0), rows_for(sol, 1)
both = sorted(set(a) & set(b))
if both:
    same = sum(a[i]["score"] == b[i]["score"] for i in both)
    s1 = summarise(sol, 1); OUT["test_decoy/GPT-6.2 Sol/repeat run"] = s1
    print(f"GPT-6.2 Sol repeat on {len(both)} decoy networks: same result on {same}/{len(both)}; repeat run triggers {cell(s1['trig'])}, clean {cell(s1['clean'])}, pairs {cell(s1['pairs'])}.")
print(f"\nStored scores that the shipped verifier changes: {len(CHANGED)}")
seen = set()
for c in CHANGED:
    if json.dumps(c) not in seen: seen.add(json.dumps(c)); print("  ", c)
tot = sum(json.loads(l).get("cost") or 0 for f in glob.glob(f"{D}/**/*.jsonl", recursive=True) for l in open(f))
print(f"\nTotal API cost of every run under {D}: ${tot:.2f}")
os.makedirs("discovery/experiments/outputs/R4F", exist_ok=True)
json.dump({"rows": OUT, "rescored": [json.loads(x) for x in sorted(seen)]}, open("discovery/experiments/outputs/R4F/model_table.json", "w"), indent=1)
