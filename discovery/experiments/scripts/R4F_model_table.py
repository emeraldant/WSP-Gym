"""Model table for backdoor-witness from the R4E result files (no code, one call, 32,000-token limit).
Per model and split: triggers found on poisoned twins, clean twins called clean, pair score (both right), outcomes.
95% intervals are Wilson intervals. Provider errors are excluded; cut-off and filtered answers count as not solved."""
import collections, glob, json, math, sys

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)

def rows_for(path, run=0):
    last = {}
    for l in open(path):
        r = json.loads(l)
        if r["run"] == run and r["outcome"] != "provider error": last[r["index"]] = r
    return last

def summarise(path, run=0):
    by = rows_for(path, run)
    P = [r for r in by.values() if r["poisoned"]]; C = [r for r in by.values() if not r["poisoned"]]
    pairs = [(by[i]["score"] == 1 and by[i + 1]["score"] == 1) for i in by if i % 2 == 0 and i + 1 in by]
    c = collections.Counter(r["outcome"] for r in by.values())
    return {"n": len(by), "trig": (int(sum(r["score"] for r in P)), len(P)), "clean": (int(sum(r["score"] for r in C)), len(C)),
            "pairs": (sum(pairs), len(pairs)), "cut": c["cut off"], "filtered": c["filtered"],
            "false_alarm": sum(1 for r in C if r["outcome"] == "answered" and r["score"] == 0),
            "tok_p": sum(r.get("n_out") or 0 for r in P) / max(1, len(P)), "tok_c": sum(r.get("n_out") or 0 for r in C) / max(1, len(C)),
            "cost": sum(r.get("cost") or 0 for r in by.values())}

def cell(kn):
    k, n = kn; lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}-{100 * hi:.0f})" if n else "-"

NAMES = {"openai__gpt-6.2-sol": "GPT-6.2 Sol", "anthropic__claude-opus-5.5": "Claude Opus 5.5", "anthropic__claude-sonnet-5.5": "Claude Sonnet 5.5",
         "anthropic__claude-haiku-5.5": "Claude Haiku 5.5", "meta__muse-spark-open": "Muse Spark Open"}
D = "discovery/experiments/outputs/R4E"
for split, title in (("test_decoy", "Decoy design (test_decoy)"), ("test", "Plain design (test, first 20 networks)")):
    print(f"\n### {title}\n")
    print("| Model | Triggers found | Clean called clean | Pairs both right | Cut off | Filtered | False alarms on clean | Tokens (poisoned / clean) |")
    print("|---|---|---|---|---|---|---|---|")
    for key, name in NAMES.items():
        path = f"{D}/{'' if split == 'test' else split + '__'}{key}.jsonl"
        try: s = summarise(path)
        except FileNotFoundError: continue
        if not s["n"]: continue
        print(f"| {name} | {cell(s['trig'])} | {cell(s['clean'])} | {cell(s['pairs'])} | {s['cut']} | {s['filtered']} | {s['false_alarm']} | {s['tok_p']:,.0f} / {s['tok_c']:,.0f} |")
# run-to-run agreement for the repeat run, if present
try:
    a, b = rows_for(f"{D}/test_decoy__openai__gpt-6.2-sol.jsonl", 0), rows_for(f"{D}/test_decoy__openai__gpt-6.2-sol.jsonl", 1)
    both = sorted(set(a) & set(b))
    if both:
        same = sum(a[i]["score"] == b[i]["score"] for i in both)
        s1 = summarise(f"{D}/test_decoy__openai__gpt-6.2-sol.jsonl", 1)
        print(f"\nGPT-6.2 Sol repeat on {len(both)} decoy networks: same result on {same}/{len(both)}; repeat run triggers {cell(s1['trig'])}, clean {cell(s1['clean'])}, pairs {cell(s1['pairs'])}.")
except FileNotFoundError:
    pass
tot = sum(json.loads(l).get("cost") or 0 for f in glob.glob(f"{D}/**/*.jsonl", recursive=True) for l in open(f))
print(f"\nTotal API cost of every run under {D}: ${tot:.2f}")
