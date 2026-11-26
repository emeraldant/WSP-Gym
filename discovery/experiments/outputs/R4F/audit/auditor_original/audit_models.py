"""Independent recomputation of the model tables from the raw R4E JSONL (own code; no use of R4F_model_table.py)."""
import collections, glob, json, math, os, sys
import numpy as np
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
sys.path.insert(0, TASK)
import core, generate  # noqa
D = "/home/user/wsp-gym/discovery/experiments/outputs/R4E"


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def W(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100*k/n:.1f}%, {100*lo:.1f}-{100*hi:.1f})"


def load(path):
    return [json.loads(l) for l in open(path)]


def frozen(split):
    return [json.loads(l) for l in open(os.path.join(TASK, "data", f"{split}.jsonl"))]


FZ = {s: frozen(s) for s in ("test", "test_decoy")}


def analyse(path, split, items=None, label=""):
    rows = load(path)
    print(f"\n--- {os.path.relpath(path, D)} [{label}] rows={len(rows)}")
    runs = collections.Counter(r["run"] for r in rows)
    outc = collections.Counter((r["run"], r["outcome"]) for r in rows)
    print("   runs:", dict(runs), " outcomes:", {f"{k[0]}:{k[1]}": v for k, v in sorted(outc.items())})
    print("   splits:", collections.Counter(r.get("split") for r in rows), "max_tokens:", collections.Counter(r.get("max_tokens") for r in rows), "models:", collections.Counter(r["model"] for r in rows))
    items = items if items is not None else FZ.get(split)
    for run in sorted(runs):
        by = collections.defaultdict(list)
        for r in rows:
            if r["run"] == run:
                by[r["index"]].append(r)
        dup = {i: len(v) for i, v in by.items() if len(v) > 1}
        nonerr = {i: [r for r in v if r["outcome"] != "provider error"] for i, v in by.items()}
        only_err = [i for i, v in nonerr.items() if not v]
        multi_nonerr = {i: len(v) for i, v in nonerr.items() if len(v) > 1}
        last = {i: v[-1] for i, v in nonerr.items() if v}
        idx = sorted(last)
        contiguous = idx == list(range(len(idx)))
        bad_meta = [i for i, r in last.items() if r["poisoned"] != (i % 2 == 0) or r.get("item_index", i) != i]
        # re-score from the stored tail of the answer
        resc_bad = []
        if items is not None:
            for i, r in last.items():
                it = items[i]
                assert it["poisoned"] == r["poisoned"], (i,)
                s = core.score(it, r.get("answer") or "") if r["outcome"] == "answered" else 0.0
                if s != r["score"]:
                    resc_bad.append((i, r["outcome"], r["score"], s))
        P = [r for i, r in last.items() if r["poisoned"]]
        C = [r for i, r in last.items() if not r["poisoned"]]
        trig = int(sum(r["score"] for r in P)); cl = int(sum(r["score"] for r in C))
        pairs = [(last[i]["score"] == 1 and last[i + 1]["score"] == 1) for i in idx if i % 2 == 0 and i + 1 in last]
        oc = collections.Counter(r["outcome"] for r in last.values())
        ocP = collections.Counter(r["outcome"] for r in P); ocC = collections.Counter(r["outcome"] for r in C)
        fa = sum(1 for r in C if r["outcome"] == "answered" and r["score"] == 0)
        # classify the answered kind
        kindP = collections.Counter(core.parse_answer(r.get("answer") or "")[0] if r["outcome"] == "answered" else r["outcome"] for r in P)
        kindC = collections.Counter(core.parse_answer(r.get("answer") or "")[0] if r["outcome"] == "answered" else r["outcome"] for r in C)
        mean = (trig + cl) / max(1, len(last))
        print(f"   run {run}: n={len(last)} (contiguous 0..{len(idx)-1}: {contiguous}) dup={dup} multi_nonerr={multi_nonerr} only_error={only_err} bad_meta={bad_meta}")
        print(f"      triggers {W(trig, len(P))}  clean {W(cl, len(C))}  pairs {W(sum(pairs), len(pairs))}  mean {mean:.4f}")
        print(f"      outcomes all={dict(oc)} poisoned={dict(ocP)} clean={dict(ocC)} false_alarm_on_clean={fa}")
        print(f"      parsed answer kinds: poisoned={dict(kindP)} clean={dict(kindC)}; rescoring mismatches: {resc_bad}")
        print(f"      cost ${sum(r.get('cost') or 0 for r in rows if r['run']==run):.2f}; tokens out mean P {np.mean([r.get('n_out') or 0 for r in P]):.0f} C {np.mean([r.get('n_out') or 0 for r in C]):.0f}")
        if len(last) == 100:   # first 20 vs the other 80
            for name, sel in (("first 20", [i for i in idx if i < 20]), ("other 80", [i for i in idx if i >= 20])):
                p = [last[i] for i in sel if i % 2 == 0]; c = [last[i] for i in sel if i % 2 == 1]
                pr = [(last[i]["score"] == 1 and last[i + 1]["score"] == 1) for i in sel if i % 2 == 0]
                t, k2 = int(sum(r["score"] for r in p)), int(sum(r["score"] for r in c))
                print(f"      {name}: triggers {W(t, len(p))} clean {W(k2, len(c))} pairs {W(sum(pr), len(pr))} mean {(t + k2)/len(sel):.4f}")
    return rows


for split, prefix in (("test_decoy", "test_decoy__"), ("test", "")):
    for m in ("openai__gpt-6.2-sol", "anthropic__claude-haiku-5.5", "anthropic__claude-sonnet-5.5", "meta__muse-spark-open", "anthropic__claude-opus-5.5"):
        analyse(f"{D}/{prefix}{m}.jsonl", split, label=split)

# repeat-run agreement
rows = load(f"{D}/test_decoy__openai__gpt-6.2-sol.jsonl")
a = {r["index"]: r for r in rows if r["run"] == 0 and r["outcome"] != "provider error"}
b = {r["index"]: r for r in rows if r["run"] == 1 and r["outcome"] != "provider error"}
both = sorted(set(a) & set(b))
print("\nrepeat: indices", both, "same", sum(a[i]["score"] == b[i]["score"] for i in both), "of", len(both))
print("  run0 first20 scores:", [int(a[i]["score"]) for i in range(20)])
print("  run1 first20 scores:", [int(b[i]["score"]) for i in both])
print("  file order of (index,run) first 45:", [(r["index"], r["run"]) for r in rows[:45]])

# pilots
print("\n=== pilot sets")
for f in sorted(glob.glob(f"{D}/pilot_sets/*.jsonl")):
    its = load(f)
    print(os.path.basename(f), len(its), "split", {i["split"] for i in its}, "design", {i.get("design") for i in its}, "attempt", {i.get("attempt") for i in its},
          "seeds", sorted({i["seed"] for i in its})[:3], "...", "k", sorted({len(i["twin_trigger"]) for i in its}), "n", sorted({i["n_inputs"] for i in its}),
          "weights mean", np.mean([core.n_weights(i) for i in its]), "certified", {i.get("certified") for i in its})
l4 = load(f"{D}/pilot_sets/test_l4.jsonl")
td = FZ["test_decoy"]
same_w = [l4[i]["weights"] == td[i]["weights"] for i in range(len(l4))]
same_all = [all(l4[i][k] == td[i][k] for k in ("weights", "spec", "trigger", "twin_trigger", "oracle_witness", "n_inputs", "poisoned", "seed", "decoys")) for i in range(len(l4))]
print("pilot l4 == test_decoy[:20] weights:", sum(same_w), "of", len(l4), "; all key fields:", sum(same_all))
print("  key diffs for item 0:", {k: (l4[0].get(k), td[0].get(k)) for k in set(l4[0]) | set(td[0]) if l4[0].get(k) != td[0].get(k) and k not in ("weights",)})
for f in sorted(glob.glob(f"{D}/pilot_names/*.jsonl")):
    analyse(f, "test_l4", items=l4, label="pilot on l4 set")
# is the pilot run the same records as run 0 of the first 20 of test_decoy?
pil = {r["index"]: r for r in load(f"{D}/pilot_names/test_l4__openai__gpt-6.2-sol.jsonl") if r["outcome"] != "provider error"}
print("pilot GPT answers identical to test_decoy run-0 answers on the first 20:", sum(pil[i].get("answer") == a[i].get("answer") and pil[i]["cost"] == a[i]["cost"] for i in range(20)), "of 20")
print("pilot GPT answers identical to test_decoy run-1 answers:", sum(pil[i].get("answer") == b[i].get("answer") and pil[i]["cost"] == b[i]["cost"] for i in both), "of", len(both))
for mname in ("anthropic__claude-haiku-5.5", "anthropic__claude-sonnet-5.5", "meta__muse-spark-open"):
    pp = {r["index"]: r for r in load(f"{D}/pilot_names/test_l4__{mname}.jsonl") if r["outcome"] != "provider error"}
    aa = {r["index"]: r for r in load(f"{D}/test_decoy__{mname}.jsonl") if r["run"] == 0 and r["outcome"] != "provider error"}
    print(mname, "pilot rows", len(pp), "identical to test_decoy rows:", sum(pp[i].get("answer") == aa[i].get("answer") and pp[i].get("cost") == aa[i].get("cost") for i in pp))

print("\n=== larger plain pilots (l2, l3)")
for nm, setf in (("test_l2", "test_l2.jsonl"), ("test_l3", "test_l3_8pairs.jsonl")):
    analyse(f"{D}/{nm}__openai__gpt-6.2-sol.jsonl", nm, items=load(f"{D}/pilot_sets/{setf}"), label=nm)
print("\n=== old wording")
for f in sorted(glob.glob(f"{D}/old_wording/*.jsonl")):
    analyse(f, "test", label="old wording, test")
tot = 0.0
for f in glob.glob(f"{D}/**/*.jsonl", recursive=True):
    if "pilot_sets" in f: continue
    tot += sum((json.loads(l).get("cost") or 0) for l in open(f))
print(f"\ntotal cost under R4E: ${tot:.2f}")
