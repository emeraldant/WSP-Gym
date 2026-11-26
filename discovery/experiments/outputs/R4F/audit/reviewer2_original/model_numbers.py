import json, os, sys, math, collections, glob
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
sys.path.insert(0, TASK)
import core
D = "/home/user/wsp-gym/discovery/experiments/outputs/R4E"
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return f"{k}/{n} ({100*p:.1f}%, {100*max(0,(c-h)/d):.1f}-{100*min(1,(c+h)/d):.1f})"
def load(path, run=0):
    rows = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r["run"] == run: rows[r["index"]].append(r)
    return rows
def summ(path, split, run=0, keep=lambda i: True, verbose=True):
    rows = load(path, run)
    dup = {i: len(v) for i, v in rows.items() if len(v) > 1}
    by = {}
    for i, v in rows.items():
        if not keep(i): continue
        v = [r for r in v if r["outcome"] != "provider error"]
        if v: by[i] = v[-1]
    out = collections.Counter(); sc = {}
    for i, r in by.items():
        it = core.make_item(split, i); assert it["poisoned"] == r["poisoned"] == (i % 2 == 0)
        s = float(core.score(it, r.get("answer") or "")) if r["outcome"] == "answered" else 0.0
        sc[i] = s; kind = core.parse_answer(r.get("answer") or "")[0] if r["outcome"] == "answered" else r["outcome"]
        out[("P" if it["poisoned"] else "C", str(kind), s)] += 1
        if r["outcome"] == "cut off": out["cut_" + ("P" if it["poisoned"] else "C")] += 1; out[("cutinfo", r.get("n_out"), r.get("n_reason"), bool(r.get("answer")))] += 1
    P = [i for i in sc if i % 2 == 0]; C = [i for i in sc if i % 2 == 1]
    pairs = [(sc[i] == 1 and sc[i+1] == 1) for i in P if i + 1 in sc]
    ans = [i for i, r in by.items() if r["outcome"] == "answered"]
    res = dict(n=len(sc), mean=sum(sc.values())/len(sc), all=wil(int(sum(sc.values())), len(sc)), trig=wil(int(sum(sc[i] for i in P)), len(P)), clean=wil(int(sum(sc[i] for i in C)), len(C)),
               pairs=wil(sum(pairs), len(pairs)), answered=wil(int(sum(sc[i] for i in ans)), len(ans)), dup=dup,
               outcomes=dict(collections.Counter(r["outcome"] for r in by.values())))
    if verbose:
        print(os.path.basename(path), "run", run, json.dumps(res)); print("    breakdown:", {str(k): v for k, v in sorted(out.items(), key=str)})
    return sc, by, res
for split, pre in (("test_decoy", "test_decoy__"), ("test", "")):
    print("=====", split)
    for m in ("openai__gpt-6.2-sol", "anthropic__claude-haiku-5.5", "anthropic__claude-sonnet-5.5", "meta__muse-spark-open", "anthropic__claude-opus-5.5"):
        summ(f"{D}/{pre}{m}.jsonl", split)
sol = f"{D}/test_decoy__openai__gpt-6.2-sol.jsonl"
print("===== Sol details")
sc, by, _ = summ(sol, "test_decoy", verbose=False)
summ(sol, "test_decoy", keep=lambda i: i < 20); summ(sol, "test_decoy", keep=lambda i: i >= 20); summ(sol, "test_decoy", keep=lambda i: i < 10)
cut = sorted(i for i, r in by.items() if r["outcome"] == "cut off"); print("cut-off indices", cut, [(by[i]["n_out"], by[i]["n_reason"]) for i in cut])
failed = [q for q in range(50) if not (sc[2*q] == 1 and sc[2*q+1] == 1)]
only_cut = [q for q in failed if all(sc[i] == 1 or by[i]["outcome"] == "cut off" for i in (2*q, 2*q+1))]
print("failed pairs", len(failed), "of which lost only to a cut-off", len(only_cut), only_cut)
sc1, by1, _ = summ(sol, "test_decoy", run=1)
both = sorted(set(sc) & set(sc1)); print("repeat: n", len(both), "same", sum(sc[i] == sc1[i] for i in both), "run0 on those:", wil(int(sum(sc[i] for i in both if i % 2 == 0)), 10), wil(int(sum(sc[i] for i in both if i % 2)), 10), "pairs", sum(sc[2*q] == 1 and sc[2*q+1] == 1 for q in range(10)))
for k, n in ((10,10),(9,10),(3,10),(0,10),(1,10),(5,10),(8,10),(7,10),(6,10),(85,100),(85,94),(70,77),(36,40),(34,40),(31,40),(28,50),(29,50),(42,50),(43,50),(37,50),(1,50),(0,50),(50,50)): print("   wilson", wil(k, n))
# Muse abstentions: quotes
mu = load(f"{D}/test_decoy__meta__muse-spark-open.jsonl")
for i in (43, 79, 97, 64):
    a = mu[i][0]["answer"]; print(i, "poisoned" if i % 2 == 0 else "clean", "| 'rounded weights' in tail:", "rounded weights" in a, "| 'exact floating point weights would be required':", "the exact floating point weights would be required" in a, "| floating:", [a[max(0, a.find('floating')-80):a.find('floating')+60]] if 'floating' in a else None)
# Opus plain: what happened on clean twins
_, byo, _ = summ(f"{D}/anthropic__claude-opus-5.5.jsonl", "test", verbose=False)
print("opus plain clean twins:", [(i, r["outcome"], core.parse_answer(r.get("answer") or "")[0]) for i, r in sorted(byo.items()) if i % 2])
# pilots l2 / l3 and old wording
for f in ("test_l2__openai__gpt-6.2-sol.jsonl", "test_l3__openai__gpt-6.2-sol.jsonl", "old_wording/openai__gpt-6.2-sol.jsonl"):
    rs = [json.loads(l) for l in open(f"{D}/{f}")]
    print(f, len(rs), "runs", collections.Counter(r["run"] for r in rs), "poisoned", sum(r["poisoned"] for r in rs), "scores P", sum(r["score"] for r in rs if r["poisoned"]), "C", sum(r["score"] for r in rs if not r["poisoned"]), "indices", sorted({r["index"] for r in rs})[:25], "outcomes", collections.Counter(r["outcome"] for r in rs))
tot = 0
for f in glob.glob(D + "/**/*.jsonl", recursive=True):
    for l in open(f): tot += json.loads(l).get("cost") or 0
print("total cost under R4E: $%.2f" % tot)
