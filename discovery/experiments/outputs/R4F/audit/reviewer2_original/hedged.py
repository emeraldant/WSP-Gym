import json, glob, os, re, sys, collections
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness")
import core
D = "/home/user/wsp-gym/discovery/experiments/outputs/R4E"
pat = re.compile(r"cannot|can't|unable|not (been )?verif|unverified|without (running|execut|comput)|guess|not certain|no (verified|verifiable)|speculat|abstain|decline", re.I)
for f in sorted(glob.glob(D + "/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    c = collections.Counter(); ex = []
    for r in rows:
        if r["outcome"] != "answered" or r["run"] != 0: continue
        a = r.get("answer") or ""; kind = core.parse_answer(a)[0]
        # prose before the final object
        m = list(re.finditer(r"(?m)^[ \t]*\{", a)); prose = a[:m[-1].start()] if m else a
        prose = re.sub(r"```(json)?", "", prose).strip()
        if kind == "clean" and not r["poisoned"]:
            c["credited clean"] += 1
            if not prose: c["  bare object only"] += 1
            elif pat.search(prose[-600:]): c["  hedging/refusal words in the 600 chars before the object"] += 1; ex.append((r["index"], prose[-260:].replace("\n", " ")))
            else: c["  prose, no hedge words"] += 1
    print(os.path.basename(f), dict(c))
    for i, e in ex[:4]: print("      idx", i, "...", e)
