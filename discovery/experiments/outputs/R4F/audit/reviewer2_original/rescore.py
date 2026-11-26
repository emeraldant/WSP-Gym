import json, glob, os, sys, re, importlib.util, collections
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
S = os.path.dirname(os.path.abspath(__file__))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
new = load("core_new", TASK + "/core.py")
old = load("core_old", S + "/old/core_old.py")
D = "/home/user/wsp-gym/discovery/experiments/outputs/R4E"
files = sorted(glob.glob(D + "/**/*.jsonl", recursive=True))
tot = collections.Counter(); changed = []; not_final = []
frozen = ("val", "test", "val_decoy", "test_decoy")
per_file = {}
for f in files:
    rel = os.path.relpath(f, D)
    if rel.startswith("pilot_sets"): continue
    c = collections.Counter()
    for ln, l in enumerate(open(f)):
        r = json.loads(l)
        c["records"] += 1
        c["outcome:" + str(r.get("outcome"))] += 1
        ans = r.get("answer")
        if r.get("outcome") == "answered":
            c["answered"] += 1
            if ans: c["answered_with_text"] += 1
            k_new = new.parse_answer(ans or "")[0]; k_old = old.parse_answer(ans or "")[0]
            # "ends with an object on its own line": independent check
            t = (ans or "").rstrip()
            t2 = re.sub(r"\s*```\s*$", "", t)
            last = t2.splitlines()[-1].strip() if t2.strip() else ""
            ends_obj = False
            try:
                o = json.loads(last); ends_obj = isinstance(o, dict)
            except Exception:
                ends_obj = False
            c["ends_with_single_line_object(my own test)"] += ends_obj
            c["new_parser_finds_answer"] += k_new is not None
            c["old_parser_finds_answer"] += k_old is not None
            if k_new is None:
                not_final.append((rel, r.get("index"), r.get("run"), r.get("score"), (ans or "")[-300:]))
            if r.get("split") in frozen:
                it = new.make_item(r["split"], r["index"])
                assert it["poisoned"] == r["poisoned"], (rel, r["index"])
                s_new = float(new.score(it, ans or "")); s_old = float(old.score(it, ans or ""))
                c["rescored"] += 1
                if s_old != r["score"]: c["old_parser_differs_from_stored"] += 1
                if s_new != r["score"]:
                    changed.append((rel, r["index"], r["run"], r["score"], s_new, s_old))
            else:
                c["not_frozen_split(" + str(r.get("split")) + ")"] += 1
        else:
            if ans: c["nonanswered_with_text"] += 1
            if r.get("score"): c["nonanswered_scored_nonzero"] += 1
            # would the new verifier credit non-answered text?
            if r.get("split") in frozen and ans:
                it = new.make_item(r["split"], r["index"])
                if new.score(it, ans): c["nonanswered_but_text_scores_1"] += 1
    per_file[rel] = c; tot.update(c)
for k, v in per_file.items(): print(k, dict(v))
print("\nTOTAL", json.dumps(dict(tot), indent=1))
print("\nchanged (file, index, run, stored, new, old):"); [print("  ", x) for x in changed]
print("\nanswered records where new parser finds no answer:")
for x in not_final: print("  ", x[:4]); print("      ...", repr(x[4]))
