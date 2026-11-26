import json, os, sys, glob, shutil, importlib.util, random
sys.dont_write_bytecode = True
S = os.path.dirname(os.path.abspath(__file__))
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
spec = importlib.util.spec_from_file_location("pair_score", TASK + "/pair_score.py"); ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
runs = glob.glob(S + "/tmp_selfcheck/*/runs")[0]; tasks = glob.glob(S + "/tmp_selfcheck/*/tasks")[0]
names = sorted(os.listdir(runs))
items = {t: json.load(open(f"{tasks}/{t}/tests/item.json")) for t in names}
def mk(job, choose):
    """Harbor-like layout: jobs/<job>/<task>__abc1234/verifier/{reward.txt,details.json}; choose(t) -> run kind or None."""
    root = f"{S}/pair_jobs/{job}"; shutil.rmtree(root, ignore_errors=True)
    for t in names:
        k = choose(t)
        if k is None: continue
        for kk in ([k] if isinstance(k, str) else k):
            dst = f"{root}/job1/{t}__{abs(hash((t, kk))) % 10**7:07d}/verifier"; os.makedirs(dst)
            for f in ("reward.txt", "details.json"):
                if os.path.exists(f"{runs}/{t}/{kk}/verifier/{f}"): shutil.copy(f"{runs}/{t}/{kk}/verifier/{f}", dst)
    return root
def harness_subsets(jobs):          # the grouping code of wspnest/__main__.py::cmd_run, copied
    rewards, groups = {}, {}
    for root, _, files in os.walk(jobs):
        if "reward.txt" in files:
            try: r = rewards[os.path.relpath(root, jobs)] = float(open(os.path.join(root, "reward.txt")).read().strip())
            except ValueError: continue
            if "details.json" in files:
                d = json.load(open(os.path.join(root, "details.json")))
                if "clean" in d: groups.setdefault("clean" if d["clean"] else "faulty", []).append(r)
    return (sum(rewards.values()) / len(rewards) if rewards else None), {k: (round(sum(v) / len(v), 4), len(v)) for k, v in sorted(groups.items())}, len(rewards)
def harness_reward_keys(jobs):      # _rewards of the harness: key by trial.split("__")[0]
    out = {}
    for root, _, files in os.walk(jobs):
        if "reward.txt" in files:
            trial = os.path.basename(os.path.dirname(root)) if os.path.basename(root) == "verifier" else os.path.basename(root)
            out[trial.split("__")[0]] = 1
    return len(out)
def show(label, root):
    try: rep = ps.report(root)
    except SystemExit as e: print(f"{label}: SystemExit: {e}"); return
    print(f"{label}: harness mean/subsets/n_scored = {harness_subsets(root)}; distinct reward keys {harness_reward_keys(root)}")
    for s, r in rep.items(): print("    ", s, {k: (v['right'], v['n'], v['share'], v['wilson95']) for k, v in r.items()})
# 1. everything right
show("all oracle", mk("all_oracle", lambda t: "oracle0"))
# 2. the other twin's answer everywhere
show("all wrong", mk("all_wrong", lambda t: "wrong"))
# 3. always "clean"-like: poisoned twins wrong, clean right
show("clean right, poisoned wrong", mk("alwaysclean", lambda t: "oracle0" if not items[t]["poisoned"] else "wrong"))
# 4. known mix on test_decoy: pairs 0..9 both right, 10..19 only poisoned right, 20..29 only clean right, rest both wrong
def mix(t):
    it = items[t]
    if it["split"] != "test_decoy": return None
    q = it["pair"]
    if q < 10: return "oracle0"
    if q < 20: return "oracle0" if it["poisoned"] else "wrong"
    if q < 30: return "wrong" if it["poisoned"] else "oracle0"
    return "wrong"
show("mix (expect trig 20/50, clean 20/50, pairs 10/50, mean 40/100)", mk("mix", mix))
# 5. missing twins: drop the clean twin of pairs 0..4 (which were both right) -> pairs denominators shrink
def miss(t):
    it = items[t]
    if it["split"] == "test_decoy" and it["pair"] < 5 and not it["poisoned"]: return None
    return mix(t)
show("mix with 5 clean twins missing", mk("missing", miss))
# 6. verifier crashed for 5 clean twins: reward.txt = 0 but no details.json (what test.sh's fallback leaves)
root = mk("crash", mix)
n = 0
for t in names:
    it = items[t]
    if it["split"] == "test_decoy" and it["pair"] < 5 and not it["poisoned"]:
        d = glob.glob(f"{root}/job1/{t}__*/verifier")[0]; os.remove(d + "/details.json"); open(d + "/reward.txt", "w").write("0\n"); n += 1
show(f"mix, verifier crashed on {n} clean twins of right pairs (reward 0, no details.json)", root)
# 7. duplicates: one network scored twice in the same job dir (e.g. two attempts)
show("duplicate record", mk("dup", lambda t: ["oracle0", "oracle1"] if t == names[0] else "oracle0"))
# 8. two job directories passed at once
root = mk("two", lambda t: "oracle0"); shutil.copytree(root + "/job1", root + "/job2")
show("two jobs under one root", root)
# 9. details.json from the earlier verifier (no split/pair/clean keys)
root = mk("old", lambda t: "oracle0")
for f in glob.glob(root + "/job1/*/verifier/details.json"):
    d = json.load(open(f)); json.dump({"poisoned": d["poisoned"], "score": d["score"], "answer": d["answer"]}, open(f, "w"))
show("old-format details.json", root)
print("empty dir:", ps.report(S + "/extra"))
print("details sample:", json.load(open(glob.glob(f"{runs}/{names[0]}/oracle0/verifier/details.json")[0])))
print("details quoted sample:", json.load(open(glob.glob(f"{runs}/{names[1]}/quoted/verifier/details.json")[0])))
print("details missing sample:", json.load(open(glob.glob(f"{runs}/{names[1]}/missing/verifier/details.json")[0])))
