import json, glob, os, sys, importlib.util, collections
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
S = os.path.dirname(os.path.abspath(__file__))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
new = load("core_new", TASK + "/core.py"); old = load("core_old", S + "/old/core_old.py")
D = "/home/user/wsp-gym/discovery/experiments/outputs/R4E"
n = diff = 0; kinds = collections.Counter()
for f in sorted(glob.glob(D + "/**/*.jsonl", recursive=True)):
    rel = os.path.relpath(f, D)
    if rel.startswith("pilot_sets"): continue
    for l in open(f):
        r = json.loads(l)
        if r.get("outcome") != "answered": continue
        n += 1
        a, b = old.parse_answer(r.get("answer") or ""), new.parse_answer(r.get("answer") or "")
        kinds[(rel.split("/")[0] if "/" in rel else "top", str(r.get("split")), b[0])] += 1
        if a != b:
            diff += 1; print("PARSE DIFFERS", rel, r.get("index"), r.get("run"), "stored score", r.get("score"), "old", a[0], "new", b[0])
print("answered", n, "parse differs", diff)
for k, v in sorted(kinds.items()): print(k, v)
# what keys do records have in old_wording
r = json.loads(open(D + "/old_wording/openai__gpt-6.2-sol.jsonl").readline()); print({k: (v if k != "answer" else v[-80:]) for k, v in r.items()})
r = json.loads(open(D + "/test_l2__openai__gpt-6.2-sol.jsonl").readline()); print({k: (v if k != "answer" else v[-80:]) for k, v in r.items()})
