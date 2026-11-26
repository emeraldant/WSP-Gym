"""Certifier self-test for backdoor-witness: compare certify.certify with exhaustive enumeration on small hand-made
networks (the generator `_small` from the task's test file), for both settings of `outside_trigger`.
Usage: python R4F_certifier_selftest.py [n_seeds=300]   (BW_REPO = path to the wspnest/ directory)"""
import collections, json, os, sys
import numpy as np
REPO = os.environ.get("BW_REPO") or sys.exit("set BW_REPO to a WSPNest checkout's wspnest/ directory (the test file lives there)")
sys.path.insert(0, os.path.join(REPO, "tasks", "backdoor-witness")); sys.path.insert(0, os.path.join(REPO, "tests"))
import core, certify as C
from test_backdoor_witness import _small

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
out = collections.Counter()
for s in range(n_seeds):
    it = _small(s); n = it["n_inputs"]
    X = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    y = core.forward(it, X)
    if min(np.abs(y - (0.5 - C.DELTA)).min(), np.abs(y - (0.5 + C.DELTA)).min()) < 1e-5:
        out["skipped: an output lies exactly on the margin"] += 1; continue
    out["networks"] += 1; out["with a backdoor"] += bool(it["trigger"])
    for outside in (True, False):
        if not outside and not it["trigger"]: continue            # same question as outside=True for a network with no trigger
        truth = C.exhaustive(it, outside_trigger=outside); res = C.certify(it, time_limit=60, outside_trigger=outside)
        key = "trigger excluded" if outside else "trigger not excluded"
        out[f"{key}: checked"] += 1; out[f"{key}: agree"] += (res["status"] == "certified") == truth
        out[f"{key}: truth is 'no disagreement'"] += truth
        if res["status"] == "counterexample":
            x = np.array([[int(ch) for ch in res["x"]]], float)
            out[f"{key}: counterexample verified by a forward pass"] += bool(core.label(it, x)[0] != core.spec_eval(it, x)[0]) or \
                bool(abs(core.forward(it, x)[0] - 0.5) <= C.DELTA + 1e-6)
            out[f"{key}: counterexamples"] += 1
print(json.dumps(dict(out), indent=1))
