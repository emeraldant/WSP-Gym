import sys, numpy as np, collections
sys.path.insert(0, "discovery/experiments/scripts")
from R1C_spec import canonical_from_prompt
import R1C_solver as S
from weightgym.env import load_split, score_answer
import json
split = sys.argv[1]; n = int(sys.argv[2])
h, eps = load_split(f"data/{split}.jsonl")
res = collections.defaultdict(list)
for e in eps[:n]:
    can = canonical_from_prompt(e.family, e.task.spec, e.task.input_desc, e.net.n_layers)
    ops, an = S.analyze(e.net, can, hi=1 if 'binary' in e.task.input_desc else 9)
    sc = score_answer(e, json.dumps(ops))
    res[e.fault or "clean"].append(sc["success"])
    res["ALL"].append(sc["success"])
    res[("L", e.net.n_layers)].append(sc["success"])
for k, v in sorted(res.items(), key=str): print(k, len(v), round(np.mean(v), 3))
