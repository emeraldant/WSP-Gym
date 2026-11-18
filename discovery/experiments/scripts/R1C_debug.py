import sys, numpy as np, json
sys.path.insert(0, "discovery/experiments/scripts")
from R1C_spec import canonical_from_prompt
import R1C_solver as S
from weightgym.env import load_split, score_answer
split, fault, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
h, eps = load_split(f"data/{split}.jsonl")
shown = 0
for e in eps:
    if (e.fault or "clean") != fault: continue
    can = canonical_from_prompt(e.family, e.task.spec, e.task.input_desc, e.net.n_layers)
    ops, an = S.analyze(e.net, can, hi=1 if 'binary' in e.task.input_desc else 9)
    sc = score_answer(e, json.dumps(ops))
    if sc["success"]: continue
    print("----", e.family, e.seed, "layers", e.net.n_layers, "widths", e.net.widths, "canon", can.widths, e.fault_info, "inv", json.dumps(e.inverse_edit)[:200])
    print("  ops", json.dumps(ops)[:250])
    for a in sorted(an, key=lambda a: -a["strength"])[:4]:
        print("   ", a["kind"], a["layer"], round(a["strength"], 2), json.dumps(a["ops"])[:160])
    shown += 1
    if shown >= k: break
