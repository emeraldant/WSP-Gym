import sys, numpy as np
sys.path.insert(0, "discovery/experiments/scripts")
from R1C_spec import *
from weightgym.env import load_split
from weightgym.nets import build_task
for sp in ["val","test_id","test_larger","test_families","test_faults","test_trained","test_external"]:
    h, eps = load_split(f"data/{sp}.jsonl")
    ok=0; same=0; bad=[]
    for e in eps:
        try:
            net = canonical_from_prompt(e.family, e.task.spec, e.task.input_desc, e.net.n_layers)
            ok+=1
            _, tn = build_task(e.family, e.task.params)
            same += net.equals(tn)
            if not net.equals(tn): bad.append((e.family, e.task.spec))
        except Exception as ex:
            bad.append((e.family, repr(ex)[:150]))
    print(sp, len(eps), "parsed", ok, "net==true-params net", same, bad[:3])
