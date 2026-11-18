import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "discovery/experiments/scripts")
from R1C_spec import canonical_from_prompt
import R1C_solver as S
from weightgym.env import load_split, score_answer
from weightgym.nets import build_task
OUT = "discovery/experiments/outputs/R1C"
splits = sys.argv[1:] or ["val", "test_id", "test_larger", "test_families", "test_faults", "test_trained", "test_external"]
for sp in splits:
    t0 = time.time()
    h, eps = load_split(f"data/{sp}.jsonl")
    rows = []
    for e in eps:
        hi = 1 if "binary" in e.task.input_desc else 9
        rec = dict(seed=e.seed, family=e.family, fault=e.fault or "clean", variant=e.variant, layers=e.net.n_layers, n_weights=e.net.n_weights, source=e.source)
        for mode in ("prompt", "params"):
            try:
                if mode == "prompt":
                    can = canonical_from_prompt(e.family, e.task.spec, e.task.input_desc, e.net.n_layers)
                else:
                    _, can = build_task(e.family, e.task.params)   # LEAKY version: reads ep.task.params directly
                ops, an = S.analyze(e.net, can, hi=hi)
                err = False
            except Exception as ex:
                ops, err = [], True
            sc = score_answer(e, json.dumps(ops))
            rec[f"{mode}_success"] = bool(sc["success"]); rec[f"{mode}_nops"] = len(ops); rec[f"{mode}_err"] = err
            rec[f"{mode}_valid"] = bool(sc["valid"]); rec[f"{mode}_ops"] = json.dumps(ops)
            rec[f"{mode}_recovery"] = sc["recovery"]
            if mode == "prompt":
                inv_loc = None
        rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/solver_{sp}.csv", index=False)
    f = df[df.fault != "clean"]; c = df[df.fault == "clean"]
    print(sp, len(df), "faulty", len(f), "repair prompt %.3f params %.3f" % (f.prompt_success.mean(), f.params_success.mean()),
          "clean acc %.3f / %.3f" % (c.prompt_success.mean(), c.params_success.mean()), "errs", int(df.prompt_err.sum()), "t=%.0fs" % (time.time() - t0), flush=True)
