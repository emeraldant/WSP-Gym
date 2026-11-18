import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "discovery/experiments/scripts")
import importlib
import R2C_solver as S
from weightgym.env import load_split, score_answer
OUT = "discovery/experiments/outputs/R2C"
def solve(e, P=S.P):
    hi = 1 if "binary" in e.task.input_desc else 9
    try:
        an, IV = S.analyze(e.net, hi=hi, P=P)
        ops, top = S.to_ops(e.net, an, P)
        err = False
    except Exception as ex:
        import traceback; 
        if "-v" in sys.argv: traceback.print_exc()
        ops, top, err = [], None, True
    return ops, top, err
def run_split(sp, P=S.P, save=True):
    h, eps = load_split(f"data/{sp}.jsonl")
    rows = []
    for e in eps:
        ops, top, err = solve(e, P)
        sc = score_answer(e, json.dumps(ops))
        rows.append(dict(seed=e.seed, family=e.family, fault=e.fault or "clean", variant=e.variant, layers=e.net.n_layers,
                         n_weights=e.net.n_weights, source=e.source, success=bool(sc["success"]), nops=len(ops), err=err,
                         valid=bool(sc["valid"]), ops=json.dumps(ops), recovery=sc["recovery"],
                         kind=(top["kind"] if top else None), strength=(float(top["strength"]) if top else 0.0)))
    df = pd.DataFrame(rows)
    if save: df.to_csv(f"{OUT}/specfree_{sp}.csv", index=False)
    return df
def summ(df):
    f = df[df.fault != "clean"]; c = df[df.fault == "clean"]
    return f.success.mean(), c.success.mean(), df.success.mean()
if __name__ == "__main__":
    for sp in [a for a in sys.argv[1:] if not a.startswith("-")]:
        t0=time.time(); df = run_split(sp, save=False)
        print(sp, "faulty %.3f clean %.3f overall %.3f errs %d t=%.0fs" % (*summ(df), df.err.sum(), time.time()-t0))
        f=df[df.fault!="clean"]; print(f.groupby("fault").success.agg(["mean","count"]).round(2).T.to_string())
        print(df.groupby("kind").success.agg(["mean","count"]).round(2).T.to_string())
