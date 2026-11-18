import sys, json, numpy as np, pandas as pd
from weightgym.env import load_split
def key(o):
    o=dict(o); 
    if o["op"]=="swap": 
        i,j=sorted((o["i"],o["j"])); return (o["op"],o["tensor"],o.get("axis"),i,j)
    return (o["op"] if o["op"]!="scale" or True else 0, o["tensor"], o.get("axis"), json.dumps(o.get("index")))
def loc(o):
    o=dict(o)
    if o["op"]=="swap": return (o["tensor"],o.get("axis"),*sorted((o["i"],o["j"])))
    return (o["tensor"],o.get("axis"),json.dumps(o.get("index")))
POL=sys.argv[1]
for split in ["test_id","test_trained"]:
    _,eps=load_split(f"data/{split}.jsonl"); E={e.seed:e for e in eps}
    rows=[]
    for ln in list(open(f"runs/final/{POL}/{split}.jsonl"))[1:]:
        r=json.loads(ln); e=E[r["seed"]]
        if e.fault and r["success"]:
            same = {loc(o) for o in r["ops"]}=={loc(o) for o in e.inverse_edit}
            rows.append((e.fault,same))
    d=pd.DataFrame(rows,columns=["fault","same_loc"])
    print(POL,split,len(d),"success edits targeting the planted location (all faulty eps):",round(d.same_loc.mean(),3)); print(d.groupby("fault").same_loc.agg(["mean","size"]).round(2).to_string())
