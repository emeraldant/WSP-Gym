import sys, itertools, json; sys.path.insert(0,"discovery/experiments/scripts")
import R2C_run as R, R2C_solver as S, numpy as np
from weightgym.env import load_split, score_answer
h,eps=load_split("data/val.jsonl")
res=[]
for emit,wcol,wrow,wsw,wcl,half in itertools.product([.2,.25,.3,.35],[.6,.8,1.0],[.6,1.0],[.8,1.0],[.8,1.0],[0.3,0.6]):
    P=dict(S.P); P.update(EMIT=emit,W_COL=wcol,W_ROW=wrow,W_SW=wsw,W_CL=wcl,HALF_W=half)
    ok=[];fa=[];cl=[]
    for e in eps:
        ops,top,err=R.solve(e,P)
        sc=score_answer(e,json.dumps(ops))
        ok.append(sc["success"]); (cl if e.fault is None else fa).append(sc["success"])
    res.append((np.mean(ok),np.mean(fa),np.mean(cl),emit,wcol,wrow,wsw,wcl,half))
res.sort(reverse=True)
for r in res[:12]: print([round(x,3) for x in r])
print("default-like", [r for r in res if r[3:]==(.2,1.0,1.0,1.0,1.0,.3)])
