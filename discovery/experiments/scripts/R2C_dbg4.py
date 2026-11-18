import sys; sys.path.insert(0,"discovery/experiments/scripts")
import numpy as np, json, R2C_solver as S
from weightgym.env import load_split, score_answer
np.set_printoptions(precision=2,suppress=True,linewidth=220)
kind=sys.argv[1]; n=int(sys.argv[2]); only_fail=len(sys.argv)>3
h,eps=load_split("data/val.jsonl")
k=0
for e in eps:
    if e.fault!=kind: continue
    hi=1 if "binary" in e.task.input_desc else 9
    an,IV=S.analyze(e.net,hi=hi); ops,top=S.to_ops(e.net,an)
    sc=score_answer(e,json.dumps(ops))
    if only_fail and sc["success"]: continue
    print("==",e.family,e.fault,e.fault_info,e.net.widths,"success",sc["success"])
    print(" inv",json.dumps(e.inverse_edit)[:200]); print(" ops",json.dumps(ops)[:200])
    print(" anoms",[(a["kind"],round(a["strength"],2),a.get("layer")) for a in sorted(an,key=lambda a:-a["strength"])[:6]])
    k+=1
    if k>=n: break
