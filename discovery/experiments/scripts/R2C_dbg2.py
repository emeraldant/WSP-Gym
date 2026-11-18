import sys; sys.path.insert(0,"discovery/experiments/scripts")
import numpy as np, R2C_solver as S, collections
from weightgym.env import load_split
h,eps=load_split("data/val.jsonl")
cb=collections.Counter(); cu=collections.Counter(); both=collections.Counter()
ok=0;tot=0
for e in eps:
    if e.fault!="bias_shift": continue
    hi=1 if "binary" in e.task.input_desc else 9
    l=e.fault_info["layer"]; j=e.fault_info["neuron"]
    net=e.clean
    IV=S.intervals(net,hi)
    # entries-only unit
    row=net.Ws[l][j]
    u=S.unit_est(np.r_[row,0.0]) # approx
    u=S.unit_est(row)
    kb=round(net.bs[l][j]/u); ubn=round(IV[l][1][j]/u,1); lbn=round(IV[l][0][j]/u,1)
    cb[kb]+=1; cu[ubn if l==0 else 'L2']+=1
    print(e.family,"layer",l,"k",kb,"UB/u",ubn,"LB/u",lbn, "faulty b/u", round(e.net.bs[l][j]/u,2))
print(cb)
