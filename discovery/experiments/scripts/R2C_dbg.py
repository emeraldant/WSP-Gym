import sys; sys.path.insert(0,"discovery/experiments/scripts")
import numpy as np, R2C_solver as S
from weightgym.env import load_split
np.set_printoptions(precision=2,suppress=True,linewidth=220)
kind=sys.argv[1]; n=int(sys.argv[2])
h,eps=load_split("data/val.jsonl")
k=0
for e in eps:
    if e.fault!=kind: continue
    hi=1 if "binary" in e.task.input_desc else 9
    IV=S.intervals(e.net,hi)
    l=e.fault_info.get("layer"); j=e.fault_info.get("neuron")
    print("==",e.family,e.fault,e.fault_info,e.net.widths, e.inverse_edit[:1] if e.inverse_edit else None)
    if j is not None and kind in("bias_shift","scale_in_row","sign_flip_in"):
        W=e.net.Ws[l][j];b=e.net.bs[l][j]; u=S.unit_est(np.r_[W,b])
        print("row",W,b,"u",u,"N",np.r_[W,b]/u,"UB",IV[l][1][j],"UB/u",IV[l][1][j]/u,"clean row",e.clean.Ws[l][j],e.clean.bs[l][j])
        print("col",e.net.Ws[l+1][:,j])
    k+=1
    if k>=n: break
