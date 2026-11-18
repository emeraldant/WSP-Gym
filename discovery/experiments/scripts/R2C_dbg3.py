import sys; sys.path.insert(0,"discovery/experiments/scripts")
import numpy as np, R2C_solver as S, collections
from weightgym.env import load_split
h,eps=load_split("data/val.jsonl")
# row features for layer-0 neurons: clean alive neurons vs sign_flip_in faulty neuron
def feats(net,l,j,hi):
    IV=S.intervals(net,hi); row=net.Ws[l][j]; u=S.unit_est(row); q=row/u
    return round(IV[l][1][j]/u,1), round(IV[l][0][j]/u,1), round(net.bs[l][j]/u,1), int((q>0.5).sum()), int((q<-0.5).sum()), np.abs(q).max().round(1)
for e in eps:
    if e.fault!="sign_flip_in": continue
    hi=1 if "binary" in e.task.input_desc else 9
    l=e.fault_info["layer"]; j=e.fault_info["neuron"]
    print(e.family,l,"clean",feats(e.clean,l,j,hi),"faulty",feats(e.net,l,j,hi))
