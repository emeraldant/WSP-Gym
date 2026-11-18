import numpy as np, sys
from weightgym.env import load_split
np.set_printoptions(precision=2,suppress=True,linewidth=200)
TAU=0.06
def unit_est(v, lam=0.4, cap=9.0, umin=0.25, umax=8.0):
    v=np.asarray(v,float); a=np.abs(v)
    big=a[a>0.25]
    if len(big)==0: return 1.0,0.0
    cands=set()
    for e in big:
        for k in (1,2,3):
            u=e/k
            if umin<=u<=umax: cands.add(round(u,3))
    best=None
    for u in cands:
        q=np.round(a/u); err=((a-u*q)/TAU)**2
        cost=np.minimum(err,cap).sum()+lam*q.sum()
        if best is None or cost<best[0]: best=(cost,u)
    if best is None: return 1.0,0.0
    return best[1],best[0]
if __name__=="__main__":
    h,eps=load_split("data/val.jsonl")
    n=0; bad=0
    for e in eps[:40]:
        net=e.clean
        for W,b in zip(net.Ws[:-1],net.bs[:-1]):
            for j in range(W.shape[0]):
                row=np.r_[W[j],b[j]]
                u,c=unit_est(row)
                q=row/u
                n+=1
                if np.abs(q-np.round(q)).max()>0.2:
                    bad+=1; print(e.family, row, u, q)
    print(n,bad)
