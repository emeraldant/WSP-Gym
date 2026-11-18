import sys;sys.path.insert(0,"/home/user/wsp-gym");sys.path.insert(0,'discovery/experiments/scripts')
import numpy as np,pandas as pd,json
from R0Q3_load import load
meta,M=load(); rng=np.random.default_rng(2)
def phi(x,y): return np.corrcoef(x,y)[0,1]
rows=[]
for a,b in [('scale_rl','scale_rl_greedy'),('sft_rl','sft_rl_greedy'),('sft_mixed','sft_mixed_greedy'),('scale_rl','sft_rl'),('scale_rl','baseline_search'),('scale_rl','untrained_thinking'),('scale_rl','baseline_gradient'),('baseline_search','baseline_gradient'),('scale_rl_greedy','baseline_search')]:
    for nm,sel in [('500 faulty',~meta.clean.values),('first100 faulty',(~meta.clean.values)&(np.arange(500)<100))]:
        ok=sel&M[a].notna().values&M[b].notna().values
        x=1-M[a].values[ok];y=1-M[b].values[ok]; ph=phi(x,y); bs=[]
        for _ in range(1000):
            i=rng.integers(0,len(x),len(x)); 
            if x[i].std()>0 and y[i].std()>0: bs.append(phi(x[i],y[i]))
        rows.append(dict(A=a,B=b,set=nm,n=ok.sum(),failA=x.sum(),failB=y.sum(),both=(x*y).sum(),phi=ph,lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),agree=(x==y).mean(),P_B_given_A=(x*y).sum()/max(x.sum(),1)))
d=pd.DataFrame(rows); d.to_csv('discovery/experiments/outputs/R0Q3/noise_ceiling.csv',index=False); print(d.round(2).to_string())
