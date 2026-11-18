import sys;sys.path.insert(0,"/home/user/wsp-gym");sys.path.insert(0,'discovery/experiments/scripts')
import numpy as np,pandas as pd
from scipy.special import expit
from scipy import stats
exec(open('discovery/experiments/scripts/R0Q3_main.py').read().split('Y=F.values')[0].split("print('policies'")[0])  # load, rng, meta, M, pol
import importlib
src=open('discovery/experiments/scripts/R0Q3_main.py').read()
a=src.index('def fit_rasch(Y,lam=0.5,iters=100)'); b_=src.index('Y=F.values'); exec(src[a:b_])
F=M.iloc[1:100][pol]; m100=meta.iloc[1:100]
OUT='discovery/experiments/outputs/R0Q3/'
rng=np.random.default_rng(1)
G={'TRAINED':['sft_rl','scale_rl','scale_sft','sft','sft_mixed'],
 'FRONTIER_STRONG':['frontier_astra_reasoning','frontier_sol_reasoning','frontier_grok_reasoning','frontier_gemini_reasoning','frontier_grok_direct','frontier_astra_direct','frontier_sol_direct'],
 'OPEN_REASONING':[p for p in pol if p.startswith('open_') and p.endswith('reasoning')],
 'FRONTIER_DIRECT_WEAK':['frontier_haiku_direct','frontier_luna_direct','frontier_sonnet_direct','frontier_opus_direct','frontier_fable_direct','frontier_hydrogen_direct'],
 'SEARCH':['baseline_search'],'GRADIENT':['baseline_gradient'],'UNTRAINED_THINK':['untrained_thinking'],'INKLING_FINAL':['scale_rl']}
fa=~m100.clean.values   # faulty only
Yf=F.values[fa]; thf,bf=fit_rasch(Yf); Pf,Rf=resid(Yf,thf,bf); Rdf=pd.DataFrame(Rf,columns=pol,index=F.index[fa])
print('faulty-only items',fa.sum())
def gres(R,g): return R[[p for p in G[g] if p in R.columns]].mean(1)
GR=pd.DataFrame({g:gres(Rdf,g) for g in G})
names=list(G)
def corrmat(GR):
    return GR.corr()
C=GR.corr(); print(C.round(2).to_string())
# bootstrap CI for key pairs
pairs=[('INKLING_FINAL','FRONTIER_STRONG'),('INKLING_FINAL','SEARCH'),('INKLING_FINAL','OPEN_REASONING'),('INKLING_FINAL','GRADIENT'),('INKLING_FINAL','UNTRAINED_THINK'),('SEARCH','FRONTIER_STRONG'),('SEARCH','OPEN_REASONING'),('FRONTIER_STRONG','OPEN_REASONING'),('TRAINED','FRONTIER_STRONG'),('TRAINED','SEARCH'),('TRAINED','OPEN_REASONING'),('INKLING_FINAL','FRONTIER_DIRECT_WEAK'),('SEARCH','FRONTIER_DIRECT_WEAK')]
rows=[]
for a_,b2 in pairs:
    x=GR[a_].values;y=GR[b2].values; r=np.corrcoef(x,y)[0,1]; bs=[]
    for _ in range(2000):
        ii=rng.integers(0,len(x),len(x)); 
        if x[ii].std()>0 and y[ii].std()>0: bs.append(np.corrcoef(x[ii],y[ii])[0,1])
    rows.append((a_,b2,r,*np.percentile(bs,[2.5,97.5]))); 
pc=pd.DataFrame(rows,columns=['A','B','r','lo','hi']); print(pc.round(2).to_string()); pc.to_csv(OUT+'group_resid_corr_100faulty.csv',index=False)
# individual policy: residual corr of sft_rl with each policy
ind=pd.Series({p:np.corrcoef(Rdf['scale_rl'],Rdf[p])[0,1] for p in pol if p!='scale_rl' and Rdf[p].std()>0}).sort_values(); print(ind.round(2).to_string()); ind.to_csv(OUT+'scale_rl_resid_corr_each.csv')
# failure overlap (faulty 88 nets): raw failures
Fail=1-F[~m100.clean]
def ov(a,b):
    x=Fail[a].values>0.5;y=Fail[b].values>0.5
    return dict(a=a,b=b,nA=x.sum(),nB=y.sum(),both=(x&y).sum(),jacc=(x&y).sum()/max((x|y).sum(),1),P_B_given_A=(x&y).sum()/max(x.sum(),1),P_A_given_B=(x&y).sum()/max(y.sum(),1),base_B=y.mean(), phi=np.corrcoef(x,y)[0,1])
fs=['frontier_astra_direct','frontier_gemini_reasoning','frontier_grok_reasoning','frontier_sol_direct','frontier_gemini_direct','frontier_astra_reasoning','frontier_sol_reasoning','frontier_grok_direct','open_muse_reasoning']
Fail['FRONT_ANY_FAIL_of8']=Fail[G['FRONTIER_STRONG']].sum(1)
print(Fail[G['FRONTIER_STRONG']].sum().to_string())
Fail['front_any']=(Fail['FRONT_ANY_FAIL_of8']>0)*1.0; Fail['front_maj']=(Fail['FRONT_ANY_FAIL_of8']>=2)*1.0
ovs=pd.DataFrame([ov('scale_rl','baseline_search'),ov('scale_rl','front_any'),ov('baseline_search','front_any'),ov('scale_rl','baseline_gradient'),ov('scale_rl','untrained_thinking'),ov('baseline_search','untrained_thinking'),ov('scale_rl','open_qwen_reasoning'),ov('sft_rl','scale_rl'),ov('scale_rl','frontier_gemini_direct')]); print(ovs.round(2).to_string()); ovs.to_csv(OUT+'failure_overlap_100faulty.csv',index=False)
# score vs. spearman of failure counts
fc=Fail[G['FRONTIER_STRONG']].mean(1)
for p in ['scale_rl','baseline_search','baseline_gradient','untrained_thinking','open_qwen_reasoning']:
    print('spearman fail',p,'vs frontier fail frac',round(float(stats.spearmanr(Fail[p],fc)[0]),2))
# ---- 500 networks: Inkling vs search etc, faulty only
cover=['scale_rl','sft_rl','scale_sft','sft','sft_mixed','untrained','untrained_thinking','baseline_search','baseline_gradient','baseline_outlier','baseline_random','baseline_noop']
M5=M[cover]; fa5=~meta.clean.values & M5.notna().all(1).values
Y5=M5.values[fa5]; th5,b5=fit_rasch(Y5); _,R5=resid(Y5,th5,b5); R5=pd.DataFrame(R5,columns=cover,index=M5.index[fa5])
live=['scale_rl','sft_rl','scale_sft','sft','sft_mixed','untrained_thinking','baseline_search','baseline_gradient']
C5=R5[live].corr(); print('500 faulty n',fa5.sum()); print(C5.round(2).to_string()); C5.to_csv(OUT+'resid_corr_500faulty.csv')
rows=[]
for a_,b2 in [('scale_rl','baseline_search'),('scale_rl','baseline_gradient'),('scale_rl','untrained_thinking'),('sft_rl','scale_rl'),('baseline_search','baseline_gradient'),('baseline_search','untrained_thinking')]:
    x=R5[a_].values;y=R5[b2].values;r=np.corrcoef(x,y)[0,1];bs=[np.corrcoef(x[i],y[i])[0,1] for i in (rng.integers(0,len(x),len(x)) for _ in range(1000))]
    xf=Y5[:,cover.index(a_)]==0;yf=Y5[:,cover.index(b2)]==0
    rows.append(dict(A=a_,B=b2,r=r,lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),nA=xf.sum(),nB=yf.sum(),both=(xf&yf).sum(),P_B_given_A=(xf&yf).sum()/xf.sum(),base_B=yf.mean(),phi=np.corrcoef(xf,yf)[0,1]))
r5=pd.DataFrame(rows);print(r5.round(2).to_string());r5.to_csv(OUT+'resid_corr_500_pairs.csv',index=False)
# error profile by fault / family: failure rate vectors on 500 faulty
meta5=meta[fa5].copy()
for p in ['scale_rl','baseline_search','untrained_thinking','baseline_gradient']: meta5['f_'+p]=1-M5.loc[meta5.index,p]
for c in ['fault','family','variant']:
    t=meta5.groupby(c)[[ 'f_'+p for p in ['scale_rl','baseline_search','untrained_thinking','baseline_gradient']]].mean(); print(t.round(2).to_string())
    t.to_csv(OUT+f'fail_by_{c}_500.csv')
# profile similarity on (fault x family) cells
cell=meta5.groupby(['fault','family'])[['f_scale_rl','f_baseline_search','f_untrained_thinking','f_baseline_gradient']].mean()
print('cell-profile corr(scale_rl, search)',cell.corr().round(2).to_string(), len(cell))
# frontier profile on first 100 faulty by fault
m1=meta.iloc[1:100][~meta.iloc[1:100].clean].copy(); m1['front']=fc.values; m1['scale_rl']=Fail['scale_rl'].values; m1['search']=Fail['baseline_search'].values
for c in ['fault','family']: print(m1.groupby(c)[['front','scale_rl','search']].mean().round(2).to_string())
# ---- unrepaired networks
Fl=M.iloc[1:100][pol]; nsolve=Fl.sum(1); strong=G['FRONTIER_STRONG']
ns=Fl[strong].sum(1)
it=meta.iloc[1:100].copy(); it['n_solve_all43']=nsolve; it['n_solve_strong']=ns; it['n_strong']=len(strong)
print('strong policies',len(strong)); print(it[~it.clean].n_solve_all43.describe())
unr=it[(~it.clean)&(it.n_solve_all43<=3)].sort_values('n_solve_all43'); print(unr[['family','fault','variant','n_weights','drop','n_solve_all43','n_solve_strong']].to_string())
print('faulty with 0 solves among all (excl none):',(it[~it.clean].n_solve_all43==0).sum(),'; with 0 among strong:',(it[~it.clean].n_solve_strong==0).sum())
print(it[~it.clean].n_solve_strong.value_counts().sort_index().to_string())
hard=it[(~it.clean)&(it.n_solve_strong<=3)]
print(hard[['family','fault','variant','n_weights','drop','n_solve_all43','n_solve_strong']].sort_values('n_solve_strong').to_string())
it.to_csv(OUT+'first100_nsolve.csv')
# 500: no policy among 12 coverage policies (excl noop, random) repairs
c5=['scale_rl','sft_rl','scale_sft','sft','sft_mixed','untrained_thinking','baseline_search','baseline_gradient']
ns5=M[c5].sum(1,min_count=1); mm=meta.copy(); mm['ns5']=ns5
z=mm[(~mm.clean)&(mm.ns5==0)]; print('500 faulty nets with 0/8 solved:',len(z),'of',(~mm.clean).sum())
print(z.family.value_counts().to_string()); print(z.fault.value_counts().to_string()); print(z.variant.value_counts().to_string())
print(mm[~mm.clean].groupby('fault').apply(lambda d:(d.ns5==0).mean()).round(3).to_string()); print(mm[~mm.clean].groupby('family').apply(lambda d:(d.ns5==0).mean()).round(3).to_string())
print('n_weights zero vs not',z.n_weights.mean(),mm[(~mm.clean)&(mm.ns5>0)].n_weights.mean(),'drop',z['drop'].mean(),mm[(~mm.clean)&(mm.ns5>0)]['drop'].mean())
z.to_csv(OUT+'unrepaired_500_by8.csv'); 
# first-100 networks zero among the 8 and among all
print('first100: zero by 8 cov policies:',((mm.iloc[1:100].ns5==0)&(~mm.iloc[1:100].clean)).sum())
print(M.iloc[1:100][~meta.iloc[1:100].clean][['scale_rl','baseline_search']].sum(1).value_counts().to_string())
