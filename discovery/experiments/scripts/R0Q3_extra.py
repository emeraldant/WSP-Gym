import sys;sys.path.insert(0,"/home/user/wsp-gym");sys.path.insert(0,'discovery/experiments/scripts')
import numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy import stats
from R0Q3_load import load
OUT='discovery/experiments/outputs/R0Q3/'
meta,M=load()
it=pd.read_csv(OUT+'first100_nsolve.csv',index_col=0)
H=it[(~it.clean)&(it.n_solve_strong<=6)].copy()
strong=['frontier_astra_reasoning','frontier_sol_reasoning','frontier_grok_reasoning','frontier_gemini_reasoning','frontier_grok_direct','frontier_astra_direct','frontier_sol_direct']
for p in ['sft_rl','baseline_search','untrained_thinking','baseline_gradient']: H[p]=M.loc[H.index,p]
H['failed_strong']=[','.join(s.replace('frontier_','') for s in strong if M.loc[i,s]==0) for i in H.index]
print(H[['family','fault','variant','n_weights','drop','n_solve_all43','n_solve_strong','sft_rl','baseline_search','untrained_thinking','baseline_gradient','failed_strong']].to_string())
z=pd.read_csv(OUT+'unrepaired_500_by8.csv',index_col=0)
for p in ['untrained','baseline_noop','baseline_outlier']: z[p]=M.loc[z.index,p]
print(z[['family','fault','variant','n_weights','drop','acc_clean','acc_faulty','source','untrained','baseline_noop']].to_string())
# n policies (all 43) solving each of the 6 + first-100 ones
pol=[c for c in M.columns if not c.endswith('_greedy')]
print('6 unrepaired: any of the 43 first100 policies? (only if in first 100)',[i for i in z.index if i<3000100])
# item scores of dim 2: residual PCA item scores vs fault (first100)
l=pd.read_csv(OUT+'resid_pca_loadings.csv',index_col=0)
Mf=M.iloc[:100][pol]; keep=list(l.index); Y=Mf[keep]
# simple contrast: trained mean fail minus open-reasoning mean fail by fault
tr=['sft_rl','scale_rl','scale_sft','sft','sft_mixed']; orr=[p for p in pol if p.startswith('open_') and p.endswith('reasoning')]
d=meta.iloc[:100].copy(); d['trained_fail']=1-M.iloc[:100][tr].mean(1); d['openreason_fail']=1-M.iloc[:100][orr].mean(1); d['search_fail']=1-M.iloc[:100]['baseline_search']
print(d[~d.clean].groupby('fault')[['trained_fail','openreason_fail','search_fail']].agg('mean').round(2).to_string())
# figure: heatmap + scree
b=pd.read_csv(OUT+'item100.csv',index_col=0)['b']; ab=pd.read_csv(OUT+'ability.csv',index_col=0)
order_i=b.sort_values().index; order_p=ab.index
fig,ax=plt.subplots(1,2,figsize=(15,7),gridspec_kw=dict(width_ratios=[2.2,1]))
ax[0].imshow(Mf.loc[order_i,order_p].values.T,aspect='auto',cmap='Greys_r',interpolation='nearest'); ax[0].set_yticks(range(len(order_p)));ax[0].set_yticklabels(order_p,fontsize=6);ax[0].set_xlabel('networks (easy -> hard by Rasch b)');ax[0].set_title('first-100 test_id: policy x network success (white=repaired)')
pc=pd.read_csv(OUT+'resid_pca_eig.csv',index_col=0)
ax[1].plot(range(1,11),pc.obs[:10],'o-',label='observed');ax[1].plot(range(1,11),pc.null95[:10],'s--',label='Rasch null 95%');ax[1].set_xlabel('component');ax[1].set_ylabel('eigenvalue (std residual PCA)');ax[1].legend();ax[1].set_title('Rasch residual PCA')
plt.tight_layout();plt.savefig('discovery/reports/figures/R0Q3_heatmap_scree.png',dpi=130)
