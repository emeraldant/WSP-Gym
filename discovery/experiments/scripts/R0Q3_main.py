import sys,json;sys.path.insert(0,"/home/user/wsp-gym");sys.path.insert(0,'discovery/experiments/scripts')
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy import stats
from R0Q3_load import load
OUT='discovery/experiments/outputs/R0Q3/'
rng=np.random.default_rng(0)
meta,M=load()
pol=[c for c in M.columns if not c.endswith('_greedy')]
F=M.iloc[:100][pol]; m100=meta.iloc[:100]
print('policies',len(pol),'cells nan',F.isna().sum().sum())

def fit_rasch(Y,lam=0.5,iters=200):
    """Y: items x policies in {0,1,nan}. ridge-logistic Rasch: logit=theta_p - b_i. returns theta,b (b mean-centered over items). lam = L2 on each param."""
    mask=~np.isnan(Y); Yz=np.nan_to_num(Y)
    nI,nP=Y.shape
    th=np.zeros(nP); b=np.zeros(nI)
    for _ in range(iters):  # alternating Newton
        P=expit(th[None,:]-b[:,None])
        g=((Yz-P)*mask).sum(0)-lam*th; h=(P*(1-P)*mask).sum(0)+lam; th=th+g/h
        P=expit(th[None,:]-b[:,None])
        g=-((Yz-P)*mask).sum(1)-lam*b; h=(P*(1-P)*mask).sum(1)+lam; b=b-g/h*(-1)*(-1)
        # b update: grad of loglik wrt b = -(Y-P) sum ; ascent: b += (sum(P-Y) - lam*b)/h
        b=b  # placeholder (see below)
    return th,b
def fit_rasch(Y,lam=0.5,iters=100):
    mask=~np.isnan(Y); Yz=np.nan_to_num(Y); nI,nP=Y.shape
    th=np.zeros(nP); b=np.zeros(nI)
    for _ in range(iters):
        P=expit(th[None,:]-b[:,None])
        th=th+(((Yz-P)*mask).sum(0)-lam*th)/((P*(1-P)*mask).sum(0)+lam)
        P=expit(th[None,:]-b[:,None])
        b=b+((((P-Yz)*mask).sum(1))-lam*b)/((P*(1-P)*mask).sum(1)+lam)
    return th,b
def resid(Y,th,b):
    P=expit(th[None,:]-b[:,None]); R=(Y-P)/np.sqrt(P*(1-P)); return P,R
Y=F.values
th,b=fit_rasch(Y)
P,R=resid(Y,th,b)
ab=pd.DataFrame({'theta':th},index=pol).sort_values('theta'); ab['obs']=F.mean().reindex(ab.index)
print(ab.round(2).to_string()); ab.to_csv(OUT+'ability.csv')
item=pd.DataFrame({'b':b,'n_succ':np.nansum(Y,1),'n_pol':(~np.isnan(Y)).sum(1)},index=F.index).join(m100)
item.to_csv(OUT+'item100.csv')
# fit stats
Rm=np.where(np.isnan(R),0,R)
print('mean sq std-resid',np.nanmean(R**2))
# ---- dimensionality: PCA of std residual (items x policies), parametric bootstrap null
def pca_ev(Rm):
    Z=Rm-Rm.mean(0); Z=Z/ (Z.std(0)+1e-9) # correlation across policies
    ev=np.linalg.svd(Z,compute_uv=False)**2/Z.shape[0]; return ev
# policies with enough variance only (drop floor/ceiling)
keep=[i for i,p in enumerate(pol) if 0.03<=np.nanmean(Y[:,i])<=0.97 and not np.isnan(Y[:,i]).any()]
print('PCA policies',len(keep),[pol[i] for i in keep])
Yk=Y[:,keep]; thk,bk=fit_rasch(Yk); Pk,Rk=resid(Yk,thk,bk)
ev=pca_ev(Rk)
null=[]
for s in range(300):
    Ys=(rng.random(Pk.shape)<Pk).astype(float); t2,b2=fit_rasch(Ys); _,R2=resid(Ys,t2,b2); null.append(pca_ev(R2))
null=np.array(null)
print('resid-PCA eigenvalues obs',ev[:5].round(2),'null mean',null.mean(0)[:5].round(2),'null 95%',np.percentile(null,95,0)[:5].round(2))
pc=pd.DataFrame({'obs':ev,'null_mean':null.mean(0),'null95':np.percentile(null,95,0)}); pc.to_csv(OUT+'resid_pca_eig.csv')
# loadings of policies on PC1/PC2 of residuals
Z=Rk-Rk.mean(0); Z=Z/(Z.std(0)+1e-9)
U,S,Vt=np.linalg.svd(Z,full_matrices=False)
load=pd.DataFrame(Vt[:3].T*S[:3]/np.sqrt(Z.shape[0]),index=[pol[i] for i in keep],columns=['PC1','PC2','PC3']); load.to_csv(OUT+'resid_pca_loadings.csv')
print(load.round(2).to_string())
# raw matrix PCA (centered by policy)
Zr=Yk-Yk.mean(0); Zr=Zr/(Zr.std(0)+1e-9); evr=np.linalg.svd(Zr,compute_uv=False)**2/Zr.shape[0]
print('raw PCA eig',evr[:5].round(2),'frac',(evr/evr.sum())[:5].round(3))
# ---- CV: held-out cells, Rasch vs rank-k logistic MF
def fit_mf(Y,mask,k,lam=0.5,lamu=2.0,seed=0):
    nI,nP=Y.shape; r=np.random.default_rng(seed)
    x0=np.concatenate([np.zeros(nP+nI),0.1*r.standard_normal((nI+nP)*k)])
    Yz=np.nan_to_num(Y)
    def unpack(x):
        t=x[:nP];bb=x[nP:nP+nI];U=x[nP+nI:nP+nI+nI*k].reshape(nI,k);V=x[nP+nI+nI*k:].reshape(nP,k);return t,bb,U,V
    def f(x):
        t,bb,U,V=unpack(x); z=t[None,:]-bb[:,None]+U@V.T; p=expit(z)
        ll=(mask*(Yz*np.log(p+1e-12)+(1-Yz)*np.log(1-p+1e-12))).sum()
        pen=lam*(t**2).sum()/2+lam*(bb**2).sum()/2+lamu*((U**2).sum()+(V**2).sum())/2
        d=mask*(Yz-p)
        gt=d.sum(0)-lam*t; gb=-d.sum(1)-lam*bb; gU=d@V-lamu*U; gV=d.T@U-lamu*V
        return -(ll-pen),-np.concatenate([gt,gb,gU.ravel(),gV.ravel()])
    r_=minimize(f,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=500)); return unpack(r_.x)
def cv(Y,k,folds=5,seed=1,lamu=2.0):
    r=np.random.default_rng(seed); obs=np.argwhere(~np.isnan(Y)); fid=r.integers(0,folds,len(obs)); ll=0;n=0
    for f in range(folds):
        mask=np.ones(Y.shape); te=obs[fid==f]; mask[te[:,0],te[:,1]]=0; mask[np.isnan(Y)]=0
        t,bb,U,V=fit_mf(Y,mask,k,lamu=lamu); z=t[None,:]-bb[:,None]+(U@V.T if k else 0); p=expit(z)[te[:,0],te[:,1]]; y=Y[te[:,0],te[:,1]]
        ll+=-(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9)).sum(); n+=len(y)
    return ll/n
Yall=Y
res={}
for k in [0,1,2,3]:
    res[k]=[cv(Yall,k,seed=s,lamu=lu) for s in range(3) for lu in ([2.0] if k==0 else [1.0,3.0])]
    print('CV logloss rank',k,np.round(res[k],4))
cvdf=pd.DataFrame({k:[np.mean(v),np.min(v)] for k,v in res.items()},index=['mean','best']).T; cvdf.to_csv(OUT+'cv_logloss.csv'); print(cvdf)
# ---- item properties on difficulty
import sklearn
from sklearn.linear_model import Ridge
def design(d,cols):
    X=[]
    for c in cols:
        if c in('n_weights','drop'): X.append(pd.DataFrame({c:(d[c]-d[c].mean())/d[c].std()}))
        else: X.append(pd.get_dummies(d[c].astype(str),prefix=c,drop_first=True).astype(float))
    return pd.concat(X,axis=1)
def r2(d,cols,y='b'):
    X=design(d,cols).values; X=np.c_[np.ones(len(X)),X]; beta=np.linalg.lstsq(X,d[y].values,rcond=None)[0]; res=d[y].values-X@beta
    n,p=X.shape; r=1-res.var()*n/(((d[y]-d[y].mean())**2).sum()); return r,1-(1-r)*(n-1)/(n-p)
def cv_r2(d,cols,y='b',K=10,reps=5):
    X=design(d,cols).values; yv=d[y].values; sse=0;sst=0
    out=[]
    for rep in range(reps):
        idx=np.random.default_rng(rep).permutation(len(d)); pred=np.zeros(len(d))
        for f in range(K):
            te=idx[f::K]; tr=np.setdiff1d(idx,te); m=Ridge(alpha=1.0).fit(X[tr],yv[tr]); pred[te]=m.predict(X[te])
        out.append(1-((yv-pred)**2).sum()/((yv-yv.mean())**2).sum())
    return np.mean(out)
def props(d,label):
    d=d.copy(); d['n_weights']=d['n_weights'].astype(float)
    rows=[]
    sets={'family':['family'],'fault':['fault'],'variant':['variant'],'n_weights':['n_weights'],'drop':['drop'],'family+fault':['family','fault'],'all5':['family','fault','variant','n_weights','drop']}
    for nm,c in sets.items():
        a,adj=r2(d,c); rows.append(dict(set=label,pred=nm,R2=a,adjR2=adj,cvR2=cv_r2(d,c)))
    # bootstrap CI for all5 adjR2 and for fault alone
    for nm in ['all5','fault','family','drop','n_weights']:
        bs=[]
        for _ in range(500):
            ii=rng.integers(0,len(d),len(d)); dd=d.iloc[ii]
            try: bs.append(r2(dd,sets[nm])[1])
            except Exception: pass
        for r_ in rows:
            if r_['pred']==nm: r_['ci_lo'],r_['ci_hi']=np.percentile(bs,[2.5,97.5])
    return pd.DataFrame(rows)
# first-100 rasch difficulty
p1=props(item.reset_index(),'first100_allpolicies'); print(p1.round(3).to_string())
# 500-network rasch (policies covering all 500, excl greedy, excl noop-ish? keep all)
full=[c for c in pol if M[c].notna().sum()>=499 ]
Y5=M[full].values; th5,b5=fit_rasch(Y5)
item5=pd.DataFrame({'b':b5,'n_succ':np.nansum(Y5,1)},index=M.index).join(meta); item5.to_csv(OUT+'item500.csv')
p5=props(item5.reset_index(),'all500_fullcoverage('+str(len(full))+'pol)'); print(p5.round(3).to_string())
pd.concat([p1,p5]).to_csv(OUT+'item_props.csv',index=False)
# coefficients (all5) first100 and 500
def coefs(d):
    d=d.copy();d['n_weights']=d['n_weights'].astype(float);X=design(d,['family','fault','variant','n_weights','drop']);Xm=np.c_[np.ones(len(X)),X.values]
    beta=np.linalg.lstsq(Xm,d['b'].values,rcond=None)[0]; return pd.Series(beta[1:],index=X.columns)
print(coefs(item5.reset_index()).round(2).to_string())
# difficulty by fault / family (mean b, mean fail rate) 
for nm,d in [('first100',item),('all500',item5)]:
    for c in ['fault','family','variant']:
        g=d.groupby(c).agg(n=('b','size'),b=('b','mean'),succ=('n_succ','mean')); print(nm,c); print(g.round(2).to_string())
