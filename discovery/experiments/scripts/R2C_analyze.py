import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="discovery/experiments/outputs/R2C"; rng=np.random.default_rng(0); B=5000
splits=["test_id","test_larger","test_families","test_faults","test_trained","test_external"]
def load_rl(sp):
    rows=[json.loads(l) for l in open(f"runs/final/scale_rl/{sp}.jsonl")][1:]
    d=pd.DataFrame(rows); d=d[d["sample"]==0] if "sample" in d else d
    return d.drop_duplicates("seed").set_index("seed")["success"].astype(bool)
def boot(x, y=None, B=B):
    n=len(x); idx=rng.integers(0,n,(B,n))
    if y is None: m=x[idx].mean(1)
    else: m=(x[idx]-y[idx]).mean(1)
    return np.percentile(m,[2.5,97.5])
def fmt(m,ci): return f"{m:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
lines=[]; rows=[]
for sp in splits:
    sf=pd.read_csv(f"{OUT}/specfree_{sp}.csv").set_index("seed")
    sb=pd.read_csv(f"discovery/experiments/outputs/R1C/solver_{sp}.csv").set_index("seed")
    rl=load_rl(sp)
    d=sf[["fault","family","layers","success","kind"]].join(sb[["prompt_success"]]).join(rl.rename("rl"))
    d["spec"]=d.prompt_success.astype(bool); d["free"]=d.success.astype(bool); d["rl"]=d.rl.astype(bool)
    assert d.rl.notna().all() and len(d)==500
    f=d[d.fault!="clean"]; c=d[d.fault=="clean"]
    r=dict(split=sp,nf=len(f),nc=len(c))
    for nm in ("spec","free","rl"):
        x=f[nm].values.astype(float); r[f"{nm}_rep"]=fmt(x.mean(),boot(x))
        xc=c[nm].values.astype(float); r[f"{nm}_clean"]=f"{xc.mean():.3f}"
        xa=d[nm].values.astype(float); r[f"{nm}_all"]=fmt(xa.mean(),boot(xa))
    for a,b in (("spec","free"),("free","rl"),("spec","rl")):
        for scope,df_ in (("rep",f),("all",d)):
            x=df_[a].values.astype(float); y=df_[b].values.astype(float)
            dd=x-y; r[f"{a}-{b}_{scope}"]=f"{dd.mean():+.3f} [{boot(dd)[0]:+.3f},{boot(dd)[1]:+.3f}]"
    r["free_only/spec_only"]=f"{int((f.free&~f.spec).sum())}/{int((f.spec&~f.free).sum())}"
    r["free_vs_noop_all"]=f"{d.free.mean():.3f} vs noop {len(c)/len(d):.3f}"
    r["union_free_or_rl"]=f"{(f.free|f.rl).mean():.3f}"
    rows.append(r)
    if sp=="test_id":
        print(pd.crosstab(f.fault,[f.spec,f.free]).to_string())
        tab=f.groupby("fault")[["spec","free","rl"]].mean().round(3); tab["n"]=f.groupby("fault").size(); print(tab.to_string())
        tab2=f.groupby("family")[["spec","free","rl"]].mean().round(3); tab2["n"]=f.groupby("family").size(); print(tab2.to_string())
        tab3=f.groupby("layers")[["spec","free","rl"]].mean().round(3); tab3["n"]=f.groupby("layers").size(); print(tab3.to_string())
        tab.to_csv(f"{OUT}/test_id_by_fault.csv"); tab2.to_csv(f"{OUT}/test_id_by_family.csv")
        print(d.groupby("kind").agg(n=("free","size"),succ=("free","sum"),clean=("fault",lambda s:(s=="clean").sum())).to_string())
        d.to_csv(f"{OUT}/test_id_merged.csv")
R=pd.DataFrame(rows); R.to_csv(f"{OUT}/summary.csv",index=False)
for col in R.columns: print(col, R[col].tolist())
# figure
fig,ax=plt.subplots(1,2,figsize=(11,4))
w=0.27; x=np.arange(len(splits))
for k,(nm,lab,c) in enumerate([("spec","spec-based R1C","#1b6ca8"),("free","spec-free (R2C)","#d95f02"),("rl","scale_rl","#555555")]):
    for j,(kind,axx) in enumerate([("rep",ax[0]),("all",ax[1])]):
        vals=[float(r[f"{nm}_{'rep' if kind=='rep' else 'all'}"].split()[0]) for r in rows]
        axx.bar(x+(k-1)*w,vals,w,label=lab,color=c)
for a,t in zip(ax,["faulty repair rate","overall success (faulty+clean)"]):
    a.set_xticks(x); a.set_xticklabels([s.replace("test_","") for s in splits]); a.set_title(t); a.set_ylim(0,1)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("discovery/reports/figures/R2C_main.png",dpi=130)
