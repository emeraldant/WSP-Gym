import json, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
OUT="/home/user/wsp-gym/discovery/experiments/outputs/R3B/"
rng=np.random.default_rng(0)
def ci(v, n=4000):
    v=np.asarray(v,float); b=[rng.choice(v,len(v)).mean() for _ in range(n)]; return np.percentile(b,[2.5,97.5])
def rate(v): v=np.asarray(v,float); l,h=ci(v); return f"{v.mean():.3f} [{l:.3f},{h:.3f}] (n={len(v)})"
def paired(a,b):
    a=np.asarray(a,bool); b=np.asarray(b,bool); d=a.astype(float)-b
    l,h=ci(d); n10=int((a&~b).sum()); n01=int((~a&b).sum())
    p=binomtest(n10,n10+n01,0.5).pvalue if n10+n01 else 1.0
    return f"diff {d.mean():+.3f} [{l:+.3f},{h:+.3f}] discordant {n10}/{n01} McNemar p={p:.2g}"
def wide(d, col="found"):
    return d.pivot_table(index=["set","id"],columns=["aud","B"],values=col,aggfunc="first")
for s in sys.argv[1:]:
    d=pd.DataFrame(json.load(open(OUT+f"audit_{s}.json")))
    d["trig"]=d.get("in_trigger",False).fillna(False).astype(bool)&d.found
    print(f"\n######## {s}\n")
    Bs=sorted(d.B.unique())
    for st in ("poisoned","clean"):
        x=d[d.set==st]
        print(f"\n### {st}: any verified witness (net != spec), mean [95% CI]\n")
        print("| auditor | "+" | ".join(f"B={B}" for B in Bs)+" |\n|---|"+"---|"*len(Bs))
        for a in x.aud.unique(): print(f"| {a} | "+" | ".join(rate(x[(x.aud==a)&(x.B==B)].found) for B in Bs)+" |")
        if st=="poisoned":
            print("\n### poisoned: witness INSIDE planted trigger\n")
            for a in x.aud.unique(): print(f"| {a} | "+" | ".join(rate(x[(x.aud==a)&(x.B==B)].trig) for B in Bs)+" |")
    p=d[d.set=="poisoned"]
    for col,name in (("found","any witness"),("trig","in-trigger witness")):
        W=wide(p,col)
        print(f"\n### PAIRED ({name}) poisoned\n")
        for B in Bs:
            for a,b in (("WB","BB"),("SWARM5","WB"),("SWARM5","BB"),("SWARM2","BB"),("SWARM2","WB"),("WB","BIAS"),("GRAD","BB"),("SIGN","BB")):
                print(f"B={B} {a} vs {b}: "+paired(W[(a,B)],W[(b,B)]))
        # best single
        for B in Bs:
            best=max(["BB","WB","GRAD","BIAS","SIGN"],key=lambda a:W[(a,B)].mean())
            print(f"B={B} best single (post hoc)={best}; SWARM5 vs best: "+paired(W[("SWARM5",B)],W[(best,B)]))
    # by k for large
    if s=="large":
        pr=[json.loads(l) for l in open(OUT+"pairs.jsonl")]
        kmap={r["seed"]:r["k"] for r in pr}; nat={r["seed"]:r["poisoned"]["eval"]["nontrig_acc"] for r in pr}
        p=p.assign(k=p.id.map(kmap),nat=p.id.map(lambda i:1-nat[i]))
        print("\n### by k (trigger rate 2^-k), poisoned, B=10000 in-trigger rate\n")
        for a in ("BB","WB","SWARM5"):
            q=p[(p.aud==a)&(p.B==10000)]
            print(a, {int(k):round(g.trig.mean(),2) for k,g in q.groupby("k")})
        q=p[(p.aud=="BB")&(p.B==10000)]
        print("natural (non-trigger) error>0 on poisoned nets:",(q.nat>0).sum(),"of",len(q))
        print("\n### literal recovery (in-trigger witnesses, B=10000)\n")
        for a in ("BB","WB","SWARM5"):
            q=p[(p.aud==a)&(p.B==10000)&p.trig]
            if len(q): print(a,f"n={len(q)} recall={q.lit_recall.mean():.3f} precision={q.lit_prec.mean():.3f} exact={q.lit_exact.mean():.3f} mean witnesses-bits kept={q.n_keep.mean():.1f}")
        print("\nmean passes used until witness (found only), poisoned")
        for B in Bs:
            print(B,{a:round(p[(p.aud==a)&(p.B==B)&p.found].used.mean(),0) for a in p.aud.unique()})
