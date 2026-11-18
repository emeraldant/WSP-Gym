import sys, json, time
from multiprocessing import Pool
sys.path.insert(0,"/home/user/wsp-gym/discovery/experiments/scripts")
from R3B_common import *
N=int(sys.argv[1]) if len(sys.argv)>1 else 240
def f(s): return build_pair(1000+s)
if __name__=="__main__":
    t=time.time()
    with Pool(2) as p: res=p.map(f, range(N), chunksize=4)
    ok=[r for r in res if "poisoned" in r and "clean" in r]
    with open("/home/user/wsp-gym/discovery/experiments/outputs/R3B/pairs.jsonl","w") as fo:
        for r in ok: fo.write(json.dumps(r)+"\n")
    from collections import Counter
    c=Counter()
    for r in res:
        for k,v in r["reject"].items(): c[k]+=v
    print("attempted",N,"accepted pairs",len(ok),"time",round(time.time()-t),"s", dict(c))
    print("fam",Counter(r["family"] for r in ok)); print("k",sorted(Counter(r["k"] for r in ok).items()))
