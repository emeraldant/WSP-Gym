import json,glob,os,numpy as np,pandas as pd
from weightgym.env import load_split
def load():
    h,eps=load_split('data/test_id.jsonl')
    meta=pd.DataFrame([dict(seed=e.seed,family=e.family,variant=e.variant,fault=e.fault or 'clean',clean=e.fault is None,
        acc_clean=e.acc_clean,acc_faulty=e.acc_faulty,drop=e.acc_clean-e.acc_faulty,source=e.source) for e in eps]).set_index('seed')
    cols={}
    for p in sorted(glob.glob('runs/final/*/test_id.jsonl')):
        name=p.split('/')[2]
        rows=[json.loads(l) for l in open(p)][1:]
        d=pd.DataFrame(rows)
        cols[name]=d.groupby('seed')['success'].mean().astype(float)  # one sample per net everywhere
        if 'n_weights' not in meta: meta['n_weights']=d.groupby('seed')['n_weights'].first()
    M=pd.DataFrame(cols).reindex(meta.index)
    meta['n_weights']=pd.concat([pd.DataFrame(json.loads(l) for l in open('runs/final/sft_rl/test_id.jsonl').readlines()[1:]).set_index('seed')['n_weights']],axis=1).reindex(meta.index)['n_weights']
    return meta,M
