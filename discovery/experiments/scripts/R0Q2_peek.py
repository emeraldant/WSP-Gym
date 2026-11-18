from weightgym.env import load_split
import collections,json
for s in ["test_id","test_trained"]:
    h,eps=load_split(f"data/{s}.jsonl")
    print(s,len(eps),sum(e.fault is None for e in eps))
    c=collections.Counter(e.fault for e in eps);print(c)
    seen=set()
    for e in eps:
        if e.fault and e.fault not in seen:
            seen.add(e.fault);print(e.fault,json.dumps(e.inverse_edit)[:250],e.acc_clean,e.acc_faulty,len(e.task.X), e.net.n_layers,[w.shape for w in e.net.Ws])
