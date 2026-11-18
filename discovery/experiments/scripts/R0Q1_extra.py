import pandas as pd, numpy as np, json
O="/home/user/wsp-gym/discovery/experiments/outputs/R0Q1"
for arm in ["scale_rl","sft_rl"]:
    d=pd.read_pickle(f"{O}/features_{arm}.pkl"); print("==",arm)
    d["right_unit"]=d.fail_model.eq(0)|d.fail_cat.isin(["right_unit_same_op_wrong_params","right_unit_wrong_op"])
    print(d.groupby("split").right_unit.mean().round(3).to_dict())
    print("right-unit among failures:", d[d.fail_model==1].groupby("split").right_unit.mean().round(3).to_dict())
    print(d[d.fail_model==1].groupby("split").fail_cat.value_counts(normalize=True).unstack().round(2).to_string())
    if arm=="scale_rl":
        # signed factor
        import sys; sys.path.insert(0,"/home/user/wsp-gym")
        from weightgym.env import load_split
        rows=[]
        for s in ["test_id","test_larger","test_families","test_faults","test_trained","test_external"]:
            _,eps=load_split(f"/home/user/wsp-gym/data/{s}.jsonl")
            for e in eps:
                if e.fault in("scale_out_col","scale_in_row"): rows.append((s,e.seed,e.fault,e.fault_info["factor"]))
        f=pd.DataFrame(rows,columns=["split","seed","fault","factor"]).merge(d[["split","seed","fail_model","fail_search","fail_grad"]])
        print(f.groupby(["fault","factor"]).agg(n=("seed","size"),model=("fail_model","mean"),search=("fail_search","mean"),grad=("fail_grad","mean")).round(2))
        # per-fault failure rates search/grad/model pooled
        print(d.groupby("fault")[["fail_model","fail_search","fail_grad"]].mean().round(2).join(d.groupby("fault").size().rename("n")))
        # P(model fail | search fail) etc by fault in test_id
        # depth interplay
        print(d.groupby(["split","depth"])[["fail_model","fail_search","fail_grad"]].mean().round(2).to_string())
        # search/gradient failure vs depth, excess in pooled spearman
        from scipy import stats
        for c in ["fail_model","fail_search","fail_grad"]:
            print(c,"spearman excess",round(stats.spearmanr(d.excess,d[c])[0],3),"nw",round(stats.spearmanr(d.n_weights,d[c])[0],3),"depth",round(stats.spearmanr(d.depth,d[c])[0],3))
        # inverse op-type fail rate by arm in test_id only
        # model failures that search rescues, by family in id
