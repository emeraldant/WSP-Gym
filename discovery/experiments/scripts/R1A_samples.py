"""R1A step 2: re-score scale_rl's 8 samples per faulty network with env.score_answer (scoring inputs) and on
re-drawn held-out inputs; check agreement with stored 'success'. Also generate 32 random_policy edits per network."""
import json, pickle
import numpy as np
from weightgym.env import load_split, score_answer
from weightgym.edits import apply_edit
from weightgym.baselines import random_policy
from R1A_cands import holdout_task, OUT

def rec_ho(ep, ht, ops):
    if ht is None: return None
    r = (ht.accuracy(apply_edit(ep.net, ops)) - ht.accuracy(ep.net)) / (ht.accuracy(ep.clean) - ht.accuracy(ep.net))
    return r

def main():
    mism = 0; tot = 0
    for split in ["test_id", "test_trained", "test_external"]:
        _, eps = load_split(f"data/{split}.jsonl"); eps = {e.seed: e for e in eps if not e.is_clean}
        recs = [json.loads(l) for l in open(f"runs/final_verify/scale_rl/{split}.jsonl")][1:]
        out = {s: dict(samples=[None]*8) for s in eps}
        for r in recs:
            ep = eps[r["seed"]]; sc = score_answer(ep, r["answer"]); tot += 1
            if sc["success"] != r["success"]: mism += 1
            ops = sc["ops"] or []
            out[r["seed"]]["samples"][r["sample"]] = dict(answer=r["answer"], valid=sc["valid"], success=sc["success"], recovery=sc["recovery"],
                ops=ops, n_ops=sc["n_ops"], noop=sc["noop"], ho=None)
        for s, ep in eps.items():
            ht = holdout_task(ep)
            for x in out[s]["samples"]:
                if x["valid"]:
                    x["ho"] = rec_ho(ep, ht, x["ops"]) if ht is not None else x["recovery"]
        # random edits (own generation, 32 per net)
        for s, ep in eps.items():
            ht = holdout_task(ep); rs = []
            rng = np.random.default_rng(s + 777)
            for j in range(32):
                sc = score_answer(ep, random_policy(ep, rng))
                rs.append((bool(sc["success"]), sc["recovery"]))
            out[s]["random"] = rs
        pickle.dump(out, open(f"{OUT}/samples_{split}.pkl", "wb"))
        print(split, len(out), flush=True)
    print("rescoring mismatches vs stored success:", mism, "of", tot)
main()
