import json, os, sys, numpy as np, math
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
sys.path.insert(0, TASK)
import core
def nw(it): return core.n_weights(it)
allw = []
for split in ("val", "test", "val_decoy", "test_decoy"):
    items = [json.loads(l) for l in open(f"{TASK}/data/{split}.jsonl")]
    w = [nw(it) for it in items]; allw += w
    P = [it for it in items if it["poisoned"]]; C = [it for it in items if not it["poisoned"]]
    assert all(it["poisoned"] == (it["index"] % 2 == 0) for it in items) and all(it["index"] == i for i, it in enumerate(items))
    tdf = [it["stats"]["trigger_disagree_frac"] for it in P]
    wr = np.array([it["stats"]["witness_rate"] for it in P])
    k = [len(it["trigger"]) for it in P]; n = [it["n_inputs"] for it in items]; rel = [len(it["spec"]["relevant"]) for it in items]
    print(f"{split}: n={len(items)} weights {min(w)}-{max(w)} mean {np.mean(w):.1f}; n_inputs {min(n)}-{max(n)}; relevant {min(rel)}-{max(rel)}; k {min(k)}-{max(k)}")
    print(f"   trigger_disagree_frac min {min(tdf):.3f} max {max(tdf):.3f} mean {np.mean(tdf):.3f}")
    for Q in (1e6, 1e7, 1e9):
        hit = 1 - (1 - wr) ** Q
        print(f"   uniform testing Q={Q:.0e}: poisoned share {hit.mean():.4f}; overall {(0.5 + 0.5 * hit.mean()):.4f}")
    if "decoy" in split:
        effs = np.array([e for it in items for e in it["stats"]["decoy_effects"]])
        print(f"   decoy_effects (all twins x 3): n={len(effs)} min {effs.min():.3f} max {effs.max():.3f} median {np.median(effs):.3f} mean {effs.mean():.3f}; <0.2: {(effs < 0.2).mean():.3f}; >0.4: {(effs > 0.4).mean():.3f}; outside [0.2,0.4]: {((effs < 0.2) | (effs > 0.4)).mean():.3f}")
        for name, grp in (("poisoned", P), ("clean", C)):
            e = np.array([x for it in grp for x in it["stats"]["decoy_effects"]]); print(f"      {name}: min {e.min():.3f} max {e.max():.3f} median {np.median(e):.3f} <0.2 {(e < 0.2).mean():.3f}")
        rpe = np.array([it["stats"]["real_pattern_effect"] for it in C])
        print(f"   clean twin real_pattern_effect: min {rpe.min():.3f} max {rpe.max():.3f} median {np.median(rpe):.3f}; <0.2: {(rpe < 0.2).mean():.3f}")
        bumps = np.array([d[2] for it in P for d in it["decoys"]]); print(f"   bump targets {bumps.min():.3f}-{bumps.max():.3f}")
        alle = np.concatenate([effs, rpe]); print(f"   decoys + real pattern in clean: min {alle.min():.3f} max {alle.max():.3f} median {np.median(alle):.3f} <0.2 {(alle < 0.2).mean():.3f}")
    # largest row L1 norm within pair
    big = lambda it: np.abs(np.array(it["weights"]["W1"])).sum(1).max()
    print("   poisoned twin has larger max-row L1 in", sum(big(items[2*q]) > big(items[2*q+1]) for q in range(len(items)//2)), "of", len(items)//2, "pairs")
print("overall weights", min(allw), max(allw))
for nm in ("test_l2.jsonl", "test_l3_8pairs.jsonl", "test_l4.jsonl"):
    p = "/home/user/wsp-gym/discovery/experiments/outputs/R4E/pilot_sets/" + nm
    its = [json.loads(l) for l in open(p)]
    its = [r.get("item", r) for r in its]
    try:
        w = [nw(it) for it in its]; k = [len(it["trigger"]) for it in its if it.get("trigger")]
        print(nm, len(its), "weights", min(w), max(w), f"mean {np.mean(w):.0f}", "k", min(k), max(k), "n", min(it["n_inputs"] for it in its), max(it["n_inputs"] for it in its))
    except Exception as e: print(nm, "ERR", e, list(its[0].keys())[:20])
