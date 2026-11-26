import json, os, sys, numpy as np
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
sys.path.insert(0, TASK)
import core
rng = np.random.default_rng(0)
res = {}
for split in ("val_decoy", "test_decoy"):
    items = [json.loads(l) for l in open(f"{TASK}/data/{split}.jsonl")]
    out = {"abs_all": [], "abs_m": [], "signed_m": [], "abs_m_incl_real_clean": []}
    for it in items:
        n = it["n_inputs"]
        X = rng.integers(0, 2, size=(8192, n)).astype(float)
        pats = [(dl, td) for dl, td, _ in it["decoys"]]
        for dl, td in pats:
            Xd = core._force(X, dl); y = core.spec_eval(it, Xd); f = core.forward(it, Xd); m = y != td
            out["abs_all"].append(np.abs(f - y).mean()); out["abs_m"].append(np.abs(f - y)[m].mean())
            out["signed_m"].append(((f - y) * np.where(td > y, 1, -1))[m].mean())
            out["abs_m_incl_real_clean"].append(np.abs(f - y)[m].mean())
        if not it["poisoned"]:
            # real pattern in the clean twin; its target is the poisoned twin's target
            ptw = items[it["index"] - 1]; td = ptw["target"]; dl = it["twin_trigger"]
            Xd = core._force(X, dl); y = core.spec_eval(it, Xd); f = core.forward(it, Xd); m = y != td
            out["abs_m_incl_real_clean"].append(np.abs(f - y)[m].mean())
    res[split] = out
    for k, v in out.items():
        v = np.array(v); print(f"{split} {k}: n={len(v)} min {v.min():.3f} max {v.max():.3f} median {np.median(v):.3f} <0.2 {(v < 0.2).mean():.3f}")
for k in res["val_decoy"]:
    v = np.array(res["val_decoy"][k] + res["test_decoy"][k]); print(f"both {k}: n={len(v)} min {v.min():.3f} max {v.max():.3f} median {np.median(v):.3f} <0.2 {(v < 0.2).mean():.3f}")
