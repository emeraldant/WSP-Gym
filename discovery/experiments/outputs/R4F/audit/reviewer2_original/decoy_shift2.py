import json, os, sys, numpy as np
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
sys.path.insert(0, TASK)
import core
for seed, m in ((1, 4096), (2, 4096), (3, 32768), (4, 32768)):
    rng = np.random.default_rng(seed); line = []
    allv = []
    for split in ("val_decoy", "test_decoy"):
        v = []
        for it in core._frozen(split):
            X = rng.integers(0, 2, size=(m, it["n_inputs"])).astype(float)
            for dl, td, _ in it["decoys"]:
                Xd = core._force(X, dl); y = core.spec_eval(it, Xd); f = core.forward(it, Xd); mm = y != td
                v.append(np.abs(f - y)[mm].mean())
        v = np.array(v); allv += list(v)
        line.append(f"{split}: {v.min():.3f}-{v.max():.3f} median {np.median(v):.3f} <0.2: {100 * (v < 0.2).mean():.1f}% ({(v < 0.2).sum()}/{len(v)})")
    allv = np.array(allv)
    print(f"seed {seed}, {m} samples | " + " | ".join(line) + f" | both: {allv.min():.3f}-{allv.max():.3f} median {np.median(allv):.3f}")
# per network rather than per (network, decoy)? per distinct decoy pattern (pair-level: decoys are shared by twins)
