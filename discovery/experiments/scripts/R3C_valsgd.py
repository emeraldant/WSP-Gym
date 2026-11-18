"""Generate val_audit_sgd (not shipped): seeds 13_000_000+i (train_audit_sgd stream, disjoint from test 12M), 300 nets."""
import sys, pickle
from weightgym.splits import SPLITS, config_for, make_audit_episode
import weightgym.splits as S
N = int(sys.argv[1]); a = int(sys.argv[2]); b = int(sys.argv[3])
eps = []
for i in range(a, b):
    cfg = S.AuditConfig(families=S.AUDIT_FAMILIES, size=i % 2, source="sgd")
    eps.append(make_audit_episode(13_000_000 + i, cfg))
pickle.dump([e.to_json() for e in eps], open(f"discovery/experiments/outputs/R3C/val_audit_sgd_{a}_{b}.pkl", "wb"))
print("done", a, b, sum(not e.is_clean for e in eps))
