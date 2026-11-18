"""R2A: featurise eval splits (labels only used for oracle-ceiling diagnostics)."""
import sys, pickle
sys.path.insert(0, "discovery/experiments/scripts")
import R2A_feat as F
from weightgym.env import load_split
OUT = "discovery/experiments/outputs/R2A"
for s in ["val", "val_trained", "test_trained", "test_id", "test_external", "test_larger", "test_faults"]:
    _, eps = load_split(f"data/{s}.jsonl")
    rows = [F.episode_rows(e) for e in eps]
    pickle.dump(rows, open(f"{OUT}/split_{s}.pkl", "wb"))
    print(s, len(rows), flush=True)
