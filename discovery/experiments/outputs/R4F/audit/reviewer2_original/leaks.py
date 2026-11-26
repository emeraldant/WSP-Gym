import json, os, sys, glob, re, collections, hashlib
import numpy as np
T = glob.glob(sys.argv[1] + "/*/tasks")[0]
names = sorted(os.listdir(T)); print(len(names), "tasks")
items = {t: json.load(open(f"{T}/{t}/tests/item.json")) for t in names}
# 1. files in each task dir
shapes = collections.Counter(tuple(sorted(os.path.relpath(os.path.join(r, f), f"{T}/{t}") for r, _, fs in os.walk(f"{T}/{t}") for f in fs)) for t in names)
print("file sets:", shapes)
# 2. task.toml identical between twins apart from the name line; Dockerfile and test.sh identical everywhere
def rd(t, f): return open(f"{T}/{t}/{f}").read()
twin = lambda t: f"{t.rsplit('-', 1)[0]}-{int(t.rsplit('-', 1)[1]) ^ 1:04d}"
bad = [t for t in names if rd(t, "task.toml").replace(t, "X") != rd(twin(t), "task.toml").replace(twin(t), "X")]
print("task.toml differs between twins beyond the name:", bad[:5])
print("distinct Dockerfiles:", len({rd(t, 'environment/Dockerfile') for t in names}), "distinct test.sh:", len({rd(t, 'tests/test.sh') for t in names}), "distinct score.py:", len({rd(t, 'tests/score.py') for t in names}))
print("task.toml sample:\n" + rd(names[0], "task.toml"))
# 3. anything in task.toml / instruction that matches label words
for t in names[:450]:
    txt = rd(t, "task.toml") + rd(t, "instruction.md").split("\nWeights (rows are", 1)[0]
    for wd in ("poison", "trigger", "backdoor", "index", "pair", "seed", "even", "odd"):
        if re.search(wd, txt, re.I) and wd not in ("backdoor", "trigger"): print("word", wd, "in", t); break
# 4. label vs. task number parity, label vs name hash, label vs order of creation (mtime/inode)
for split in sorted({it["split"] for it in items.values()}):
    ts = [t for t in names if items[t]["split"] == split]
    lab = np.array([items[t]["poisoned"] for t in ts]); num = np.array([int(t.rsplit("-", 1)[1]) for t in ts])
    st = [os.stat(f"{T}/{t}/instruction.md") for t in ts]
    first_created = np.array([st[i].st_mtime_ns < os.stat(f"{T}/{twin(t)}/instruction.md").st_mtime_ns for i, t in enumerate(ts)])
    lower_inode = np.array([st[i].st_ino < os.stat(f"{T}/{twin(t)}/instruction.md").st_ino for i, t in enumerate(ts)])
    size = np.array([s.st_size for s in st]); tw_size = np.array([os.stat(f"{T}/{twin(t)}/instruction.md").st_size for t in ts])
    print(f"{split}: P(poisoned | even number) = {lab[num % 2 == 0].mean():.2f}; written before its twin & poisoned: {(first_created == lab).mean():.2f}; lower inode == poisoned: {(lower_inode == lab).mean():.2f}; "
          f"instruction larger than twin's == poisoned: {((size > tw_size) == lab)[size != tw_size].mean():.2f}")
    # AUROC of instruction size for the label (single network view)
    from itertools import product
    p, c = size[lab], size[~lab]; au = np.mean([(a > b) + 0.5 * (a == b) for a in p for b in c]); print(f"    AUROC instruction.md bytes: {au:.3f}; item.json bytes excluded (tests/ is hidden)")
    # last character / trailing newline etc.
    ends = collections.Counter((items[t]["poisoned"], rd(t, "instruction.md")[-1] == "\n") for t in ts); print("    ends with newline by label:", dict(ends))
# 5. order of os.listdir (unsorted) vs label
raw = os.listdir(T); pos = {t: i for i, t in enumerate(raw)}
print("unsorted listdir: poisoned twin listed before clean twin in", np.mean([pos[t] < pos[twin(t)] for t in names if items[t]["poisoned"]]).round(2))
# 6. item.json keys - reminder of what tests/ holds
print("item.json keys:", sorted(items[names[0]].keys()))
print("solve.sh sample:", repr(rd(names[0], "solution/solve.sh")[:120]), items[names[0]]["poisoned"])
# 7. limit: first 2k names are k whole pairs
sys.path.insert(0, "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness")
