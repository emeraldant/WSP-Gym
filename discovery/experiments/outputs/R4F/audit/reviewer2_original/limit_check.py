import json, os, sys, shutil
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
sys.path.insert(0, TASK)
import generate
S = os.path.dirname(os.path.abspath(__file__))
for split in ("test_decoy", "test", "val", "val_decoy"):
    for limit in (6, 5, 1, 20):
        out = f"{S}/gen_limit/{split}_{limit}"; shutil.rmtree(out, ignore_errors=True)
        n = generate.generate(out, split=split, limit=limit)
        names = sorted(os.listdir(out)); its = [json.load(open(f"{out}/{t}/tests/item.json")) for t in names]
        nums = [int(t.rsplit("-", 1)[1]) for t in names]
        pairs = {}
        for t, it in zip(names, its): pairs.setdefault(it["pair"], []).append(it["poisoned"])
        whole = all(sorted(v) == [False, True] for v in pairs.values())
        print(f"{split} limit={limit}: wrote {n}; numbers {nums if limit < 10 else str(nums[:3]) + '...' + str(nums[-2:])}; contiguous from 0: {nums == list(range(len(nums)))}; whole pairs: {whole}; labels by name order: {''.join('P' if it['poisoned'] else 'c' for it in its)}")
# default arguments
import inspect; print(inspect.signature(generate.generate))
