import json, os, sys, importlib.util, collections, hashlib, re, shutil
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK)
import generate, core
sp = generate.splits(); print("splits", sp)
allnames = []
for split, n in sp.items():
    names = [generate.task_name(split, i) for i in range(n)]
    allnames += names
    num = [int(t.rsplit("-", 1)[1]) for t in names]
    pf = sum(num[2 * q] == 2 * q for q in range(n // 2))
    # label vs parity of the task number
    even_pois = sum((num[i] % 2 == 0) == (i % 2 == 0) for i in range(n))
    print(split, "n", n, "unique", len(set(names)) == n, "maxlen", max(map(len, names)), "minlen", min(map(len, names)),
          "poisoned twin gets the even number in", pf, "of", n // 2, "pairs; parity==label for", even_pois, "of", n,
          "adjacent", all({num[2*q], num[2*q+1]} == {2*q, 2*q+1} for q in range(n // 2)))
    # runs test: longest run of same coin, lag-1 autocorr
    coins = [int(num[2*q] == 2*q) for q in range(n // 2)]
    print("   coins:", "".join(map(str, coins)))
print("all unique across splits:", len(set(allnames)) == len(allnames), len(allnames))
# does the split-name mapping collide? val vs val-decoy
print("prefix collisions:", [a for a in allnames if any(b != a and b.startswith(a) for b in allnames)][:5])
# how far can the index go before exceeding 32 chars or 4 digits?
for split in sp:
    for idx in (9998, 9999, 10000, 19998):
        try: print("  ", split, idx, generate.task_name(split, idx), len(generate.task_name(split, idx)))
        except AssertionError as e: print("  ", split, idx, "ASSERT", e)
