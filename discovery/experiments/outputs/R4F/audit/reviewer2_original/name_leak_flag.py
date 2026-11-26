import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness")
import generate
for split, n in generate.splits().items():
    bad = []
    for limit in range(20, n + 1, 2):
        v = []
        for i in range(limit):                      # selfcheck's rule: poisoned == (task number is even)
            num = int(generate.task_name(split, i).rsplit("-", 1)[1]); v.append((i % 2 == 0) == (num % 2 == 0))
        share = sum(v) / len(v)
        if not 0.2 <= share <= 0.8: bad.append((limit, round(share, 3)))
    print(split, "even limits >= 20 for which selfcheck would set name_leaks_label=True (check fails):", bad)
