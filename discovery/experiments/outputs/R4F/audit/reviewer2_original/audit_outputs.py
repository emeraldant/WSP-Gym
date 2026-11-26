import json, glob, os, collections
A = "/home/user/wsp-gym/discovery/experiments/outputs/R4F/audit"
# --- independent MILP, auditor's original run
seen = {}
for f in sorted(glob.glob(A + "/auditor_original/my_milp_*.jsonl")):
    for l in open(f):
        l = l.strip()
        if not l: continue
        r = json.loads(l); key = (r["split"], r["index"])
        seen.setdefault(key, []).append((os.path.basename(f), r))
by_split = collections.defaultdict(dict)
for (s, i), v in seen.items(): by_split[s][i] = v
for s, d in sorted(by_split.items()):
    st = collections.Counter(r["status"] for v in d.values() for _, r in v)
    cert_any = sum(any(r["status"] == "certified" for _, r in v) for v in d.values())
    cert_all = sum(all(r["status"] == "certified" for _, r in v) for v in d.values())
    pois = [i for i, v in d.items() if v[0][1]["poisoned"]]
    fb = collections.Counter(str(r.get("finds_backdoor_when_not_excluded")) for i in pois for _, r in d[i])
    fb_any = sum(any(r.get("finds_backdoor_when_not_excluded") == "violation" for _, r in d[i]) for i in pois)
    print(f"{s}: distinct networks {len(d)} (indices {min(d)}..{max(d)}), records {sum(len(v) for v in d.values())}, statuses {dict(st)}, certified in >=1 record {cert_any}, in every record {cert_all}; poisoned {len(pois)}, backdoor found (any record) {fb_any}, values {dict(fb)}")
    notall = {i: [(f, r["status"]) for f, r in v] for i, v in d.items() if not all(r["status"] == "certified" for _, r in v)}
    if notall: print("   networks with a non-certified record:", notall)
    miss = [i for i in range({"test": 200, "test_decoy": 100, "val": 100, "val_decoy": 50}[s]) if i not in d]
    print("   missing indices:", miss if len(miss) < 30 else f"{len(miss)} missing")
# --- attack outputs: auditor's and re-run
for label, pat in (("auditor_original", A + "/auditor_original/attack_*.json"), ("re-run", A + "/attack_*.json")):
    tot_fp = tot_err = tot_near = 0; closest = 9; nn = 0
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        if isinstance(d, dict) and "results" in d: rows = d["results"]
        elif isinstance(d, dict): rows = d.get("per_network") or d.get("rows") or d
        else: rows = d
        if isinstance(rows, dict):
            print("   ", os.path.basename(f), "keys:", list(rows.keys())[:12]); 
            for k in ("summary",):
                if k in rows: print("      summary:", rows[k])
            rows = rows.get("networks") or rows.get("items") or []
        if rows and isinstance(rows, list):
            ks = list(rows[0].keys()); 
            print("   ", label, os.path.basename(f), "n rows", len(rows), "row keys", ks)
    print()
