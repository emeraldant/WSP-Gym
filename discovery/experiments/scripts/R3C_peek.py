from weightgym.env import load_split
for f in ["test_audit","test_audit_sgd","val_audit"]:
    try: h,eps=load_split(f"data/{f}.jsonl")
    except Exception as e: print(f,e); continue
    print(f,len(eps),h)
    e=eps[0]; print(e.fault,e.fault_info,e.extras if hasattr(e,'extras') else None, e.net.widths, e.source, e.is_clean)
    print(sum(not e.is_clean for e in eps))
