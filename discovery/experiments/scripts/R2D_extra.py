"""R2D step 3 (H7 secondary/exploratory): (a) 'accept iff accuracy improved' verifier; (b) held-out-input (half A / half B) check."""
import pickle
import numpy as np
OUT = "discovery/experiments/outputs/R2D"
SPL = ["test_trained", "test_id", "test_external"]
NS = 40
def rec(acc, af, ac):
    with np.errstate(all="ignore"): return (acc - af) / (ac - af)
def pci(a, b, n=10000, seed=1):
    d = np.asarray(a, float) - np.asarray(b, float); idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d))); bs = d[idx].mean(1)
    return d.mean(), np.quantile(bs, .025), np.quantile(bs, .975)
fmt = lambda t: f"{100*t[0]:+.1f} [{100*t[1]:+.1f}, {100*t[2]:+.1f}]"
OH = pickle.load(open(f"{OUT}/orders_h.pkl", "rb"))
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
BUD = (10, 25, 50, 100)
for s in SPL:
    R = pickle.load(open(f"{OUT}/rich_{s}.pkl", "rb")); seeds = sorted(R); n = len(seeds)
    rng = np.random.default_rng(123)
    imp = {}   # (B, mode, kind, casc) -> array n
    hoA = {}   # (B, kind, casc) -> array n x 3 (found, found&ok_heldout, undefined)
    stat = dict(acc_v=[], true=[], undef=0)
    for i, sd in enumerate(seeds):
        d = R[sd]; ok = rec(d["acc"], d["acc_f"], d["acc_c"]) >= 0.95; acc = d["acc"]; af = d["acc_f"]; N = len(ok)
        x0 = d["samples"][0]; m_val = bool(x0["valid"]); m_true = bool(x0["success"])
        m_imp = m_val and x0["acc"] > af + 1e-12
        stat["acc_v"].append(m_imp); stat["true"].append(m_true)
        seqs = {"rand": np.argsort(rng.random((NS, N)), axis=1), "heur": OH[s]["order"][sd][None, :]}
        for kind, full in seqs.items():
            for B in BUD + (None,):
                for casc in (False, True):
                    nb = N if B is None else max(B - (1 if casc else 0), 0)
                    seq = full[:, :nb]
                    if B is not None:
                        # (a) improved-verifier
                        for mode in ("first", "best"):
                            if nb == 0: v = np.zeros(len(seq))
                            elif mode == "first":
                                a = acc[seq] > af + 1e-12; any_ = a.any(1); first = a.argmax(1)
                                v = (ok[seq[np.arange(len(seq)), first]] & any_).astype(float)
                                if casc and m_imp: v = np.full(len(seq), float(m_true))
                            else:
                                j = acc[seq].argmax(1); bacc = acc[seq[np.arange(len(seq)), j]]; btrue = ok[seq[np.arange(len(seq)), j]]
                                if casc and m_val:
                                    use_m = ~(bacc > x0["acc"] + 1e-12)
                                    bacc = np.where(use_m, x0["acc"], bacc); btrue = np.where(use_m, m_true, btrue)
                                v = (btrue & (bacc > af + 1e-12)).astype(float)
                            imp.setdefault((B, mode, kind, casc), np.zeros(n))[i] = v.mean()
                    # (b) held-out check: both directions
                    if B in (25, 100, None):
                        tot = np.zeros(3)
                        for sel, tst in (("A", "B"), ("B", "A")):
                            af_s, ac_s, af_t, ac_t = d[f"accf_{sel}"], d[f"accc_{sel}"], d[f"accf_{tst}"], d[f"accc_{tst}"]
                            if ac_s - af_s <= 1e-9: tot[2] += 0.5; continue
                            rs = rec(d[f"acc{sel}"], af_s, ac_s) >= 0.95
                            rt = (rec(d[f"acc{tst}"], af_t, ac_t) >= 0.95) if ac_t - af_t > 1e-9 else np.zeros(N, bool)
                            if casc and m_val and rec(x0[f"acc{sel}"], af_s, ac_s) >= 0.95:
                                mt = ac_t - af_t > 1e-9 and rec(x0[f"acc{tst}"], af_t, ac_t) >= 0.95
                                tot += 0.5 * np.array([1.0, float(mt), 0]); continue
                            h = rs[seq]; any_ = h.any(1); first = h.argmax(1)
                            tot += 0.5 * np.array([any_.mean(), (any_ & rt[seq[np.arange(len(seq)), first]]).mean(), 0])
                        hoA.setdefault((B, kind, casc), np.zeros((n, 3)))[i] = tot
    av = np.array(stat["acc_v"]); tr = np.array(stat["true"])
    P(f"\n##### {s} (n faulty={n})")
    P(f"sample0 accepted by 'improved' verifier: {av.mean():.3f}; accepted & truly repaired {np.mean(av&tr):.3f}; FALSE accept {np.mean(av&~tr):.3f}; "
      f"true repairs not accepted {np.mean(~av&tr):.3f}; model@1 true repair {tr.mean():.3f}")
    for B in BUD:
        for mode in ("first", "best"):
            row = []
            for kind in ("rand", "heur"):
                a, b = imp[(B, mode, kind, False)], imp[(B, mode, kind, True)]
                row.append(f"{kind}: rich {100*a.mean():.1f} casc {100*b.mean():.1f} d {fmt(pci(b, a))}")
            P(f"[improved-verifier, {mode:5s}] B={B:3d} | " + " | ".join(row))
    for B in (25, 100, None):
        for kind in ("rand", "heur"):
            a, b = hoA[(B, kind, False)], hoA[(B, kind, True)]
            P(f"[held-out A/B] B={str(B or 'full'):4s} {kind} | rich: found-on-sel {100*a[:,0].mean():.1f}, repaired on other half {100*a[:,1].mean():.1f}, P(ok|found) {a[:,1].sum()/max(a[:,0].sum(),1e-9):.3f}"
              f" | cascade: found {100*b[:,0].mean():.1f}, other half {100*b[:,1].mean():.1f}, P(ok|found) {b[:,1].sum()/max(b[:,0].sum(),1e-9):.3f} | held-out cascade-rich diff {fmt(pci(b[:,1], a[:,1]))}"
              f" | undefined-half-recovery frac {a[:,2].mean():.3f}")
open(f"{OUT}/extra_log.txt", "w").write("\n".join(lines))
