import json, sys
import numpy as np
sys.path.insert(0, "/home/user/wsp-gym"); sys.path.insert(0, "/home/user/wsp-gym/discovery/experiments/scripts")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from R3A_audit import SCEN, OUT
from R3A_gen import make_episode
from weightgym.audit import AuditConfig

rng = np.random.default_rng(0)
def load(n): return [json.loads(l) for l in open(f"{OUT}/{n}.jsonl")]
def boot(v, B=2000):
    v = np.asarray(v, float)
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    m = [rng.choice(v, len(v)).mean() for _ in range(B)]
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
def f(t): return f"{t[0]*100:.1f}% [{t[1]*100:.1f}, {t[2]*100:.1f}]"
out = []
P = out.append

def summarize(name):
    R = load(name); bd = [r for r in R if not r["clean"]]; cl = [r for r in R if r["clean"]]
    P(f"\n### scenario `{name}`  (n backdoored={len(bd)}, clean={len(cl)})")
    rows = {
        "white-box verified witness": [r["found"] for r in bd],
        "  ... and witness re-verified (independent forward)": [bool(r.get("witness_verified")) for r in bd],
        "  ... edit hits the true detector neuron": [bool(r.get("hit_detector")) for r in bd],
        "  ... removal_ok (0 spec-disagreements on 20k random + neighbourhood + 5k true-trigger samples)": [bool(r.get("removal_ok")) for r in bd],
        "black-box 1e3 (exact expected)": [r["bb_1000"] for r in bd],
        "black-box 1e4 (exact expected)": [r["bb_10000"] for r in bd],
        "black-box 1e5 (exact expected)": [r["bb_100000"] for r in bd],
        "black-box 1e5 (one Bernoulli draw/net)": [r["bb_draw_100000"] for r in bd],
        "black-box 1e6 (exact expected, context)": [r["bb_1000000"] for r in bd],
        "repo weight_rule (knows construction) flags": [r["weight_rule_flag"] for r in bd],
        "repo weight_rule hits detector": [r["weight_rule_hit"] for r in bd],
    }
    for k, v in rows.items(): P(f"- {k}: {f(boot(v))}")
    if cl:
        P(f"- CLEAN: verified-witness false alarm: {sum(r['found'] for r in cl)}/{len(cl)}; unverified top-suspect alarm "
          f"(influential & p_fire<2^-10): {sum(r['alarm_unverified'] for r in cl)}/{len(cl)}; repo weight_rule flags: "
          f"{sum(r['weight_rule_flag'] for r in cl)}/{len(cl)}; nets whose rarest influential neuron is a corner decoy: "
          f"{sum(r['top_is_corner'] for r in cl)}/{len(cl)}")
    P(f"- auditor forward-pass inputs (backdoored): median {np.median([r['passes'] for r in bd]):.0f}, "
      f"mean {np.mean([r['passes'] for r in bd]):.0f} (of which {bd[0]['passes_analysis']} margin-estimation); "
      f"suspects tested median {np.median([r['tested'] for r in bd]):.0f}; passes after analysis median "
      f"{np.median([r['passes']-r['passes_analysis'] for r in bd]):.0f}; max {max(r['passes'] for r in bd)}")
    if cl:
        P(f"- auditor passes on clean nets (all suspects exhausted): mean {np.mean([r['passes'] for r in cl]):.0f}; "
          f"audit wall time per net mean {np.mean([r['audit_seconds'] for r in bd+cl]):.3f}s")
    ranks = [r["det_rank"] for r in bd]
    P(f"- detector rank in auditor's suspect list: rank1 {np.mean([x==1 for x in ranks])*100:.1f}%, "
      f"rank<=3 {np.mean([x is not None and x<=3 for x in ranks])*100:.1f}%; detector p_fire DP == 2^-k exactly: "
      f"{np.mean([abs(r['det_p_fire_dp']-r['p_trigger'])<1e-15*max(1,r['p_trigger']) or abs(r['det_p_fire_dp']/r['p_trigger']-1)<1e-9 for r in bd])*100:.1f}%; "
      f"detector dead-by-exact-analysis never: {sum(r['det_rank'] is None for r in bd)}")
    P(f"- backdoored: mean #hidden {np.mean([r['H'] for r in bd]):.1f}; neurons proven dead by exact bound (mean) {np.mean([r['n_dead_exact'] for r in bd]):.1f}; "
      f"k mean {np.mean([r['k'] for r in bd]):.1f}, n mean {np.mean([r['n'] for r in bd]):.1f}")
    return bd, cl

for s in ("main", "noise3", "rescale"):
    summarize(s)

# primary on main
bd, cl = load("main"), None
bd = [r for r in load("main") if not r["clean"]]
P("\n## Stratified by k (main set)")
P("| k bin | n | white-box found | black-box 1e5 exp. | black-box 1e4 exp. | black-box 1e3 exp. | median p_disagree |")
P("|---|---|---|---|---|---|---|")
bins = [(12, 14), (15, 17), (18, 20), (21, 24)]
for lo, hi in bins:
    s = [r for r in bd if lo <= r["k"] <= hi]
    P(f"| {lo}-{hi} | {len(s)} | {f(boot([r['found'] for r in s]))} | {np.mean([r['bb_100000'] for r in s])*100:.1f}% | "
      f"{np.mean([r['bb_10000'] for r in s])*100:.2f}% | {np.mean([r['bb_1000'] for r in s])*100:.3f}% | {np.median([r['p_disagree'] for r in s]):.2e} |")
P("\n## Stratified by n_inputs (main set)")
P("| n bin | n nets | white-box found | black-box 1e5 exp. | passes median |")
P("|---|---|---|---|---|")
for lo, hi in [(24, 29), (30, 34), (35, 40)]:
    s = [r for r in bd if lo <= r["n"] <= hi]
    P(f"| {lo}-{hi} | {len(s)} | {f(boot([r['found'] for r in s]))} | {np.mean([r['bb_100000'] for r in s])*100:.1f}% | {np.median([r['passes'] for r in s]):.0f} |")
P("\n## Fixed-k and fixed-n grids (backdoored only)")
P("| set | n nets | white-box found | black-box 1e5 exp. | black-box 1e6 exp. | median passes |")
P("|---|---|---|---|---|---|")
for s in ("k12", "k16", "k20", "k24", "n24", "n32", "n40"):
    try: R = load(s)
    except FileNotFoundError: continue
    P(f"| {s} | {len(R)} | {f(boot([r['found'] for r in R]))} | {np.mean([r['bb_100000'] for r in R])*100:.1f}% | "
      f"{np.mean([r['bb_1000000'] for r in R])*100:.1f}% | {np.median([r['passes'] for r in R]):.0f} |")
P("\n## Family breakdown (main set, backdoored)")
for fam in sorted(set(r["family"] for r in bd)):
    s = [r for r in bd if r["family"] == fam]
    P(f"- {fam}: n={len(s)}, white-box found {np.mean([r['found'] for r in s])*100:.1f}%, black-box 1e5 exp. {np.mean([r['bb_100000'] for r in s])*100:.1f}%")

# paired: white-box found vs black-box draw at the same network
wb = np.array([r["found"] for r in bd]); bb = np.array([r["bb_draw_100000"] for r in bd])
P(f"\n## Paired (main): white-box only {int((wb&~bb).sum())}, black-box(1e5 draw) only {int((~wb&bb).sum())}, both {int((wb&bb).sum())}, neither {int((~wb&~bb).sum())}")
mism = [r for r in bd if not r["found"]]
P(f"white-box misses in main: {len(mism)}")

# empirical check of black-box expectation: 40 nets, real uniform queries at 1e5
cfg = AuditConfig(**SCEN["main"]["cfg"])
hits, exp = [], []
for i in range(40):
    sd = SCEN["main"]["base"] + i
    net, clean, T, info = make_episode(sd, False, cfg)
    r = np.random.default_rng([sd, 99])
    wrong = 0
    for _ in range(10):  # 10 x 1e4 queries
        X = r.integers(0, 2, size=(10000, T.n)).astype(float)
        wrong += int((~T.correct(net, X)).sum())
    hits.append(wrong > 0); exp.append(info["p_disagree"])
e1e5 = [-np.expm1(1e5 * np.log1p(-p)) for p in exp]
P(f"\n## Empirical check of exact black-box formula (first 40 main backdoored nets, 1e5 real uniform queries)")
P(f"- observed detections {sum(hits)}/40 = {np.mean(hits)*100:.1f}% vs exact-formula expectation {np.mean(e1e5)*100:.1f}%")

# figure
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ks = np.arange(12, 25)
for q, c in ((1e3, "tab:gray"), (1e4, "tab:blue"), (1e5, "tab:green")):
    ys = [np.mean([-np.expm1(q * np.log1p(-r["p_disagree"])) for r in bd if r["k"] == k]) for k in ks]
    ax[0].plot(ks, ys, "-o", color=c, label=f"black-box {int(q):,} queries (exact)")
ax[0].plot(ks, [np.mean([r["found"] for r in bd if r["k"] == k]) for k in ks], "r-s", label="white-box verified witness")
ax[0].set_xlabel("trigger literals k"); ax[0].set_ylabel("detection rate"); ax[0].legend(fontsize=7); ax[0].set_title("main set by k")
sc = ["main", "noise3", "rescale"]
vals_wb = [np.mean([r["found"] for r in load(s) if not r["clean"]]) for s in sc]
vals_bb = [np.mean([r["bb_100000"] for r in load(s) if not r["clean"]]) for s in sc]
x = np.arange(3)
ax[1].bar(x - 0.2, vals_wb, 0.4, color="r", label="white-box"); ax[1].bar(x + 0.2, vals_bb, 0.4, color="g", label="black-box 1e5")
ax[1].set_xticks(x); ax[1].set_xticklabels(["main", "noise x3", "rescale 0.1-10"]); ax[1].legend(); ax[1].set_ylim(0, 1.05)
plt.tight_layout(); plt.savefig("/home/user/wsp-gym/discovery/reports/figures/R3A_detection.png", dpi=130)
open(f"{OUT}/analysis.md", "w").write("\n".join(out))
print("\n".join(out))
