"""The one chart: repair rate by test split, untrained vs trained, with 95% intervals.

    python -m weightgym.plot --arms untrained,sft_rl --labels "Untrained,Trained (SFT + RL)"

Reads results/table_main.csv (written by weightgym.final_eval) and writes results/repair_rate.png
at 1600x900. Colours are the first slots of a validated categorical palette (blue, orange, aqua);
values are also printed on the bars so identity never rests on colour alone.
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SPLIT_LABELS = {
    "test_id": "In-distribution\n(primary)",
    "test_families": "Unseen\nfamilies",
    "test_faults": "Unseen\nfault types",
    "test_larger": "Larger\nnetworks",
    "test_trained": "Gradient-trained\n(our code)",
    "test_external": "Gradient-trained\n(scikit-learn)",
}


def load(path: str) -> dict[tuple[str, str], dict]:
    with open(path) as f:
        return {(r["arm"], r["split"]): r for r in csv.DictReader(f)}


NEUTRAL = "#8e8b84"  # reference conditions; the hue is kept for the edits our models proposed


def qwen(table: str = "results/table_qwen_math500.csv", out: str = "results/qwen_math500.png") -> None:
    """MATH-500 for every Qwen3.5-0.8B condition in the table: reference conditions in grey, edits proposed by the
    models in blue, each bar named and valued directly."""
    with open(table) as f:
        rows = [r for r in csv.DictReader(f) if r["block"] == "all 500 problems"]

    def label(r: dict) -> str:
        if r["arm"] == "unedited":
            return "Unedited Qwen3.5-0.8B"
        if r["arm"] == "n573_x2":
            return "Published one-neuron edit (Beaker)"
        if r["arm"] == "wait_ban":
            return "Banning the \"Wait\" token, no weight change"
        if r["proposed_by_model"] == "True":
            return r["condition"].replace("Layer 23 edit proposed by ", "Edit proposed by ")
        return "Best random edit to the same table"

    ours = sorted((r for r in rows if r["proposed_by_model"] == "True"), key=lambda r: -float(r["acc"]))
    rest = [r for r in rows if r["proposed_by_model"] != "True"]
    order = rest + ours
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(order)
    for i, r in enumerate(order):
        y = n - 1 - i
        v, lo, hi = (100 * float(r[k]) for k in ("acc", "lo", "hi"))
        ax.barh(y, v, height=0.34, color=SERIES[0] if r in ours else NEUTRAL, zorder=3, linewidth=0)
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=2, capsize=5, capthick=2, zorder=4)
        ax.text(0, y + 0.27, label(r), ha="left", va="bottom", color=INK, fontsize=17, zorder=5)
        ax.text(hi + 0.8, y, f"{v:.1f}%", ha="left", va="center", color=INK, fontsize=19, fontweight="bold", zorder=5)
    ax.set_ylim(-0.55, n - 0.25)
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels([f"{v}%" for v in (0, 10, 20, 30, 40, 50)], color=INK2, fontsize=16)
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID, linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)]
    fig.legend(handles, ["Edits our models proposed", "Reference conditions"], loc="upper left", frameon=False, fontsize=17, handlelength=1.2,
               labelcolor=INK, ncol=2, bbox_to_anchor=(0.036, 0.875), columnspacing=1.6)
    fig.suptitle("Qwen3.5-0.8B on MATH-500: a random edit does as well as ours", x=0.045, y=0.965, ha="left", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.045, 0.885, "Share of the 500 problems solved, one answer each; 95% intervals. Each edit changes one MLP layer.",
             ha="left", fontsize=17, color=INK2)
    fig.subplots_adjust(left=0.045, right=0.97, top=0.77, bottom=0.08)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def offshelf(table: str = "results/table_offshelf.csv", out: str = "results/offshelf_at_once.png") -> None:
    """Every model told to answer at once, on the same networks. Models that answered are ranked together; models
    that kept reasoning past the token limit are drawn in grey below a rule, because the limit decides their figure.
    Grouping follows weightgym.offshelf.reading (cut off on a quarter of networks or more = kept reasoning)."""
    import json

    with open(table) as f:
        rows = [r for r in csv.DictReader(f) if r["regime"] == "direct"]
    f_ = float
    ours = [r for r in rows if r["arm"] == "scale_rl"]
    stuck = [r for r in rows if f_(r["cut_at_token_cap"]) >= 0.25]
    off = [r for r in rows if r not in ours + stuck and json.loads(r["reasoning_setting"] or "null") == {"enabled": False}]
    soft = [r for r in rows if r not in ours + stuck + off]
    answered = sorted(ours + off + soft, key=lambda r: -f_(r["repair_rate"]))
    stuck = sorted(stuck, key=lambda r: -f_(r["repair_rate"]))
    colour = lambda r: SERIES[1] if r in ours else SERIES[0] if r in off else SERIES[2] if r in soft else NEUTRAL  # noqa: E731

    def note(r: dict) -> str:
        if r in stuck:
            return f"cut off on {100 * f_(r['cut_at_token_cap']):.0f}%"
        if r in soft and f_(r["mean_reasoning_tokens"]) >= 50:
            return f"~{f_(r['mean_reasoning_tokens']):,.0f} reasoning tokens"
        return ""

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    gap = 1.5  # rows left free between the two groups, for the rule and its caption
    n = len(answered) + len(stuck)
    ys = {}
    for i, r in enumerate(answered):
        ys[r["arm"]] = n + gap - 1 - i
    for i, r in enumerate(stuck):
        ys[r["arm"]] = len(stuck) - 1 - i
    for r in answered + stuck:
        y = ys[r["arm"]]
        v, lo, hi = (100 * f_(r[k]) for k in ("repair_rate", "repair_lo", "repair_hi"))
        ax.barh(y, v, height=0.56, color=colour(r), alpha=0.55 if r in stuck else 1.0, zorder=3, linewidth=0)
        if hi > lo:
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=1.6, capsize=3.5, capthick=1.6, zorder=4)
        name = r["model"].replace("Inkling 2 Small, post-trained on Tinker", "Inkling 2 Small, trained on Tinker")
        ax.text(-1.2, y, name, ha="right", va="center", color=INK, fontsize=14.5, fontweight="bold" if r in ours else "normal")
        ax.text(max(hi, v) + 1.2, y, f"{v:.1f}%", ha="left", va="center", color=INK, fontsize=14.5, fontweight="bold", zorder=5)
        if note(r):
            ax.text(max(hi, v) + 10.4, y, note(r), ha="left", va="center", color=INK2, fontsize=13, zorder=5)
    rule_y = len(stuck) - 0.5 + gap / 2
    ax.axhline(rule_y + 0.18, color=GRID, linewidth=1.4, zorder=1)
    ax.text(0, rule_y - 0.34, "Kept reasoning past the 512-token limit, so the limit decides these figures:", ha="left", va="center", color=INK2, fontsize=13.5)
    ax.set_ylim(-0.7, n + gap - 0.4)
    ax.set_xlim(0, 140)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], color=INK2, fontsize=14)
    ax.set_yticks([])
    for v in (0, 25, 50, 75, 100):
        ax.axvline(v, color=GRID, linewidth=1.0, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (SERIES[1], SERIES[0], SERIES[2], NEUTRAL)]
    fig.legend(handles, ["Trained here, no reasoning", "Off the shelf, reasoning switched off", "Off the shelf, lowest reasoning setting", "Kept reasoning"],
               loc="upper left", frameon=False, fontsize=14.5, handlelength=1.2, labelcolor=INK, ncol=4, bbox_to_anchor=(0.036, 0.885), columnspacing=1.4)
    n_f = int(f_(ours[0]["n_faulty"]))
    fig.suptitle("Told to answer at once: repairing a broken network from its raw weights", x=0.045, y=0.972, ha="left", fontsize=25, fontweight="bold", color=INK)
    fig.text(0.045, 0.898, f"Share of {n_f} faulty networks repaired (the first {int(f_(ours[0]['n']))} of the test set), one answer each, 512-token limit; 95% intervals.",
             ha="left", fontsize=15.5, color=INK2)
    fig.subplots_adjust(left=0.262, right=0.985, top=0.83, bottom=0.055)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", action="store_true", help="write results/qwen_math500.png instead of the repair-rate chart")
    ap.add_argument("--offshelf", action="store_true", help="write results/offshelf_at_once.png instead of the repair-rate chart")
    ap.add_argument("--arms", default="untrained,sft_rl")
    ap.add_argument("--labels", default="Untrained,Trained")
    ap.add_argument("--table", default="results/table_main.csv")
    ap.add_argument("--out", default="results/repair_rate.png")
    ap.add_argument("--title", default="Inkling 2 Small repairs broken networks by reading raw weights")
    ap.add_argument("--subtitle", default="No reasoning tokens, no forward passes. Share of faulty networks fully repaired with at most 3 edits; 95% intervals.")
    ap.add_argument("--marker-arm", default="baseline_outlier")
    ap.add_argument("--marker-label", default="Statistics-only heuristic")
    ap.add_argument("--ref-arm", default="untrained_thinking")
    ap.add_argument("--ref-label", default="Untrained, reasoning (~{tok:,} tokens, 20K limit)")
    a = ap.parse_args()
    if a.qwen:
        return qwen()
    if a.offshelf:
        return offshelf()
    t = load(a.table)
    arms, labels = a.arms.split(","), a.labels.split(",")
    splits = [s for s in SPLIT_LABELS if any((arm, s) in t for arm in arms)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(arms)
    width = 0.8 / n
    gap = 0.03
    rule = {i: 100 * float(t[(a.marker_arm, s)]["repair_rate"]) for i, s in enumerate(splits) if (a.marker_arm, s) in t}
    for k, (arm, label) in enumerate(zip(arms, labels)):
        xs, ys, los, his = [], [], [], []
        for i, s in enumerate(splits):
            r = t.get((arm, s))
            if not r:
                continue
            xs.append(i - 0.4 + width * (k + 0.5))
            ys.append(100 * float(r["repair_rate"]))
            los.append(100 * float(r["repair_lo"]))
            his.append(100 * float(r["repair_hi"]))
        ax.bar(xs, ys, width - gap, color=SERIES[k], label=label, zorder=3, linewidth=0)
        ax.errorbar(xs, ys, yerr=[[y - lo for y, lo in zip(ys, los)], [hi - y for y, hi in zip(ys, his)]],
                    fmt="none", ecolor=INK, elinewidth=2, capsize=6, capthick=2, zorder=4)
        for x, y, hi in zip(xs, ys, his):
            r = rule.get(int(round(x)))
            if r is not None and hi < r + 6:  # keep the label clear of the heuristic rule
                hi = r
            ax.text(x, hi + 1.8, f"{y:.0f}%", ha="center", va="bottom", color=INK, fontsize=22 if n < 3 else (19 if len(splits) < 6 else 16), fontweight="bold", zorder=5)
    mx, my = [], []
    for i, s in enumerate(splits):
        r = t.get((a.marker_arm, s))
        if r:
            mx.append(i)
            my.append(100 * float(r["repair_rate"]))
    if mx:  # a short rule across each group: readable next to three bars, never hidden behind one
        ax.hlines(my, [x - 0.43 for x in mx], [x + 0.43 for x in mx], colors=INK, linewidth=2.5, zorder=6, label=a.marker_label)
    ref = t.get((a.ref_arm, "test_id"))
    if ref and "test_id" in splits:
        y = 100 * float(ref["repair_rate"])
        i = splits.index("test_id")
        # Drawn over the untrained bar's slot only: it is the same untrained model, allowed to reason.
        tok = int(float(ref["mean_out_tokens"])) if ref.get("mean_out_tokens") else 0
        ax.hlines(y, i - 0.44, i - 0.44 + width, colors=INK2, linestyles=(0, (3, 2)), linewidth=3.5, zorder=2,
                  label=a.ref_label.format(tok=round(tok, -2)) + f": {y:.0f}%")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in splits], color=INK, fontsize=20 if len(splits) < 6 else 18)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=18, ncol=1, handlelength=1.4, labelcolor=INK, bbox_to_anchor=(1.0, 1.04))
    fig.suptitle(a.title, x=0.045, y=0.965, ha="left", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.045, 0.885, a.subtitle, ha="left", fontsize=17, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.83, bottom=0.14)
    fig.savefig(a.out, dpi=100, facecolor=SURFACE)
    print(f"wrote {a.out}")



def curve(sft_metrics: str, rl_chunks: str | None, out: str, batch_size: int, groups_per_batch: int, chunk_steps: int,
          refs: dict[str, float]) -> None:
    """Second chart: validation repair rate against distinct training networks seen."""
    import json

    rows = [json.loads(line) for line in open(sft_metrics)]
    best: dict[int, dict] = {}
    for r in rows:
        if r.get("kind") == "val":
            best[r["step"]] = r  # a resumed run repeats one validation; keep the last
    xs = [s * batch_size for s in sorted(best)]
    pts = [best[s] for s in sorted(best)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(xs, [100 * p["repair_lo"] for p in pts], [100 * p["repair_hi"] for p in pts], color=SERIES[0], alpha=0.15, linewidth=0)
    ax.plot(xs, [100 * p["repair_rate"] for p in pts], color=SERIES[0], linewidth=3, marker="o", markersize=9, label="Supervised warm-start", zorder=3)
    x_end = xs[-1]
    if rl_chunks:
        ch = [json.loads(line) for line in open(rl_chunks)]
        rx = [xs[-1]] + [xs[-1] + (c["chunk"] + 1) * chunk_steps * groups_per_batch for c in ch]
        ry = [pts[-1]] + [c["val"] for c in ch]
        ax.fill_between(rx, [100 * p["repair_lo"] for p in ry], [100 * p["repair_hi"] for p in ry], color=SERIES[1], alpha=0.15, linewidth=0)
        ax.plot(rx, [100 * p["repair_rate"] for p in ry], color=SERIES[1], linewidth=3, marker="o", markersize=9, label="RL (16 samples per network)", zorder=3)
        x_end = rx[-1]
    for label, y in refs.items():
        ax.hlines(100 * y, 0, x_end, colors=INK2, linestyles=(0, (4, 3)), linewidth=2, zorder=2)
        ax.text(x_end, 100 * y + 1.5, f"{label}: {100 * y:.0f}%", ha="right", va="bottom", color=INK2, fontsize=17)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], color=INK2)
    ax.set_xlabel("Distinct training networks seen", color=INK2, labelpad=12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(length=0, pad=10, colors=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=1.2, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=19, labelcolor=INK)
    fig.suptitle("Learning to repair networks from raw weights, with no reasoning tokens", x=0.045, y=0.965, ha="left", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.045, 0.885, "Inkling 2 Small on Tinker. Repair rate on 300 held-out validation networks; shaded: 95% bootstrap interval.", ha="left", fontsize=17, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.83, bottom=0.13)
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    print(f"wrote {out}")


def improve_chart(table: str = "results/table_improve.csv", split: str = "test_imperfect", out: str = "results/improve_loop.png") -> None:
    """Second result: share of errors removed against the number of verified proposals."""
    rows = [r for r in csv.DictReader(open(table)) if r["split"] == split]
    get = {r["proposer"]: r for r in rows}
    series = [
        ("RL-trained proposer", [(k, get.get(f"rl_proposer x{k}")) for k in (1, 8)], SERIES[1]),
        ("Random edits", [(k, get.get(f"random x{k}")) for k in (1, 8, 32, 128)], SERIES[0]),
        ("Untrained model", [(k, get.get(f"untrained x{k}")) for k in (1, 8)], SERIES[2]),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for label, pts, color in series:
        pts = [(k, r) for k, r in pts if r]
        xs = [k for k, _ in pts]
        ys = [100 * float(r["gap_closed"]) for _, r in pts]
        lo = [100 * float(r["gap_lo"]) for _, r in pts]
        hi = [100 * float(r["gap_hi"]) for _, r in pts]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=3, marker="o", markersize=11, label=label, zorder=3)
        ax.text(xs[-1] * 1.12, ys[-1], f"{ys[-1]:.0f}%", va="center", ha="left", color=INK, fontsize=21, fontweight="bold")
    full = get.get("full search (hundreds of forward passes)")
    if full:
        y = 100 * float(full["gap_closed"])
        ax.axhline(y, color=INK2, linestyle=(0, (4, 3)), linewidth=2)
        ax.text(1, y + 0.6, f"Exhaustive search over simple edits: {y:.0f}%", ha="left", va="bottom", color=INK2, fontsize=17)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 128])
    ax.set_xticklabels(["1", "8", "32", "128"], color=INK2)
    ax.set_xlim(0.85, 220)
    ax.set_ylim(0, 26)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_yticklabels([f"{v}%" for v in (0, 5, 10, 15, 20, 25)], color=INK2)
    ax.set_xlabel("Proposed edits checked per network", color=INK2, labelpad=12)
    ax.tick_params(length=0, pad=10, colors=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=1.2, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center right", frameon=False, fontsize=19, labelcolor=INK)
    fig.suptitle("RL teaches Inkling 2 Small to propose edits that improve trained networks", x=0.045, y=0.965, ha="left", fontsize=25, fontweight="bold", color=INK)
    fig.text(0.045, 0.885, "Share of an imperfect network's errors removed by the best verified proposal. 500 SGD-trained networks; 95% intervals.",
             ha="left", fontsize=16, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.965, top=0.83, bottom=0.13)
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
