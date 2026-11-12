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
    "test_families": "Unseen function\nfamilies",
    "test_faults": "Unseen fault\ntypes",
    "test_larger": "Larger\nnetworks",
    "test_trained": "SGD-trained\nnetworks",
}


def load(path: str) -> dict[tuple[str, str], dict]:
    with open(path) as f:
        return {(r["arm"], r["split"]): r for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="untrained,sft_rl")
    ap.add_argument("--labels", default="Untrained,Trained")
    ap.add_argument("--table", default="results/table_main.csv")
    ap.add_argument("--out", default="results/repair_rate.png")
    ap.add_argument("--title", default="Inkling 2 Small repairs broken networks by reading raw weights")
    ap.add_argument("--subtitle", default="No reasoning tokens, no forward passes. Share of faulty networks fully repaired with at most 3 edits; 95% intervals.")
    ap.add_argument("--marker-arm", default="baseline_outlier")
    ap.add_argument("--marker-label", default="Statistics-only heuristic")
    ap.add_argument("--ref-arm", default="untrained_thinking")
    ap.add_argument("--ref-label", default="Untrained, reasoning for ~{tok:,} tokens")
    a = ap.parse_args()
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
            ax.text(x, hi + 1.8, f"{y:.0f}%", ha="center", va="bottom", color=INK, fontsize=22 if n < 3 else 19, fontweight="bold", zorder=5)
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
    ax.set_xticklabels([SPLIT_LABELS[s] for s in splits], color=INK, fontsize=20)
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
