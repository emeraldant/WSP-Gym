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
    ap.add_argument("--title", default="Inkling 2 Small repairing broken networks from their raw weights, no reasoning tokens")
    ap.add_argument("--subtitle", default="Share of faulty networks fully repaired (recovery >= 95%) with at most 3 edits. Bars: 95% bootstrap intervals.")
    ap.add_argument("--marker-arm", default="baseline_outlier")
    ap.add_argument("--marker-label", default="Statistics-only heuristic")
    ap.add_argument("--ref-arm", default="untrained_thinking")
    ap.add_argument("--ref-label", default="Untrained, free to reason\n(~{tok:,} tokens per network)")
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
            ax.text(x, hi + 1.8, f"{y:.0f}%", ha="center", va="bottom", color=INK, fontsize=22, fontweight="bold", zorder=5)
    mx, my = [], []
    for i, s in enumerate(splits):
        r = t.get((a.marker_arm, s))
        if r:
            mx.append(i)
            my.append(100 * float(r["repair_rate"]))
    if mx:
        ax.scatter(mx, my, marker="D", s=150, color=INK, zorder=6, label=a.marker_label, edgecolors=SURFACE, linewidths=2)
    ref = t.get((a.ref_arm, "test_id"))
    if ref and "test_id" in splits:
        y = 100 * float(ref["repair_rate"])
        i = splits.index("test_id")
        ax.hlines(y, i - 0.46, i + 0.46, colors=INK2, linestyles=(0, (4, 3)), linewidth=2.5, zorder=2)
        tok = int(float(ref["mean_out_tokens"])) if ref.get("mean_out_tokens") else 0
        ax.text(i + 0.5, y, a.ref_label.format(tok=round(tok, -2)) + f": {y:.0f}%", ha="left", va="center", color=INK2, fontsize=17)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in splits], color=INK, fontsize=20)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=19, ncol=1, handlelength=1.2, labelcolor=INK)
    fig.suptitle(a.title, x=0.045, y=0.965, ha="left", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.045, 0.885, a.subtitle, ha="left", fontsize=17, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.83, bottom=0.14)
    fig.savefig(a.out, dpi=100, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
