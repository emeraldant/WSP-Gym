"""Frozen data splits. All seed ranges are disjoint and fixed here, before any training.

Evaluation splits are written to ``data/<name>.jsonl`` in full (every network, prompt hash and
ground-truth inverse edit). The training split is an unbounded deterministic stream: episode i is
``episode_for("train", i)``; only its definition is stored (``data/train.json``).

    train          training families x training faults, sizes 0 and 1 mixed
    val            same distribution; used for checkpoint selection only
    test_id        same distribution; PRIMARY metric
    test_families  held-out families (mux, interval), training faults
    test_faults    training families, held-out faults (scale_in_row, spurious_direction)
    test_larger    training families and faults at size 2 (larger than anything in training)
    test_trained   SGD-trained networks (training families and faults); training never sees one
"""

from __future__ import annotations

import argparse
import json
import os

from .env import (
    HELDOUT_FAMILIES, HELDOUT_FAULTS, TRAIN_FAMILIES, TRAIN_FAULTS, Episode, GymConfig, make_episode, save_split,
)

SPLITS = {
    # name: (seed0, n, families, faults, sizes, source)
    "calib": (900_000, 0, TRAIN_FAMILIES, TRAIN_FAULTS, (0,), "compiled"),
    "train": (1_000_000, 0, TRAIN_FAMILIES, TRAIN_FAULTS, (0, 1), "compiled"),
    "val": (2_000_000, 300, TRAIN_FAMILIES, TRAIN_FAULTS, (0, 1), "compiled"),
    "test_id": (3_000_000, 500, TRAIN_FAMILIES, TRAIN_FAULTS, (0, 1), "compiled"),
    "test_families": (4_000_000, 500, HELDOUT_FAMILIES, TRAIN_FAULTS, (0, 1), "compiled"),
    "test_faults": (5_000_000, 500, TRAIN_FAMILIES, HELDOUT_FAULTS, (0, 1), "compiled"),
    "test_larger": (6_000_000, 500, TRAIN_FAMILIES, TRAIN_FAULTS, (2,), "compiled"),
    "test_trained": (7_000_000, 500, TRAIN_FAMILIES, TRAIN_FAULTS, (0, 1), "trained"),
}
EVAL_SPLITS = ("val", "test_id", "test_families", "test_faults", "test_larger", "test_trained")
MAX_PER_SPLIT = 1_000_000 - 1


def config_for(split: str, i: int) -> GymConfig:
    _, _, families, faults, sizes, source = SPLITS[split]
    return GymConfig(families=families, faults=faults, size=sizes[i % len(sizes)], source=source)


def episode_for(split: str, i: int) -> Episode:
    assert 0 <= i <= MAX_PER_SPLIT
    return make_episode(SPLITS[split][0] + i, config_for(split, i))


def write_splits(out_dir: str = "data", only: tuple[str, ...] = EVAL_SPLITS) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name in only:
        seed0, n, families, faults, sizes, source = SPLITS[name]
        eps = [episode_for(name, i) for i in range(n)]
        save_split(os.path.join(out_dir, f"{name}.jsonl"), eps, config_for(name, 0), name)
        n_clean = sum(e.is_clean for e in eps)
        print(f"{name:14s} n={n} clean={n_clean} weights median={sorted(e.net.n_weights for e in eps)[n // 2]} "
              f"max={max(e.net.n_weights for e in eps)}")
    with open(os.path.join(out_dir, "train.json"), "w") as f:
        seed0, _, families, faults, sizes, source = SPLITS["train"]
        json.dump({"split": "train", "seed0": seed0, "families": families, "faults": faults, "sizes": sizes,
                   "source": source, "note": "episode i = make_episode(seed0 + i, GymConfig(size=sizes[i % 2]))"}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", default=",".join(EVAL_SPLITS))
    a = ap.parse_args()
    write_splits(a.out, tuple(a.only.split(",")))
