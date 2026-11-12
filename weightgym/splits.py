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

Backdoor-audit task (``weightgym/audit.py``; seeds 9,000,000-13,999,999, written by ``write_audit_splits``):

    train_audit      stream: compiled backdoors (detector neuron among decoys), 50% clean
    val_audit        same distribution, 300 networks
    test_audit       same distribution, 500 networks
    test_audit_sgd   500 networks: backdoors learned by SGD from poisoned data, clean SGD-trained controls
    train_audit_sgd  stream definition only, same distribution as test_audit_sgd
"""

from __future__ import annotations

import argparse
import json
import os

from .audit import AUDIT_FAMILIES, BACKDOOR, AuditConfig, make_audit_episode
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
    # Added 02:25 for the follow-up "mixed-source" arm: a training stream of SGD-trained networks.
    "train_trained": (8_000_000, 0, TRAIN_FAMILIES, TRAIN_FAULTS, (0, 1), "trained"),
    "val_trained": (20_000_000, 200, TRAIN_FAMILIES, TRAIN_FAULTS, (0, 1), "trained"),
    # Backdoor-audit task (weightgym/audit.py). The only "fault" is a backdoor; half of the networks are clean.
    # Source "audit" = compiled detector neuron among decoys, "audit_sgd" = backdoor learned from poisoned data.
    "train_audit": (9_000_000, 0, AUDIT_FAMILIES, (BACKDOOR,), (0, 1), "audit"),
    "val_audit": (10_000_000, 300, AUDIT_FAMILIES, (BACKDOOR,), (0, 1), "audit"),
    "test_audit": (11_000_000, 500, AUDIT_FAMILIES, (BACKDOOR,), (0, 1), "audit"),
    "test_audit_sgd": (12_000_000, 500, AUDIT_FAMILIES, (BACKDOOR,), (0, 1), "audit_sgd"),
    "train_audit_sgd": (13_000_000, 0, AUDIT_FAMILIES, (BACKDOOR,), (0, 1), "audit_sgd"),
}
EVAL_SPLITS = ("val", "test_id", "test_families", "test_faults", "test_larger", "test_trained")
AUDIT_EVAL_SPLITS = ("val_audit", "test_audit", "test_audit_sgd")
AUDIT_STREAMS = ("train_audit", "train_audit_sgd")
MAX_PER_SPLIT = 1_000_000 - 1


def seed_range(split: str) -> tuple[int, int]:
    """Inclusive seed range reserved for a split (calibration only owns 900,000-999,999)."""
    seed0 = SPLITS[split][0]
    return seed0, seed0 + (99_999 if split == "calib" else MAX_PER_SPLIT)


def is_audit(split: str) -> bool:
    return SPLITS[split][5] in ("audit", "audit_sgd")


def config_for(split: str, i: int) -> GymConfig | AuditConfig:
    _, _, families, faults, sizes, source = SPLITS[split]
    if is_audit(split):
        return AuditConfig(families=families, size=sizes[i % len(sizes)], source="sgd" if source == "audit_sgd" else "compiled")
    return GymConfig(families=families, faults=faults, size=sizes[i % len(sizes)], source=source)


def episode_for(split: str, i: int) -> Episode:
    assert 0 <= i <= MAX_PER_SPLIT
    if is_audit(split):
        return make_audit_episode(SPLITS[split][0] + i, config_for(split, i))
    return make_episode(SPLITS[split][0] + i, config_for(split, i))


def _audit_one(args) -> dict:
    return episode_for(*args).to_json()


def write_audit_splits(out_dir: str = "data", only: tuple[str, ...] = AUDIT_EVAL_SPLITS + AUDIT_STREAMS, workers: int = 1) -> None:
    """Write the frozen audit splits (``<name>.jsonl``) and the stream definitions (``<name>.json``).
    Touches no other split. SGD-poisoned episodes take about half a second each; ``workers`` parallelises."""
    from multiprocessing import Pool

    os.makedirs(out_dir, exist_ok=True)
    for name in only:
        seed0, n, families, faults, sizes, source = SPLITS[name]
        assert is_audit(name), name
        if n == 0:
            with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
                json.dump({"split": name, "seed0": seed0, "families": families, "sizes": sizes, "source": source,
                           "config": config_for(name, 0).to_json(),
                           "note": "episode i = make_audit_episode(seed0 + i, AuditConfig(size=sizes[i % 2], source=...)); "
                                   "see splits.config_for"}, f, indent=1)
            continue
        jobs = [(name, i) for i in range(n)]
        if workers > 1:
            with Pool(workers) as pool:
                rows = pool.map(_audit_one, jobs, chunksize=8)
        else:
            rows = [_audit_one(j) for j in jobs]
        eps = [Episode.from_json(r) for r in rows]
        save_split(os.path.join(out_dir, f"{name}.jsonl"), eps, config_for(name, 0), name)
        print(f"{name:14s} n={n} clean={sum(e.is_clean for e in eps)} weights median={sorted(e.net.n_weights for e in eps)[n // 2]} "
              f"max={max(e.net.n_weights for e in eps)}")


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
    ap.add_argument("--only", default=None)
    ap.add_argument("--audit", action="store_true", help="write the backdoor-audit splits instead (no other file is touched)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.audit:
        write_audit_splits(a.out, tuple(a.only.split(",")) if a.only else AUDIT_EVAL_SPLITS + AUDIT_STREAMS, a.workers)
    else:
        write_splits(a.out, tuple((a.only or ",".join(EVAL_SPLITS)).split(",")))
