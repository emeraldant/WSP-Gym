import json
import os

import pytest

from weightgym.env import HELDOUT_FAMILIES, HELDOUT_FAULTS, TRAIN_FAMILIES, TRAIN_FAULTS, load_split
from weightgym.splits import EVAL_SPLITS, MAX_PER_SPLIT, SPLITS, episode_for, seed_range
from weightgym.tinker_train import target_text
from weightgym.env import score_answer

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_seed_ranges_are_disjoint():
    ranges = sorted(seed_range(name) for name in SPLITS)
    for (_, hi), (lo, _) in zip(ranges, ranges[1:]):
        assert hi < lo


def test_split_definitions():
    assert SPLITS["test_families"][2] == HELDOUT_FAMILIES and SPLITS["test_families"][3] == TRAIN_FAULTS
    assert SPLITS["test_faults"][2] == TRAIN_FAMILIES and SPLITS["test_faults"][3] == HELDOUT_FAULTS
    for name in ("train", "val", "test_id", "test_larger", "test_trained"):
        assert SPLITS[name][2] == TRAIN_FAMILIES and SPLITS[name][3] == TRAIN_FAULTS
    assert max(SPLITS["train"][4]) < min(SPLITS["test_larger"][4])


def test_training_stream_is_deterministic_and_in_distribution():
    for i in range(40):
        a, b = episode_for("train", i), episode_for("train", i)
        assert a.prompt() == b.prompt() and a.prompt(direct=True) != a.prompt()
        assert a.family in TRAIN_FAMILIES and (a.fault is None or a.fault in TRAIN_FAULTS)
        assert a.seed == SPLITS["train"][0] + i


def test_supervised_target_is_a_valid_repair():
    for i in range(60):
        ep = episode_for("train", i)
        res = score_answer(ep, target_text(ep))
        assert res["valid"] and res["success"]
        assert json.loads(target_text(ep)) == [] if ep.is_clean else res["n_ops"] >= 1


@pytest.mark.parametrize("name", [s for s in EVAL_SPLITS if SPLITS[s][5] == "compiled"])
def test_frozen_split_matches_generator(name):
    path = os.path.join(DATA, f"{name}.jsonl")
    if not os.path.exists(path):
        pytest.skip("split not generated")
    header, eps = load_split(path)
    assert header["n"] == len(eps) == SPLITS[name][1]
    assert len({e.seed for e in eps}) == len(eps)
    lo = SPLITS[name][0]
    assert all(lo <= e.seed <= lo + MAX_PER_SPLIT for e in eps)
    for i in list(range(8)) + [len(eps) - 1]:
        fresh = episode_for(name, i)
        assert fresh.prompt() == eps[i].prompt() and fresh.net.equals(eps[i].net)
    with open(path) as f:
        f.readline()
        stored = json.loads(f.readline())
    assert stored["prompt_hash"] == eps[0].prompt_hash()
