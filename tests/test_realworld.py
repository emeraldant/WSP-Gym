"""Real-data improve task (weightgym/realworld.py): frozen dataset list, frozen splits, items, scoring."""

import json
import os

import numpy as np
import pytest

from weightgym import realworld as rw

HAVE_FILES = os.path.exists(rw.REGISTRY) and all(os.path.exists(os.path.join(rw.DATA_DIR, f"{s}.jsonl")) for s in ("real_val", "real_test_seen", "real_test_unseen"))
pytestmark = pytest.mark.skipif(not HAVE_FILES, reason="real-data registry or splits not generated")
# Tests that need the datasets themselves are skipped when the (git-ignored) cache is absent, so no test uses the network.
needs_data = pytest.mark.skipif(not (HAVE_FILES and rw.cache_ready()), reason="cache/realworld/ is absent (run python -m weightgym.realworld datasets)")
FROZEN = ("real_val", "real_test_seen", "real_test_unseen")


# --- files only (no data needed) ---------------------------------------------------------------


def test_registry_is_24_datasets_with_every_third_held_out():
    reg = rw.registry()
    names = [e["name"] for e in reg["datasets"]]
    assert len(names) == 24 == len(set(names)) and names == sorted(names, key=str.lower)
    assert reg["heldout"] == names[2::3] and len(reg["heldout"]) == 8
    assert reg["train"] == [n for n in names if n not in reg["heldout"]] and len(reg["train"]) == 16
    for e in reg["datasets"]:
        assert e["rows"] >= 1000 and 1 <= len(e["features"]) <= 8 and 2 <= len(e["classes"]) <= 4
        assert e["role"] == ("heldout" if e["name"] in reg["heldout"] else "train")
        assert e["licence"] and e["url"] and sum(e["pool_rows"].values()) == e["rows"]


def test_frozen_splits_keep_heldout_datasets_out_of_training_splits():
    train, held = set(rw.dataset_names("train")), set(rw.dataset_names("heldout"))
    assert not train & held
    seeds = []
    for split in FROZEN:
        header, recs = rw.read_split(split)
        seed0, n, role = rw.SPLITS[split]
        assert len(recs) == n == header["n"] and [r["seed"] for r in recs] == list(range(seed0, seed0 + n)) and seed0 >= 60_000_000
        used = {r["dataset"] for r in recs}
        assert used <= (held if role == "heldout" else train) and not used & (train if role == "heldout" else held)
        seeds += [r["seed"] for r in recs]
        for r in recs:
            rows = {k: set(v) for k, v in r["rows"].items()}
            assert not rows["fit"] & rows["select"] and not rows["fit"] & rows["report"] and not rows["select"] & rows["report"]
            assert set(r["examples"]["wrong"]) | set(r["examples"]["right"]) <= rows["select"]  # nothing from report is ever shown
            assert len(r["examples"]["wrong"]) == 8 and len(r["examples"]["right"]) == 4
            assert r["n_weights"] <= rw.MAX_WEIGHTS and all(rw.HIDDEN[0] <= w <= rw.HIDDEN[1] for w in r["widths"][1:-1])
            assert rw.ACC_RANGE[0] <= r["acc_select"] <= rw.ACC_RANGE[1] and r["acc_select"] <= r["meta"]["acc_full"] - rw.MIN_GAP + 1e-12
    assert len(seeds) == len(set(seeds))


# --- with the cached datasets --------------------------------------------------------------------


@needs_data
def test_roles_never_share_rows_and_data_matches_checksum():
    for e in rw.registry()["datasets"]:
        d = rw.load_dataset(e["name"])  # raises if the cached data does not match the recorded checksum
        p = d.pools
        assert len(np.intersect1d(p["fit"], p["select"])) == len(np.intersect1d(p["fit"], p["report"])) == len(np.intersect1d(p["select"], p["report"])) == 0
        assert np.array_equal(d.X, np.round(d.X, 1)) and d.X.shape[1] == len(e["features"]) and set(np.unique(d.y)) == set(range(len(e["classes"])))
    for split in FROZEN:
        for r in rw.read_split(split)[1][:40]:
            p = rw.load_dataset(r["dataset"]).pools
            assert all(set(r["rows"][k]) <= set(p[k].tolist()) for k in ("fit", "select", "report"))


@needs_data
def test_items_are_deterministic_and_match_the_frozen_file():
    for split in FROZEN:
        recs = rw.read_split(split)[1]
        for i in (0, 7):
            a, b = rw.real_item_for(split, i), rw.real_item_for(split, i)
            assert a.net.equals(b.net) and a.prompt() == b.prompt() and a.to_json() == b.to_json()
            assert json.loads(json.dumps(a.to_json())) == recs[i]


@needs_data
def test_stored_network_reproduces_recorded_accuracy_and_prompts_are_short():
    for split in FROZEN:
        recs = rw.read_split(split)[1]
        for r, it in zip(recs, rw.load_real_split(split)):
            assert it.acc_select == pytest.approx(r["acc_select"], abs=1e-12) and it.acc_report == pytest.approx(r["acc_report"], abs=1e-12)
            assert it.net.equals(rw.MLP.from_json(r["net"])) and all(np.array_equal(W, np.round(W, 2)) for W in it.net.Ws + it.net.bs)
            assert it.prompt_hash() == r["prompt_hash"] and len(it.prompt(True)) < 12_000
            assert (it.fault, it.is_clean, it.variant, it.family) == ("imperfect", False, "weights+failures", r["dataset"])


@needs_data
def test_scoring_and_verification():
    it = rw.load_real_split("real_val", 3)[2]
    res = it.score("[]")
    assert res["reward"] == 0.0 and res["valid"] and res["noop"] and res["gain"] == 0.0 and res["ops"] == []
    for bad in ("no json here", '[{"op": "set", "tensor": "W9", "index": [0, 0], "value": 1}]', "[" + ", ".join(['{"op": "set", "tensor": "b1", "index": 0, "value": 0}'] * 4) + "]"):
        res = it.score(bad)
        assert res["reward"] == -1.0 and not res["valid"] and res["error"] and res["ops"] is None
    # a harmful edit is a valid proposal: reward 0, negative gain, and the verifier does not accept it
    kill = [{"op": "scale", "tensor": f"W{it.net.n_layers}", "axis": "row", "index": 0, "factor": -1.0}]
    res = it.score(json.dumps(kill))
    assert res["valid"] and res["reward"] >= 0.0 and set(res) >= {"reward", "valid", "success", "recovery", "acc_after", "n_ops", "noop", "error", "ops", "gain"}
    if res["gain"] <= 0:
        assert res["reward"] == 0.0 and it.verify_and_report([kill]) == {"gain": 0.0, "gap": 0.0, "accepted": False, "select_gain": 0.0}
    assert it.verify_and_report([]) == it.verify_and_report([[{"op": "nonsense"}]]) == {"gain": 0.0, "gap": 0.0, "accepted": False, "select_gain": 0.0}
    # the reward is the gap closed on select; the report rows are only read by verify_and_report
    g = rw.gradient_reference(it)
    assert g["accepted"] and g["select_gain"] > 0
    from weightgym.eval import record

    rec = record(it, it.score("[]"), sample=0)
    assert rec["family"] == it.dataset and rec["fault"] == "imperfect" and rec["clean"] is False and rec["n_weights"] == it.net.n_weights


@needs_data
def test_training_stream_never_uses_a_heldout_dataset():
    held = set(rw.dataset_names("heldout"))
    seen = set()
    for i in range(48):
        it = rw.real_train_item(i)
        assert it.seed == 60_000_000 + i and it.dataset not in held
        seen.add(it.dataset)
    assert seen <= set(rw.dataset_names("train")) and len(seen) >= 8


@needs_data
@pytest.mark.skipif(not os.path.exists(rw.HF_PATH) or not os.path.isdir(rw.HF_CACHE), reason="Hugging Face case study not built or not cached")
def test_hf_items_reproduce_their_recorded_accuracy():
    recs = [json.loads(line) for line in open(rw.HF_PATH)]
    usable = [r for r in recs if r["status"] == "usable"]
    assert usable and all(r.get("reason") for r in recs if r["status"] == "rejected")
    for r, it in zip(usable, rw.load_hf_items()):
        assert it.seed >= 64_000_000 and it.source == "hf:" + r["repo"] and it.net.n_weights == r["n_weights"]
        assert it.acc_select == pytest.approx(r["accuracy_select"]) and it.acc_report == pytest.approx(r["accuracy_report"])
        assert it.score("[]")["reward"] == 0.0 and it.score("x")["reward"] == -1.0 and len(it.prompt(True)) < 12_000
        assert not set(it.rows["select"]) & set(it.rows["report"])
