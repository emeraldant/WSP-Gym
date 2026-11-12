"""Hub image-classifier heads (weightgym/imagehead.py). Skipped when the feature cache is absent
(run `PYTHONPATH=. uv run python -m weightgym.imagehead extract` to build it)."""

import json
import os

import numpy as np
import pytest

from weightgym import imagehead as ih
from weightgym.edits import MAX_OPS
from weightgym.eval import record

pytestmark = pytest.mark.skipif(
    not os.path.exists(ih.RECORDS) or not os.path.isdir(ih.CACHE) or not any(f.endswith(".npz") for f in os.listdir(ih.CACHE)),
    reason="no cached image-head features",
)


@pytest.fixture(scope="module")
def items():
    its = ih.load_items()
    assert its, "records exist but no model has cached features"
    return its


def test_records_are_complete():
    recs = ih.load_records()
    assert len({r["id"] for r in recs}) == len(recs) and len({r["seed"] for r in recs}) == len(recs)
    for r in recs:
        assert r["seed"] == ih.seed_for(r["id"]) and r["head_weights"] == r["n_classes"] * r["feature_dim"] + r["n_classes"]
        assert 0 < r["claimed_accuracy"] <= 1 and r["claimed_source"] and r["report_split"] == "test" and r["select_split"] == "train"
        assert r["n_select"] <= 5000 and r["n_report"] <= 5000


def test_head_reproduces_recorded_accuracy(items):
    for it in items:
        r = it.rec
        assert it.net.n_layers == 1 and it.net.Ws[0].shape == (r["n_classes"], r["feature_dim"])
        assert it.acc_select == pytest.approx(r["select_accuracy"], abs=1e-12) and it.acc_report == pytest.approx(r["test_accuracy"], abs=1e-12)
        # the generic MLP forward pass (double precision) gives the same accuracy, and the head agrees with the model itself
        assert float(np.mean(np.argmax(it.net.forward(it.F_report), 1) == it.y_report)) == pytest.approx(r["test_accuracy"], abs=0.001)
        assert float(np.mean(ih._preds(it.net, it.F_report) == it.pred_report_model)) >= 0.999
        assert abs(r["test_accuracy_3sf"] - r["test_accuracy"]) <= 0.001 + 1e-12  # printing 3 significant figures loses nothing


def test_score_noop_invalid_and_keys(items):
    keys = {"reward", "valid", "success", "recovery", "acc_after", "n_ops", "noop", "error", "ops", "gain"}
    for it in items:
        res = it.score("[]")
        assert set(res) == keys and res["reward"] == 0 and res["valid"] and res["noop"] and res["gain"] == 0
        assert record(it, res)["fault"] == "imperfect" and record(it, res)["family"] == it.rec["dataset"]
        too_many = json.dumps([{"op": "add", "tensor": "b1", "index": 0, "value": 0.1}] * (MAX_OPS + 1))
        for bad in ("", "no edit here", '[{"op": "scale", "tensor": "W2", "axis": "row", "index": 0, "factor": 2}]',
                    '[{"op": "set", "tensor": "b1", "index": 999, "value": 0}]', too_many):
            res = it.score(bad)
            assert res["reward"] == -1 and not res["valid"] and res["error"]
        res = it.score('I would try [{"op": "scale", "tensor": "W1", "axis": "row", "index": 0, "factor": 0}]')
        assert res["valid"] and 0 <= res["reward"] <= 1 and res["recovery"] <= res["reward"]


def test_prompts_are_deterministic_and_bounded(items):
    again = {it.hub_id: it for it in ih.load_items()}
    for it in items:
        for direct in (False, True):
            p = it.prompt(direct)
            assert p == again[it.hub_id].prompt(direct) and it.prompt_hash(direct) == again[it.hub_id].prompt_hash(direct)
            assert len(p) <= ih.PROMPT_CAP and it.hub_id in p and "W1" in p and "W2" not in p
        assert it.prompt(True).startswith(it.prompt(False)) and it.prompt(True) != it.prompt(False)
        assert len(it.shown_columns) == (ih.N_SHOWN_COLS if it.restricted else it.rec["feature_dim"])


def test_hidden_columns_cannot_be_addressed(items):
    for it in items:
        if not it.restricted:
            continue
        hidden = int(np.setdiff1d(np.arange(it.rec["feature_dim"]), it.shown_columns)[0])
        shown = int(it.shown_columns[0])
        assert it.score(json.dumps([{"op": "set", "tensor": "W1", "index": [0, hidden], "value": 0.0}]))["reward"] == -1
        assert it.score(json.dumps([{"op": "scale", "tensor": "W1", "axis": "col", "index": hidden, "factor": 0}]))["reward"] == -1
        assert it.score(json.dumps([{"op": "set", "tensor": "W1", "index": [0, shown], "value": 0.0}]))["valid"]
        assert it.score(json.dumps([{"op": "scale", "tensor": "W1", "axis": "row", "index": 0, "factor": 1.1}]))["valid"]


def test_verify_and_report(items):
    it = items[0]
    none = it.verify_and_report([[], "not an edit", [{"op": "scale", "tensor": "W1", "axis": "row", "index": 0, "factor": 0}]])
    assert set(none) >= {"accepted", "change_points", "report_before", "report_after", "select_before", "select_after"}
    if not none["accepted"]:
        assert none["change_points"] == 0 and none["report_after"] == it.acc_report
    rng = np.random.default_rng(0)
    props = [ih.random_valid_edit(it, rng) for _ in range(8)]
    a, b = it.verify_and_report(props), it.verify_and_report(props)
    assert a == b and a["select_after"] >= a["select_before"] and a["accepted"] == (a["select_after"] > a["select_before"])
    bias = ih.best_bias_edit(it)
    assert len(bias) <= 1 and it.verify_and_report([bias])["select_after"] >= it.acc_select
