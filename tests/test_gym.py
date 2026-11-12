import json

import numpy as np
import pytest

from weightgym import nets
from weightgym.edits import MAX_OPS, EditError, apply_edit, extract_json_list, parse_edit, validate_edit
from weightgym.env import (
    HELDOUT_FAMILIES, HELDOUT_FAULTS, TRAIN_FAMILIES, TRAIN_FAULTS, Episode, GymConfig, load_split,
    make_episode, save_split, score_answer,
)
from weightgym.faults import FAULTS, plant_fault
from weightgym.nets import FAMILIES, sample_task

ALL = GymConfig(families=tuple(FAMILIES), faults=tuple(FAULTS))


def _nets(n=8, size=1):
    rng = np.random.default_rng(0)
    for fam in FAMILIES:
        for _ in range(n):
            yield (*sample_task(fam, rng, size), rng)


# --- networks --------------------------------------------------------------------


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("size", [0, 1, 2])
def test_compiled_networks_are_exact(family, size):
    rng = np.random.default_rng(1)
    for _ in range(25):
        task, net = sample_task(family, rng, size)
        assert task.accuracy(net) == 1.0
        rebuilt_task, rebuilt = nets.build_task(family, json.loads(json.dumps(task.params)))
        assert rebuilt.equals(net) and np.array_equal(rebuilt_task.X, task.X) and np.array_equal(rebuilt_task.Y, task.Y)


def test_families_and_faults_partition():
    assert set(TRAIN_FAMILIES) | set(HELDOUT_FAMILIES) == set(FAMILIES) and not set(TRAIN_FAMILIES) & set(HELDOUT_FAMILIES)
    assert set(TRAIN_FAULTS) | set(HELDOUT_FAULTS) == set(FAULTS) and not set(TRAIN_FAULTS) & set(HELDOUT_FAULTS)
    assert len(TRAIN_FAMILIES) >= 8 and len(HELDOUT_FAMILIES) == 2 and len(HELDOUT_FAULTS) == 2


# --- symmetries preserve the function ------------------------------------------------


def _same(a, b, X):
    return np.allclose(a.forward(X), b.forward(X), atol=1e-9)


def test_permutation_preserves_function():
    for task, net, rng in _nets():
        for l in range(net.n_layers - 1):
            assert _same(net, nets.permute_hidden(net, l, rng.permutation(net.Ws[l].shape[0])), task.X)


def test_rescaling_preserves_function():
    for task, net, rng in _nets():
        for l in range(net.n_layers - 1):
            assert _same(net, nets.rescale_hidden(net, l, rng.uniform(0.3, 3.0, net.Ws[l].shape[0])), task.X)


def test_split_preserves_function():
    for task, net, rng in _nets():
        for l in range(net.n_layers - 1):
            out = nets.split_neuron(net, l, int(rng.integers(net.Ws[l].shape[0])), float(rng.uniform(0.2, 0.8)))
            assert out.Ws[l].shape[0] == net.Ws[l].shape[0] + 1 and _same(net, out, task.X)


def test_distractors_preserve_function():
    for task, net, rng in _nets():
        for l in range(net.n_layers - 1):
            assert _same(net, nets.add_dead_neuron(net, l, task.X, rng), task.X)
            assert _same(net, nets.add_cancel_pair(net, l, rng), task.X)


def test_obfuscation_keeps_accuracy_and_rounds():
    cfg = GymConfig(size=1).obf()
    for task, net, rng in _nets():
        out = nets.obfuscate(net, task, rng, cfg)
        assert task.accuracy(out) >= cfg.min_clean_acc
        for T in out.Ws + out.bs:
            assert np.array_equal(T, np.round(T, nets.DECIMALS))
        assert all(o.shape[0] > n.shape[0] for o, n in zip(out.Ws[:-1], net.Ws[:-1]))


# --- faults ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(FAULTS))
def test_fault_breaks_and_inverse_restores(kind):
    cfg = GymConfig(size=1)
    n = 0
    for task, net, rng in _nets(n=6):
        clean = nets.obfuscate(net, task, rng, cfg.obf())
        planted = plant_fault(clean, task, kind, rng, cfg.min_drop)
        if planted is None:
            continue
        faulty, inverse, _, acc_faulty = planted
        n += 1
        acc_clean = task.accuracy(clean)
        assert acc_clean - acc_faulty >= cfg.min_drop
        assert not faulty.equals(clean)
        restored = apply_edit(faulty, validate_edit(inverse, faulty))
        assert len(inverse) <= MAX_OPS and task.accuracy(restored) >= acc_clean - 1e-3 * (acc_clean - acc_faulty) - 1e-12
    assert n >= 5


def test_sign_flips_and_swaps_keep_weight_multiset():
    cfg = GymConfig(size=1)
    for kind in ("sign_flip_out", "sign_flip_in", "swap_out_cols", "label_swap"):
        for task, net, rng in _nets(n=3):
            clean = nets.obfuscate(net, task, rng, cfg.obf())
            planted = plant_fault(clean, task, kind, rng)
            if planted is None:
                continue
            a = np.sort(np.abs(np.concatenate([t.ravel() for t in clean.Ws + clean.bs])))
            b = np.sort(np.abs(np.concatenate([t.ravel() for t in planted[0].Ws + planted[0].bs])))
            assert np.array_equal(a, b)


# --- DSL ------------------------------------------------------------------------------


def _net():
    rng = np.random.default_rng(3)
    return nets.MLP([rng.normal(size=(4, 3)), rng.normal(size=(2, 4))], [rng.normal(size=4), rng.normal(size=2)])


VALID = [
    {"op": "scale", "tensor": "W2", "axis": "col", "index": 3, "factor": -1},
    {"op": "scale", "tensor": "W1", "axis": "row", "index": 0, "factor": 0.5},
    {"op": "scale", "tensor": "b1", "index": 2, "factor": 2},
    {"op": "swap", "tensor": "W2", "axis": "col", "i": 1, "j": 3},
    {"op": "swap", "tensor": "W2", "axis": "row", "i": 0, "j": 1},
    {"op": "swap", "tensor": "b2", "i": 0, "j": 1},
    {"op": "set", "tensor": "W1", "index": [3, 0], "value": 0.5},
    {"op": "set", "tensor": "b1", "index": 3, "value": -1.5},
    {"op": "add", "tensor": "W2", "axis": "col", "index": 2, "vector": [0.5, -0.5]},
    {"op": "add", "tensor": "W1", "axis": "row", "index": 1, "vector": [1, 2, 3]},
    {"op": "add", "tensor": "b2", "index": 1, "value": 0.25},
]


@pytest.mark.parametrize("op", VALID)
def test_dsl_roundtrip_and_locality(op):
    net = _net()
    norm = validate_edit(json.loads(json.dumps([op])), net)
    assert validate_edit(json.loads(json.dumps(norm)), net) == norm  # parse/serialise round trip
    out = apply_edit(net, norm)
    changed = [not np.array_equal(a, b) for a, b in zip(net.Ws + net.bs, out.Ws + out.bs)]
    assert sum(changed) == 1  # exactly one tensor touched
    names = ["W1", "W2", "b1", "b2"]
    T0, T1 = (net.Ws + net.bs)[names.index(op["tensor"])], (out.Ws + out.bs)[names.index(op["tensor"])]
    diff = T0 != T1
    if T0.ndim == 2:  # one row or one column (two for a swap)
        lines = max(1, 2 if op["op"] == "swap" else 1)
        assert min(diff.any(axis=1).sum(), diff.any(axis=0).sum()) <= lines
    assert _net().equals(net)  # apply_edit does not mutate its input


def test_dsl_semantics():
    net = _net()
    out = apply_edit(net, validate_edit([{"op": "scale", "tensor": "W2", "axis": "col", "index": 3, "factor": -1}], net))
    assert np.array_equal(out.Ws[1][:, 3], -net.Ws[1][:, 3])
    out = apply_edit(net, validate_edit([{"op": "swap", "tensor": "W2", "axis": "col", "i": 1, "j": 3}], net))
    assert np.array_equal(out.Ws[1][:, 1], net.Ws[1][:, 3]) and np.array_equal(out.Ws[1][:, 3], net.Ws[1][:, 1])
    out = apply_edit(net, validate_edit([{"op": "set", "tensor": "W1", "index": [3, 0], "value": 0.5}], net))
    assert out.Ws[0][3, 0] == 0.5
    out = apply_edit(net, validate_edit([{"op": "add", "tensor": "W2", "axis": "col", "index": 2, "vector": [0.5, -0.5]}], net))
    assert np.allclose(out.Ws[1][:, 2], net.Ws[1][:, 2] + [0.5, -0.5])


INVALID = [
    "not json", "{}", '{"op": "scale"}', '[{"op": "delete", "tensor": "W1"}]',
    '[{"op": "scale", "tensor": "W3", "axis": "col", "index": 0, "factor": 1}]',
    '[{"op": "scale", "tensor": "W2", "axis": "col", "index": 4, "factor": 1}]',
    '[{"op": "scale", "tensor": "W2", "axis": "col", "index": 0, "factor": "x"}]',
    '[{"op": "scale", "tensor": "W2", "axis": "col", "index": 0, "factor": 1e9}]',
    '[{"op": "scale", "tensor": "W2", "axis": "diag", "index": 0, "factor": 1}]',
    '[{"op": "scale", "tensor": "W2", "axis": "col", "index": 0, "factor": 1, "extra": 1}]',
    '[{"op": "set", "tensor": "W1", "index": 3, "value": 1}]',
    '[{"op": "add", "tensor": "W2", "axis": "col", "index": 2, "vector": [1, 2, 3]}]',
    '[{"op": "swap", "tensor": "W2", "axis": "col", "i": 0, "j": true}]',
    json.dumps([VALID[0]] * (MAX_OPS + 1)),  # over budget
]


@pytest.mark.parametrize("text", INVALID)
def test_dsl_rejects_invalid(text):
    with pytest.raises(EditError):
        parse_edit(text, _net())


def test_extract_takes_last_list_and_ignores_prose():
    text = 'Neuron W2[3] looks wrong, maybe [] ... Final answer:\n```json\n[{"op": "set", "tensor": "b1", "index": [3], "value": 1}]\n```'
    assert extract_json_list(text) == [{"op": "set", "tensor": "b1", "index": [3], "value": 1}]
    assert parse_edit(text, _net())[0]["index"] == 3
    assert extract_json_list("nothing to fix: []") == []
    with pytest.raises(EditError):
        extract_json_list("the weights are [1, 2, 3]")


# --- episodes and rewards ---------------------------------------------------------------


def test_episode_is_deterministic_and_serialisable(tmp_path):
    eps = [make_episode(s, ALL) for s in range(60)]
    again = [make_episode(s, ALL) for s in range(60)]
    assert [e.prompt() for e in eps] == [e.prompt() for e in again]
    assert len({e.prompt_hash() for e in eps}) == len(eps)
    path = tmp_path / "split.jsonl"
    save_split(str(path), eps, ALL, "t")
    header, loaded = load_split(str(path))
    assert GymConfig.from_json(header["config"]) == ALL
    for a, b in zip(eps, loaded):
        assert a.prompt() == b.prompt() and a.net.equals(b.net) and a.clean.equals(b.clean)
        assert score_answer(a, json.dumps(a.inverse_edit)) == score_answer(b, json.dumps(b.inverse_edit))


def test_rewards_in_bounds_and_behavioural():
    rng = np.random.default_rng(0)
    from weightgym.baselines import random_policy

    n_clean = 0
    for s in range(300):
        ep = make_episode(s, ALL)
        n_clean += ep.is_clean
        assert ep.acc_clean >= 0.99
        for text in ("[]", "garbage", json.dumps(ep.inverse_edit), random_policy(ep, rng)):
            r = score_answer(ep, text)
            assert -1.0 <= r["reward"] <= 1.0
        assert score_answer(ep, "garbage")["reward"] == -1.0
        if ep.is_clean:
            assert ep.net.equals(ep.clean) and not ep.failures
            assert score_answer(ep, "[]")["reward"] == 1.0
            r = score_answer(ep, random_policy(ep, rng))
            assert -1.0 <= r["reward"] <= -0.25 and not r["success"]
        else:
            assert ep.acc_clean - ep.acc_faulty >= ALL.min_drop
            assert score_answer(ep, "[]")["reward"] == 0.0 and not score_answer(ep, "[]")["success"]
            r = score_answer(ep, json.dumps(ep.inverse_edit))
            assert r["success"] and r["reward"] >= 0.95
            if ep.variant == "weights+failures":
                assert 1 <= len(ep.failures) <= ALL.max_failures
                assert not ep.task.correct(ep.net)[ep.failures].any()
            else:
                assert "Failing inputs" not in ep.prompt()
    assert 0.10 <= n_clean / 300 <= 0.26


def test_functionally_equivalent_edit_is_accepted():
    """A repair that differs from the planted fault's literal inverse still scores as repaired."""
    cfg = GymConfig(faults=("sign_flip_out",), p_clean=0.0)
    hits = 0
    for s in range(40):
        ep = make_episode(s, cfg)
        j, l = ep.fault_info["neuron"], ep.fault_info["layer"]
        # relu(z) * (-w) can also be repaired by scaling the column by -3 and the row+bias... not in one op,
        # but "set"-ting every entry of a single-output column is the same function with a different edit.
        if ep.net.Ws[l + 1].shape[0] == 1:
            alt = [{"op": "set", "tensor": f"W{l + 2}", "index": [0, j], "value": float(-ep.net.Ws[l + 1][0, j])}]
            assert alt != ep.inverse_edit and score_answer(ep, json.dumps(alt))["success"]
            hits += 1
    assert hits > 5
