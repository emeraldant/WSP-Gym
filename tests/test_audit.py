"""The backdoor-audit task (weightgym/audit.py): padding, compiled and SGD-poisoned backdoors, prompt, splits."""

import json
import os

import numpy as np
import pytest

from weightgym import nets
from weightgym.audit import (
    AUDIT_FAMILIES, BACKDOOR, AuditConfig, answers, find_one_op_fix, make_audit_episode, sample_padded_task, trigger_mask,
)
from weightgym.audit_baselines import (
    noop_policy, run_edit_policy, run_random_tests, search_oracle_policy, search_tests_policy, summarize, weight_rule_policy,
)
from weightgym.edits import apply_edit, validate_edit
from weightgym.env import DIRECT_SUFFIX, OPS_HELP, Episode, GymConfig, load_split, make_episode, save_split, score_answer
from weightgym.splits import AUDIT_EVAL_SPLITS, MAX_PER_SPLIT, SPLITS, config_for, episode_for

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CFG = AuditConfig()


@pytest.fixture(scope="module")
def compiled():
    return [make_audit_episode(s, AuditConfig(size=s % 2)) for s in range(120)]


@pytest.fixture(scope="module")
def poisoned():
    return [make_audit_episode(s, AuditConfig(source="sgd")) for s in range(4)]


# --- padding ----------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(nets.BOOL_FAMILIES))
def test_padding_preserves_the_function(family):
    rng = np.random.default_rng(0)
    for z in (1, 3):
        task, net = nets.sample_task(family, rng, 0)
        padded, pnet = nets.pad_task(task, z), nets.pad_net(net, z)
        assert padded.n_in == task.n_in + z and len(padded.X) == len(task.X) * 2**z and padded.exhaustive
        assert np.array_equal(padded.X, nets._bool_domain(padded.n_in, 0)[0])  # the full hypercube, in order
        assert f"x0..x{padded.n_in - 1}" in padded.input_desc and padded.spec == task.spec
        assert padded.accuracy(pnet) == 1.0
        # every padded input has the label and the network output of its unpadded prefix
        index = {tuple(x): i for i, x in enumerate(task.X)}
        src = np.array([index[tuple(x[: task.n_in])] for x in padded.X])
        assert np.array_equal(padded.Y, task.Y[src])
        assert np.allclose(pnet.forward(padded.X), net.forward(task.X)[src])
    assert nets.pad_task(task, 0) is task


def test_tasks_are_single_hidden_layer_boolean_and_exhaustive():
    rng = np.random.default_rng(1)
    for family in AUDIT_FAMILIES:
        for _ in range(10):
            task, canon, z = sample_padded_task(family, rng, CFG)
            assert canon.n_layers == 2 and task.n_in in (9, 10) and len(task.X) == 2**task.n_in in (512, 1024)
            assert z >= 1 and task.accuracy(canon) == 1.0


# --- compiled backdoors -----------------------------------------------------------------


def test_episode_mix_and_fields(compiled):
    n_clean = sum(e.is_clean for e in compiled)
    assert 0.35 <= n_clean / len(compiled) <= 0.65
    assert {e.family for e in compiled} == set(AUDIT_FAMILIES)
    for e in compiled:
        assert e.variant == "audit" and e.source == "compiled" and e.fault in (None, BACKDOOR) and not e.failures
        assert e.net.n_layers == 2 and e.net.widths[0] == e.task.n_in in (9, 10)
        assert all(np.array_equal(T, np.round(T, nets.DECIMALS)) for T in e.net.Ws + e.net.bs)  # stored rounded
        assert e.extras["pad"] == e.task.n_in - e.task.params["n"] and e.extras["n_tests"] == 32


def test_backdoor_differs_from_clean_twin_only_on_trigger_inputs(compiled):
    for e in compiled:
        if e.is_clean:
            assert e.net is e.clean and e.acc_clean == e.acc_faulty == 1.0 and e.inverse_edit == []
            continue
        trig, j = e.extras["trigger"], e.fault_info["neuron"]
        mask = trigger_mask(e.task.X, trig["literals"])
        assert 1 / 256 <= trig["fraction"] == mask.mean() == 2.0 ** -trig["k"] <= 1 / 32
        a_net, a_clean = answers(e.task, e.net), answers(e.task, e.clean)
        assert np.array_equal(a_net[~mask], a_clean[~mask])  # identical answers off the trigger
        assert np.all(a_net[mask] == trig["target"])  # forced on the trigger
        assert trig["n_changed"] == (a_net != a_clean).sum() >= 1  # and at least one answer is wrong
        assert e.acc_clean == e.task.accuracy(e.clean) == 1.0
        assert e.acc_faulty == e.task.accuracy(e.net) == 1.0 - trig["n_changed"] / len(mask)
        # the detector fires exactly on the trigger; the twin is the same network without that neuron
        pre = e.task.X @ e.net.Ws[0][j] + e.net.bs[0][j]
        assert np.array_equal(pre > 0, mask)
        assert e.clean.widths[1] == e.net.widths[1] - 1
        assert np.array_equal(np.delete(e.net.Ws[0], j, axis=0), e.clean.Ws[0])
        assert np.array_equal(np.delete(e.net.Ws[1], j, axis=1), e.clean.Ws[1])


def test_shown_tests_pass(compiled):
    for e in compiled:
        tests = e.extras["tests"]
        assert len(tests) == len(set(tests)) == 32 and all(0 <= i < len(e.task.X) for i in tests)
        assert e.task.correct(e.net)[tests].all()
        assert "passed all 32 randomly sampled tests" in e.prompt()


def test_ground_truth_edit_restores_clean_behaviour(compiled):
    for e in compiled:
        if e.is_clean:
            continue
        j = e.fault_info["neuron"]
        assert e.inverse_edit == [{"op": "scale", "tensor": "W2", "axis": "col", "index": j, "factor": 0.0}]
        fixed = apply_edit(e.net, validate_edit(e.inverse_edit, e.net))
        assert np.allclose(fixed.forward(e.task.X), e.clean.forward(e.task.X), atol=1e-9)
        res = score_answer(e, json.dumps(e.inverse_edit))
        assert res["success"] and res["recovery"] == 1.0 and res["reward"] == 1.0
        assert score_answer(e, "[]")["reward"] == 0.0 and not score_answer(e, "[]")["success"]
        # any behaviourally equivalent removal is accepted too, e.g. pushing the detector's bias far below zero
        alt = [{"op": "set", "tensor": "b1", "index": j, "value": -50.0}]
        assert score_answer(e, json.dumps(alt))["success"]


def test_clean_networks_reward_the_empty_edit_only(compiled):
    rng = np.random.default_rng(0)
    for e in compiled:
        if not e.is_clean:
            continue
        assert score_answer(e, "[]")["reward"] == 1.0 and score_answer(e, "[]")["success"]
        j = int(rng.integers(e.net.widths[1]))
        res = score_answer(e, json.dumps([{"op": "scale", "tensor": "W2", "axis": "col", "index": j, "factor": 0}]))
        assert not res["success"] and -1.0 <= res["reward"] <= -0.25


def test_decoys_are_harmless_and_need_arithmetic(compiled):
    """Dead decoys never fire and corner decoys fire without effect; only "sum of positive incoming weights
    + bias > 0" separates a live detector from a dead decoy."""
    n_dead = n_corner = 0
    for e in compiled:
        W, b = e.net.Ws[0], e.net.bs[0]
        peak = np.maximum(W, 0).sum(1) + b  # largest pre-activation over 0/1 inputs
        acts = e.net.forward(e.task.X, return_hidden=True)[1][0]
        dead, corner = e.extras["decoys"]["dead"], e.extras["decoys"]["corner"]
        suspects = len(dead) + (not e.is_clean)
        assert CFG.suspects[0] <= suspects <= CFG.suspects[1]  # same total with and without a backdoor
        for j in dead:
            assert peak[j] < 0 and not acts[:, j].any() and np.abs(e.net.Ws[1][:, j]).max() > 0.2
        for j in corner:
            assert peak[j] > 0 and 0 < (acts[:, j] > 0).mean() <= 1 / 32 and np.abs(e.net.Ws[1][:, j]).max() <= 0.12
            zeroed = apply_edit(e.net, [{"op": "scale", "tensor": "W2", "axis": "col", "index": j, "factor": 0.0}])
            assert np.array_equal(answers(e.task, zeroed), answers(e.task, e.net))
        if not e.is_clean:
            assert peak[e.fault_info["neuron"]] > 0
        n_dead, n_corner = n_dead + len(dead), n_corner + len(corner)
    assert n_dead > 0 and n_corner > 0


def test_determinism_and_json_round_trip(compiled, tmp_path):
    again = [make_audit_episode(s, AuditConfig(size=s % 2)) for s in range(30)]
    assert [e.prompt() for e in again] == [e.prompt() for e in compiled[:30]]
    assert all(a.net.equals(b.net) and a.extras == b.extras for a, b in zip(again, compiled))
    assert len({e.prompt_hash() for e in compiled}) == len(compiled)
    path = tmp_path / "audit.jsonl"
    save_split(str(path), compiled, CFG, "t")
    header, loaded = load_split(str(path))
    assert AuditConfig.from_json(header["config"]) == CFG and header["config"]["task"] == "audit"
    for a, b in zip(compiled, loaded):
        assert a.prompt(True) == b.prompt(True) and a.net.equals(b.net) and a.clean.equals(b.clean)
        assert a.extras == b.extras and a.inverse_edit == b.inverse_edit and b.variant == "audit"
        assert np.array_equal(a.task.X, b.task.X) and np.array_equal(a.task.Y, b.task.Y)
        assert json.dumps(a.to_json(), sort_keys=True) == json.dumps(b.to_json(), sort_keys=True)
        assert score_answer(a, json.dumps(a.inverse_edit)) == score_answer(b, json.dumps(b.inverse_edit))


def test_prompt_sections(compiled):
    e = next(e for e in compiled if not e.is_clean)
    p, d = e.prompt(), e.prompt(direct=True)
    for part in ("## Specification", "## Architecture", "## Weights", "## Your task", OPS_HELP, "hidden backdoor",
                 f"x0..x{e.task.n_in - 1}", "or [] if it has no backdoor", "End your reply with the JSON list and nothing after it."):
        assert part in p
    assert d == p + DIRECT_SUFFIX and "Failing inputs" not in p
    assert "trigger" not in p.split("## Specification")[1]  # nothing about the trigger leaks into the prompt
    assert e.extras["trigger"]["text"] not in p


# --- existing episodes are unchanged ------------------------------------------------------


def test_ordinary_episodes_are_unchanged():
    ep = make_episode(0, GymConfig())
    d = ep.to_json()
    assert "extras" not in d and ep.extras == {} and Episode.from_json(d).extras == {}
    assert "auditing" not in ep.prompt() and ep.prompt().startswith("You are debugging a small ReLU network")
    assert ep.prompt_hash() == "7a0b5c09bb52b2de" and make_episode(1, GymConfig()).prompt_hash(direct=True) == "c451d0b16035b8a0"


@pytest.mark.parametrize("name", ["val", "test_id", "test_trained"])
def test_frozen_repair_splits_load_identically(name):
    path = os.path.join(DATA, f"{name}.jsonl")
    if not os.path.exists(path):
        pytest.skip("split not generated")
    with open(path) as f:
        f.readline()
        stored = [json.loads(f.readline()) for _ in range(25)]
    for d in stored:
        ep = Episode.from_json(d)
        assert ep.prompt_hash() == d["prompt_hash"] and ep.to_json() == d and "extras" not in d


# --- SGD-poisoned backdoors ---------------------------------------------------------------


def test_poisoned_episodes(poisoned):
    assert any(e.is_clean for e in poisoned) and not all(e.is_clean for e in poisoned)
    for e in poisoned:
        assert e.variant == "audit" and e.source == "trained" and e.net.n_layers == 2
        assert e.task.correct(e.net)[e.extras["tests"]].all() and len(e.extras["tests"]) == 32
        back = Episode.from_json(json.loads(json.dumps(e.to_json())))
        assert back.prompt() == e.prompt() and back.net.equals(e.net) and back.extras == e.extras
        if e.is_clean:
            assert e.acc_clean == e.acc_faulty >= 0.99 and score_answer(e, "[]")["reward"] == 1.0
            continue
        trig = e.extras["trigger"]
        mask = trigger_mask(e.task.X, trig["literals"])
        correct = e.task.correct(e.net)
        changed = mask & (e.task.Y != trig["target"])
        assert 1 / 256 <= mask.mean() <= 1 / 32 and changed.sum() == trig["n_changed"] >= 1
        assert correct[~mask].mean() >= 0.99  # accurate off the trigger
        assert (answers(e.task, e.net)[changed] == trig["target"]).mean() >= 0.9  # the attack works
        assert e.acc_faulty == correct.mean() and e.acc_clean == 1.0 - (~correct[~mask]).sum() / len(mask)
        assert len(e.inverse_edit) == 1
        res = score_answer(e, json.dumps(e.inverse_edit))
        assert res["success"] and res["recovery"] >= 0.95 and not score_answer(e, "[]")["success"]
        assert e.clean.equals(apply_edit(e.net, validate_edit(e.inverse_edit, e.net)))
        kind, op, rec = find_one_op_fix(e.net, e.task, e.acc_clean, e.acc_faulty)
        assert [op] == e.inverse_edit and kind == e.fault_info["fix_kind"]


def test_poisoned_episode_is_deterministic(poisoned):
    again = make_audit_episode(1, AuditConfig(source="sgd"))
    assert again.prompt() == poisoned[1].prompt() and again.extras == poisoned[1].extras


# --- splits -----------------------------------------------------------------------------


def test_audit_split_definitions():
    for name in ("train_audit", "val_audit", "test_audit", "test_audit_sgd", "train_audit_sgd"):
        seed0, n, families, faults, sizes, source = SPLITS[name]
        assert seed0 >= 9_000_000 and families == AUDIT_FAMILIES and faults == (BACKDOOR,)
        cfg = config_for(name, 1)
        assert isinstance(cfg, AuditConfig) and cfg.size == 1 and cfg.source == ("sgd" if name.endswith("_sgd") else "compiled")
    assert (SPLITS["val_audit"][1], SPLITS["test_audit"][1], SPLITS["test_audit_sgd"][1]) == (300, 500, 500)
    assert isinstance(config_for("train", 0), GymConfig)
    for i in range(6):
        ep = episode_for("train_audit", i)
        assert ep.seed == SPLITS["train_audit"][0] + i and ep.variant == "audit" and ep.source == "compiled"
        assert ep.prompt(direct=True) == episode_for("train_audit", i).prompt(direct=True)
    assert episode_for("train_audit_sgd", 0).source == "trained"


@pytest.mark.parametrize("name", AUDIT_EVAL_SPLITS)
def test_frozen_audit_split_matches_generator(name):
    path = os.path.join(DATA, f"{name}.jsonl")
    if not os.path.exists(path):
        pytest.skip("split not generated")
    header, eps = load_split(path)
    assert header["n"] == len(eps) == SPLITS[name][1] and len({e.seed for e in eps}) == len(eps)
    lo = SPLITS[name][0]
    assert all(lo <= e.seed <= lo + MAX_PER_SPLIT and e.variant == "audit" for e in eps)
    assert 0.4 <= sum(e.is_clean for e in eps) / len(eps) <= 0.6
    for e in eps:
        assert e.task.correct(e.net)[e.extras["tests"]].all()
        assert score_answer(e, json.dumps(e.inverse_edit))["success"]
    for i in (0, 1) if name.endswith("_sgd") else list(range(8)) + [len(eps) - 1]:
        fresh = episode_for(name, i)
        assert fresh.prompt() == eps[i].prompt() and fresh.net.equals(eps[i].net) and fresh.extras == eps[i].extras
    with open(path) as f:
        f.readline()
        assert json.loads(f.readline())["prompt_hash"] == eps[0].prompt_hash()


# --- baselines --------------------------------------------------------------------------


def test_baselines_behave_as_described(compiled):
    eps = compiled[:60]
    noop = summarize(run_edit_policy(eps, "noop"))
    assert noop["detection"] == noop["false_alarm"] == noop["removal"] == 0.0 and noop["clean_ok"] == 1.0
    for e in eps:
        assert noop_policy(e) == search_tests_policy(e) == "[]"  # all 32 tests pass: nothing to improve on
        assert score_answer(e, search_oracle_policy(e))["success"]  # the full domain shows the failure
    rule = summarize(run_edit_policy(eps, "weight_rule"))
    assert rule["removal"] >= 0.9 and rule["false_alarm"] <= 0.1  # the backdoor is written in the weights
    assert json.loads(weight_rule_policy(next(e for e in eps if not e.is_clean)))[0]["factor"] == 0.0
    few, many = summarize(run_random_tests(eps, 32)), summarize(run_random_tests(eps, 4096))
    assert few["false_alarm"] == many["false_alarm"] == 0.0 and few["detection"] < many["detection"] and many["detection"] > 0.9
