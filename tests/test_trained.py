"""The gradient-descent network source (weightgym/trained.py) and trained-source episodes."""

import dataclasses
import json

import numpy as np
import pytest

from weightgym import nets, trained
from weightgym.env import (
    HELDOUT_FAMILIES, TRAIN_FAMILIES, GymConfig, load_split, make_episode, save_split, score_answer,
)
from weightgym.nets import FAMILIES, sample_task
from weightgym.trained import MIN_ACC, train_network


def _task(family, seed, size=0):
    rng = np.random.default_rng(seed)
    task, canon = sample_task(family, rng, size)
    return task, canon, rng


def _is_rounded(net):
    return all(np.array_equal(T, np.round(T, nets.DECIMALS)) for T in net.Ws + net.bs)


# --- train_network ----------------------------------------------------------------------


@pytest.mark.parametrize("family", ["parity", "minmax", "count"])  # one per loss: binary, reg, class
def test_same_seed_gives_identical_network(family):
    out, states = [], []
    for global_seed in (0, 1):
        np.random.seed(global_seed)  # the global RNG must neither be read nor advanced
        before = np.random.get_state()[1].copy()
        task, canon, rng = _task(family, 5)
        out.append(train_network(task, canon, rng))
        states.append(rng.bit_generator.state)
        assert np.array_equal(np.random.get_state()[1], before)
    assert out[0] is not None and out[0].equals(out[1])
    assert states[0] == states[1]  # the caller's rng ends in the same state too
    task, canon, rng = _task(family, 5)
    rng.random()  # a different stream gives a different network
    other = train_network(task, canon, rng)
    assert other is None or not other.equals(out[0])


@pytest.mark.parametrize("family", list(FAMILIES))
def test_trained_networks_are_accurate_rounded_and_sized(family):
    rng = np.random.default_rng(2)
    lo, hi = trained.EXTRA_NEURONS
    n = 0
    for size in (0, 1):
        task, canon = sample_task(family, rng, size)
        net = train_network(task, canon, rng)
        if net is None:
            continue
        n += 1
        assert task.accuracy(net) >= MIN_ACC  # on raw inputs: no hidden input normalisation
        assert _is_rounded(net)
        assert nets.MLP.from_json(json.loads(json.dumps(net.to_json()))).equals(net)
        assert net.n_layers == canon.n_layers
        assert net.widths[0] == task.n_in and net.widths[-1] == task.n_out
        assert all(c + lo <= w <= c + hi for w, c in zip(net.widths[1:-1], canon.widths[1:-1]))
        assert max(np.abs(T).max() for T in net.Ws + net.bs) < 30.0  # ordinary magnitudes
    assert n >= 1


def test_returns_none_when_the_task_cannot_be_fit():
    task, canon, rng = _task("dnf", 0)
    start = rng.bit_generator.state
    assert train_network(task, canon, rng) is not None
    after_fit = rng.bit_generator.state
    # every input twice with opposite labels: no network can exceed 50% accuracy
    impossible = dataclasses.replace(task, X=np.vstack([task.X, task.X]), Y=np.concatenate([task.Y, 1.0 - task.Y]))
    rng.bit_generator.state = start
    assert train_network(impossible, canon, rng) is None
    assert rng.bit_generator.state == after_fit  # restarts do not change how far the caller's rng advances


# --- episodes ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", TRAIN_FAMILIES + HELDOUT_FAMILIES)
def test_trained_episode_builds_is_deterministic_and_repairable(family):
    cfg = GymConfig(source="trained", families=(family,), p_clean=0.0)
    ep, again = make_episode(3, cfg), make_episode(3, cfg)
    assert ep.source == "trained" and ep.family == family and not ep.is_clean
    assert ep.prompt() == again.prompt() and ep.net.equals(again.net) and ep.clean.equals(again.clean)
    assert ep.inverse_edit == again.inverse_edit and ep.failures == again.failures
    assert _is_rounded(ep.clean) and ep.acc_clean >= MIN_ACC
    assert ep.acc_clean == ep.task.accuracy(ep.clean) and ep.acc_faulty == ep.task.accuracy(ep.net)
    assert ep.acc_clean - ep.acc_faulty >= cfg.min_drop
    assert score_answer(ep, json.dumps(ep.inverse_edit))["success"]
    assert score_answer(ep, "[]")["reward"] == 0.0 and not score_answer(ep, "[]")["success"]


@pytest.mark.parametrize("families", [TRAIN_FAMILIES, HELDOUT_FAMILIES], ids=["train", "heldout"])
def test_trained_split_has_clean_controls_and_round_trips(families, tmp_path):
    cfg = GymConfig(source="trained", families=families, p_clean=0.5)
    eps = [make_episode(s, cfg) for s in range(6)]
    assert {e.family for e in eps} <= set(families)
    assert any(e.is_clean for e in eps) and not all(e.is_clean for e in eps)
    path = tmp_path / "trained.jsonl"
    save_split(str(path), eps, cfg, "trained")
    header, loaded = load_split(str(path))
    assert GymConfig.from_json(header["config"]) == cfg
    for a, b in zip(eps, loaded):
        assert b.source == "trained" and a.prompt() == b.prompt() and a.net.equals(b.net) and a.clean.equals(b.clean)
        if a.is_clean:
            assert a.net.equals(a.clean) and not a.failures and score_answer(b, "[]")["reward"] == 1.0
        else:
            assert score_answer(a, json.dumps(a.inverse_edit))["success"]
            assert score_answer(b, json.dumps(b.inverse_edit)) == score_answer(a, json.dumps(a.inverse_edit))


def test_source_only_changes_the_network(monkeypatch):
    """Same seed: the trained and compiled episodes share family and variant but not weights,
    and a compiled episode never trains anything."""
    compiled, trained_ep = make_episode(4, GymConfig()), make_episode(4, GymConfig(source="trained"))
    assert compiled.source == "compiled" and trained_ep.source == "trained"
    assert (compiled.family, compiled.variant, compiled.is_clean) == (trained_ep.family, trained_ep.variant, trained_ep.is_clean)
    assert not compiled.clean.equals(trained_ep.clean)

    def boom(*args, **kwargs):
        raise AssertionError("compiled episodes must not call train_network")

    monkeypatch.setattr(trained, "train_network", boom)
    assert make_episode(4, GymConfig()).prompt() == compiled.prompt()
