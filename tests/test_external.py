"""The independent-trainer split (weightgym/external.py): scikit-learn L-BFGS networks, our tasks and faults."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("sklearn")

from weightgym import external
from weightgym.env import TRAIN_FAMILIES, TRAIN_FAULTS, VARIANTS, load_split, score_answer
from weightgym.nets import sample_task

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_external.jsonl")


def _task(family, seed, size=0):
    rng = np.random.default_rng(seed)
    task, canon = sample_task(family, rng, size)
    return task, canon, rng


# --- conversion from scikit-learn ---------------------------------------------------------


@pytest.mark.parametrize("family,out_activation", [("parity", "logistic"), ("minmax", "identity"), ("count", "softmax")])
def test_converted_network_reproduces_sklearn(family, out_activation):
    """One family per output type: our forward pass on raw inputs gives scikit-learn's own decisions."""
    task, canon, rng = _task(family, 11)
    hidden = [w + 3 for w in canon.widths[1:-1]]
    model, Xn = external.fit_sklearn(task, hidden, seed=5)
    assert model.activation == "relu" and model.solver == "lbfgs" and model.out_activation_ == out_activation
    net = external.to_mlp(model, task)
    assert net is not None and net.widths == [task.n_in] + hidden + [task.n_out]
    out = net.forward(task.X)
    if task.kind == "binary":
        assert np.array_equal((out[:, 0] > 0.5).astype(int), model.predict(Xn))
        z = (out[:, 0] - 0.5)  # a positive multiple of scikit-learn's logit
        p = model.predict_proba(Xn)[:, 1]
        assert np.array_equal(z > 0, p > 0.5) and np.isclose(np.median(np.abs(z)), 0.5)
    elif task.kind == "class":
        assert np.array_equal(np.argmax(out, axis=1), model.predict(Xn))
    else:
        assert np.allclose(out, np.asarray(model.predict(Xn)).reshape(len(Xn), -1), atol=1e-6)


def test_train_external_is_accurate_rounded_sized_and_deterministic():
    lo, hi = external.EXTRA_NEURONS
    for family in ("threshold", "arith", "argmax"):
        task, canon, rng = _task(family, 3)
        net = external.train_external(task, canon, rng)
        state = rng.bit_generator.state
        assert net is not None and task.accuracy(net) >= external.MIN_ACC
        assert all(np.array_equal(T, np.round(T, 2)) for T in net.Ws + net.bs)
        for w, c in zip(net.widths[1:-1], canon.widths[1:-1]):
            assert c + lo <= w <= c + hi
        task2, canon2, rng2 = _task(family, 3)
        assert external.train_external(task2, canon2, rng2).equals(net) and rng2.bit_generator.state == state


def test_episode_is_deterministic_and_round_trips():
    for i in (0, 1, 2):
        a, b = external.episode_for(i), external.episode_for(i)
        assert a.to_json() == b.to_json()
        assert a.seed == external.SEED0 + i and a.source == external.SOURCE
        external.check_split([a])


def test_seed_range_is_not_used_by_any_other_split():
    from weightgym.imperfect import IMPERFECT_SPLITS
    from weightgym.splits import SPLITS

    lo, hi = external.SEED0, external.SEED0 + external.N - 1
    for seed0 in [v[0] for v in SPLITS.values()] + [v[0] for v in IMPERFECT_SPLITS.values()]:
        assert seed0 + 999_999 < lo or seed0 > hi


# --- the frozen file ------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(PATH), reason="data/test_external.jsonl not generated")
def test_frozen_split():
    header, eps = load_split(PATH)
    with open(PATH) as f:
        rows = [json.loads(line) for line in f][1:]
    assert header["split"] == external.SPLIT and header["n"] == len(eps) == external.N
    assert header["config"]["source"] == external.SOURCE
    assert [ep.seed for ep in eps] == list(range(external.SEED0, external.SEED0 + external.N))
    external.check_split(eps)  # accuracy, rounding, fault margin, inverse edit scores as a success, [] on clean
    assert {ep.family for ep in eps} == set(TRAIN_FAMILIES)
    assert {ep.fault for ep in eps if not ep.is_clean} == set(TRAIN_FAULTS)
    assert {ep.variant for ep in eps} == set(VARIANTS)
    assert 0.13 <= np.mean([ep.is_clean for ep in eps]) <= 0.23  # p_clean = 0.18
    assert max(ep.net.n_weights for ep in eps) < 1000
    for ep, row in zip(eps, rows):
        assert ep.prompt_hash() == row["prompt_hash"] and ep.net.n_weights == row["n_weights"]
        if ep.variant == "weights+failures" and not ep.is_clean:
            assert ep.failures and not ep.task.correct(ep.net)[ep.failures].any()
    wrong = next(ep for ep in eps if not ep.is_clean)
    assert score_answer(wrong, "not json")["reward"] == -1.0
