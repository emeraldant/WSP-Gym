"""An independent-trainer test split: networks fitted by scikit-learn, not by this repo's own generators.

``data/test_external.jsonl`` holds 500 repair episodes whose networks come from an external library:
``sklearn.neural_network.MLPClassifier`` / ``MLPRegressor`` with ``activation="relu"`` and
``solver="lbfgs"``. Compared with our own trained source (``weightgym/trained.py``: He-normal
initialisation, minibatch Adam, squared error for binary targets), scikit-learn uses Glorot-uniform
initialisation, full-batch L-BFGS, an L2 penalty, and log-loss for both binary and multi-class
targets. The *tasks* are still this repo's function families (``env.TRAIN_FAMILIES``, sizes 0 and 1)
and the *faults* are still planted by ``faults.plant_fault`` with ``env.TRAIN_FAULTS``, so this split
tests transfer to another trainer's weights, not to another benchmark's tasks or faults.

Conversion to our ``MLP`` convention (``h = relu(W @ h + b)``, linear last layer):

  * scikit-learn stores ``coefs_[l]`` as (fan_in, fan_out); ours is the transpose.
  * Inputs are standardised for the fit; the affine map is folded back into the first layer, so the
    stored network takes the raw inputs.
  * Binary tasks: ``MLPClassifier`` has one output unit with a logistic activation and predicts class 1
    when the logit ``z > 0``. We store the linear output ``y = 0.5 + s * z`` with
    ``s = 0.5 / median(|z|)``, which has the same decision under our rule ``y > 0.5`` and puts typical
    outputs near 0 and 1.
  * Class tasks: softmax output; we keep the logits (same argmax).
  * Regression tasks: identity output; nothing to change.

Before rounding, every converted network must reproduce scikit-learn's own decisions on all of the
task's scoring inputs, otherwise the fit is discarded. The network is then rounded to
``nets.DECIMALS`` and must be at least ``MIN_ACC`` accurate, as for every other source.

Episode ``i`` is ``make_external_episode(SEED0 + i, config_for(i))``. Generation is deterministic for a
given scikit-learn / SciPy / BLAS build (fits run single-threaded); the file on disk is the frozen,
canonical split.

    python -m weightgym.external --write       # write data/test_external.jsonl and check it
    python -m weightgym.external --baselines   # non-LLM baselines on the frozen file
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import Counter

import numpy as np

from .env import TRAIN_FAMILIES, TRAIN_FAULTS, Episode, GymConfig, load_split, save_split, score_answer
from .faults import applicable_faults, plant_fault
from .nets import MLP, Task, sample_task

SPLIT = "test_external"
PATH = os.path.join("data", f"{SPLIT}.jsonl")
SEED0 = 50_000_000  # seeds 50,000,000 .. 50,000,499; no other split uses this range
N = 500
SIZES = (0, 1)
SOURCE = "sklearn_lbfgs"
P_CLEAN = 0.18
SALT = 104_729  # second word of the per-episode seed sequence (env.make_episode uses 7919)

EXTRA_NEURONS = (2, 4)  # hidden neurons added to each compiled width (inclusive)
MIN_ACC = 0.99  # required accuracy of the *rounded* network on task.X
RESTARTS = 5  # fresh initialisations tried before the task is resampled
ALPHA = 0.01  # scikit-learn's L2 penalty; keeps weight magnitudes ordinary
MAX_ITER = 2000  # L-BFGS iteration cap
BASELINES = ("noop", "random", "outlier", "search", "gradient")


# ----------------------------------------------------------------------------
# scikit-learn fit and conversion
# ----------------------------------------------------------------------------


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (X - mu) / sd, mu, sd


def fit_sklearn(task: Task, hidden: list[int], seed: int):
    """One scikit-learn L-BFGS fit on ``task``. Returns ``(model, Xn)``: the fitted estimator and the
    standardised inputs it was fitted on."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    Xn, _, _ = _standardise(task.X)
    kw = dict(hidden_layer_sizes=tuple(hidden), activation="relu", solver="lbfgs", alpha=ALPHA,
              max_iter=MAX_ITER, random_state=int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if task.kind == "reg":
            model = MLPRegressor(**kw).fit(Xn, task.Y if task.n_out > 1 else task.Y[:, 0])
        else:
            model = MLPClassifier(**kw).fit(Xn, np.asarray(task.Y).astype(int))
    return model, Xn


def to_mlp(model, task: Task) -> MLP | None:
    """Convert a fitted scikit-learn MLP into our convention (see the module docstring). Returns the
    unrounded network, or ``None`` if it does not reproduce scikit-learn's decisions on ``task.X``."""
    Xn, mu, sd = _standardise(task.X)
    Ws = [np.array(c.T, dtype=float) for c in model.coefs_]
    bs = [np.array(b, dtype=float) for b in model.intercepts_]
    bs[0] = bs[0] - Ws[0] @ (mu / sd)  # W @ ((x - mu) / sd) + b == (W / sd) @ x + (b - W @ (mu / sd))
    Ws[0] = Ws[0] / sd
    out_act = model.out_activation_
    if task.kind == "reg":
        if out_act != "identity":
            return None
        net = MLP(Ws, bs)
        ref = np.asarray(model.predict(Xn), dtype=float).reshape(len(Xn), -1)
        ours = net.forward(task.X)
        same = np.allclose(ours, ref, atol=1e-6) and np.array_equal(
            np.all(np.abs(ours - task.Y) <= task.tol, axis=1), np.all(np.abs(ref - task.Y) <= task.tol, axis=1))
        return net if same else None
    classes = np.asarray(model.classes_)
    ref = np.asarray(model.predict(Xn)).astype(int)
    if task.kind == "binary":
        if out_act != "logistic" or Ws[-1].shape[0] != 1 or not np.array_equal(classes, [0, 1]):
            return None
        z = MLP(Ws, bs).forward(task.X)[:, 0]
        s = 0.5 / max(float(np.median(np.abs(z))), 1e-9)
        Ws[-1], bs[-1] = Ws[-1] * s, bs[-1] * s + 0.5
        net = MLP(Ws, bs)
        ours = (net.forward(task.X)[:, 0] > 0.5).astype(int)
    else:
        if not np.array_equal(classes, np.arange(task.n_out)):
            return None
        if out_act == "logistic" and Ws[-1].shape[0] == 1:  # two classes: one logit z -> outputs (-z/2, z/2)
            Ws[-1], bs[-1] = np.vstack([-0.5 * Ws[-1], 0.5 * Ws[-1]]), np.concatenate([-0.5 * bs[-1], 0.5 * bs[-1]])
        elif out_act != "softmax":
            return None
        net = MLP(Ws, bs)
        ours = np.argmax(net.forward(task.X), axis=1)
    return net if np.array_equal(ours, ref) else None


def train_external(task: Task, canon: MLP, rng: np.random.Generator, extra: tuple[int, int] = EXTRA_NEURONS) -> MLP | None:
    """Fit a ReLU MLP with ``canon``'s depth and slightly larger hidden widths using scikit-learn.

    Returns the converted network rounded to ``nets.DECIMALS`` if it is at least ``MIN_ACC`` accurate
    on ``task.X``; otherwise ``None`` (the caller resamples the task). ``rng`` is advanced by a fixed
    number of draws (one per hidden layer plus one) however many restarts are needed."""
    lo, hi = extra
    hidden = [w + int(rng.integers(lo, hi + 1)) for w in canon.widths[1:-1]]
    seed = int(rng.integers(2**31 - RESTARTS))
    for attempt in range(RESTARTS):
        model, _ = fit_sklearn(task, hidden, seed + attempt)
        net = to_mlp(model, task)
        if net is None:
            continue
        net.round_()
        if task.accuracy(net) >= MIN_ACC:
            return net
    return None


# ----------------------------------------------------------------------------
# Episodes and the split
# ----------------------------------------------------------------------------


def config_for(i: int) -> GymConfig:
    return GymConfig(families=TRAIN_FAMILIES, faults=TRAIN_FAULTS, size=SIZES[i % len(SIZES)], p_clean=P_CLEAN, source=SOURCE)


def make_external_episode(seed: int, cfg: GymConfig, max_tries: int = 200) -> Episode:
    """Same protocol as ``env.make_episode``, with the clean network fitted by scikit-learn."""
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng([int(seed), SALT])
    family = str(rng.choice(cfg.families))
    variant = str(rng.choice(cfg.variants))
    want_clean = bool(rng.random() < cfg.p_clean)
    with threadpool_limits(limits=1):  # single-threaded BLAS: reproducible and light on a shared machine
        for _ in range(max_tries):
            task, canon = sample_task(family, rng, cfg.size)
            clean = train_external(task, canon, rng)
            if clean is None:
                continue
            acc_clean = task.accuracy(clean)
            if want_clean:
                return Episode(seed, family, variant, None, task, clean, clean, acc_clean, acc_clean, [], {}, [], SOURCE)
            kind = str(rng.choice(applicable_faults(clean, cfg.faults)))
            planted = plant_fault(clean, task, kind, rng, cfg.min_drop)
            if planted is None:
                continue
            faulty, inverse, info, acc_faulty = planted
            failures: list[int] = []
            if variant == "weights+failures":
                wrong = np.flatnonzero(~task.correct(faulty))
                pick = rng.choice(wrong, size=min(cfg.max_failures, len(wrong)), replace=False)
                failures = sorted(int(i) for i in pick)
            return Episode(seed, family, variant, kind, task, clean, faulty, acc_clean, acc_faulty, inverse, info, failures, SOURCE)
    raise RuntimeError(f"could not build an external episode for seed {seed} ({family})")


def episode_for(i: int) -> Episode:
    assert 0 <= i < 1_000_000
    return make_external_episode(SEED0 + i, config_for(i))


def check_split(episodes: list[Episode]) -> None:
    """Every invariant the split promises; raises AssertionError on the first violation."""
    for ep in episodes:
        assert ep.source == SOURCE and ep.family in TRAIN_FAMILIES
        assert all(np.array_equal(T, np.round(T, 2)) for T in ep.net.Ws + ep.net.bs + ep.clean.Ws + ep.clean.bs), ep.id
        assert ep.acc_clean >= MIN_ACC and ep.acc_clean == ep.task.accuracy(ep.clean), ep.id
        if ep.is_clean:
            assert ep.net.equals(ep.clean) and score_answer(ep, "[]")["success"], ep.id
        else:
            assert ep.fault in TRAIN_FAULTS and ep.acc_clean - ep.acc_faulty >= 0.15, ep.id
            assert ep.acc_faulty == ep.task.accuracy(ep.net), ep.id
            assert score_answer(ep, json.dumps(ep.inverse_edit))["success"], ep.id
            assert not score_answer(ep, "[]")["success"], ep.id


def describe(episodes: list[Episode]) -> dict:
    nw = np.array([ep.net.n_weights for ep in episodes])
    faulty = [ep for ep in episodes if not ep.is_clean]
    return {
        "n": len(episodes), "n_clean": len(episodes) - len(faulty), "n_faulty": len(faulty),
        "weights": {"min": int(nw.min()), "q25": int(np.quantile(nw, 0.25)), "median": int(np.median(nw)),
                    "q75": int(np.quantile(nw, 0.75)), "max": int(nw.max())},
        "depth": dict(sorted(Counter(ep.net.n_layers for ep in episodes).items())),
        "family": dict(sorted(Counter(ep.family for ep in episodes).items())),
        "fault": dict(sorted(Counter(ep.fault for ep in faulty).items())),
        "variant": dict(sorted(Counter(ep.variant for ep in episodes).items())),
        "acc_clean_min": float(min(ep.acc_clean for ep in episodes)),
        "acc_faulty_mean": float(np.mean([ep.acc_faulty for ep in faulty])),
    }


def write_split(path: str = PATH, n: int = N) -> list[Episode]:
    episodes = [episode_for(i) for i in range(n)]
    save_split(path, episodes, config_for(0), SPLIT)
    _, loaded = load_split(path)  # check what is on disk, not what is in memory
    assert len(loaded) == n and [ep.seed for ep in loaded] == list(range(SEED0, SEED0 + n))
    check_split(loaded)
    return loaded


def run_baselines(episodes: list[Episode], names: tuple[str, ...] = BASELINES) -> dict[str, dict]:
    from .eval import run_policy, summarize

    return {name: summarize(run_policy(episodes, name)) for name in names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"generate the split and write it to --path (default {PATH})")
    ap.add_argument("--baselines", action="store_true", help="run the non-LLM baselines on the split at --path")
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--n", type=int, default=N)
    a = ap.parse_args()
    if a.write:
        print(json.dumps(describe(write_split(a.path, a.n))))
    if a.baselines:
        _, eps = load_split(a.path)
        for name, summary in run_baselines(eps).items():
            print(name, json.dumps(summary))
