"""Real-data version of the improve task: networks trained on public tabular datasets (Part A) and a
case study on small public MLPs from the Hugging Face Hub (Part B).

Part A. Each network is a small ReLU MLP (hidden widths 4-10, one hidden layer mostly, at most 150
weights) trained with ``trained._fit`` on a real classification dataset for deliberately few steps, so
its accuracy lies between 55% and 90% and at least 5 points below a fully trained network of the same
size. Nothing is planted. The proposer reads the weights and a few labelled rows and replies with an
edit in the usual DSL; the edit is checked with forward passes and kept only if it helps.

Rows. Every dataset is cut once, with a fixed seed, into three pools: ``fit`` (40%), ``select`` (30%)
and ``report`` (30%). A network uses the whole pools, or a seeded sample of at most 800 / 600 / 600
rows of them when the dataset is larger, so a row never changes role between networks:

    fit      trains the network
    select   everything the proposer and the verifier may see (prompt examples, reward, accept test)
    report   held out; only ``verify_and_report`` reads it, for the numbers that are reported

Datasets. ``data/realworld_datasets.json`` is the frozen list (name, OpenML id and version, licence,
selected features, classes, standardisation constants, checksum). Features are the numeric columns
(the 8 with the highest mutual information with the label when there are more, measured on the fit
pool), standardised with fit-pool statistics and rounded to 1 decimal. The list is sorted
alphabetically and every third dataset is held out: 16 training datasets, 8 held-out test datasets.
Raw data is never stored in the repository; it is re-fetched into ``cache/realworld/`` and checked
against the recorded checksum.

    python -m weightgym.realworld datasets     # fetch candidates, freeze the list (run once)
    python -m weightgym.realworld splits       # write data/real_{val,test_seen,test_unseen}.jsonl
    python -m weightgym.realworld baselines    # random-edit and gradient baselines on report rows
    python -m weightgym.realworld hf           # Part B: data/real_hf_models.jsonl
    python -m weightgym.realworld stats        # sizes, accuracies and prompt lengths per split
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import zlib
from dataclasses import dataclass

import numpy as np

from .edits import MAX_OPS, EditError, apply_edit, parse_edit, validate_edit
from .env import DIRECT_SUFFIX, OPS_HELP, _architecture, _vec, fmt, render_weights
from .nets import MLP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache", "realworld")
REGISTRY = os.path.join(DATA_DIR, "realworld_datasets.json")

SEED0 = 60_000_000  # every seed of this module is 60,000,000 or above; no other split uses that range
SPLITS = {  # name: (first seed, number of frozen networks, dataset role)
    "real_train": (60_000_000, 0, "train"),  # stream: real_train_item(i)
    "real_val": (61_000_000, 200, "train"),
    "real_test_seen": (62_000_000, 300, "train"),
    "real_test_unseen": (63_000_000, 300, "heldout"),
}
HF_SEED0 = 64_000_000
SALT = 15_485_863  # second word of the per-network seed sequence

N_DATASETS, HELDOUT_EVERY = 24, 3
MAX_FEATURES, MAX_CLASSES = 8, 4
MIN_ROWS, MAX_ROWS = 1000, 20_000  # rows after filtering; larger datasets are subsampled once with a fixed seed
MAX_MISSING = 0.5  # numeric columns with more missing values than this are dropped before rows are
MIN_REFERENCE = 0.65  # a dataset is used only if a fully trained 8-wide network reaches this on select
POOL_FRACTIONS = (0.4, 0.3)  # fit, select; the rest is report
ROWS_CAP = {"fit": 800, "select": 600, "report": 600}

ACC_RANGE = (0.55, 0.90)  # accuracy of the under-trained network on its select rows
MIN_GAP = 0.05  # ... and at least this far below a fully trained network of the same widths
MIN_SECOND_CLASS = 0.05  # the network must predict at least two classes (no constant predictors)
STEP_RANGE = (5, 300)  # Adam steps of the under-trained fit, log-uniform
FULL_STEPS = 1500  # Adam steps of the fully trained reference
HIDDEN = (4, 10)
P_TWO_LAYERS = 0.2
MAX_WEIGHTS = 150
N_WRONG, N_RIGHT = 8, 4
IMPROVED, HARMED = 0.02, -0.02
STANDARDISED = "Every input is standardised (zero mean, unit variance over the dataset) and rounded to 1 decimal."

# Candidates in priority order: (name, source, id, what is predicted). The first N_DATASETS that pass the
# checks in ``build_registry`` (rows, licence, reference accuracy) are used.
CANDIDATES = [
    ("banknote-authentication", "openml", 1462, "Whether a banknote is genuine or forged (two classes), from wavelet statistics of a photograph of the note."),
    ("phoneme", "openml", 1489, "Whether a speech sound is a nasal or an oral vowel (two classes), from the amplitudes of its first five harmonics."),
    ("MagicTelescope", "openml", 1120, "Whether a shower image from the MAGIC Cherenkov telescope was caused by a gamma ray (g) or a hadron (h)."),
    ("electricity", "openml", 151, "Whether the electricity price in New South Wales went UP or DOWN relative to its average over the last 24 hours."),
    ("spambase", "openml", 44, "Whether an e-mail is spam (1) or not (0), from word and character frequencies."),
    ("segment", "openml", 40984, "Which kind of surface an outdoor image region shows, from colour and edge statistics of the region."),
    ("pendigits", "openml", 32, "Which digit was written, from points sampled along the pen trajectory on a tablet."),
    ("satimage", "openml", 182, "The soil type at the centre of a 3x3 patch of a Landsat satellite image, from its spectral values."),
    ("steel-plates-fault", "openml", 1504, "The fault class of a steel plate (two classes), from geometric and luminosity measurements of the fault."),
    ("qsar-biodeg", "openml", 1494, "Whether a chemical is readily biodegradable (two classes), from molecular descriptors."),
    ("adult", "openml", 1590, "Whether a person's income is above 50K dollars a year (1994 US census), from the numeric census fields."),
    ("letter", "openml", 6, "Which capital letter an image shows, from statistics of its pixels."),
    ("texture", "openml", 40499, "Which texture an image patch shows, from statistical moments of the patch."),
    ("titanic", "openml", 40945, "Whether a passenger of the Titanic survived (1) or not (0), from the numeric passenger fields."),
    ("mozilla4", "openml", 1046, "The defect state of a Mozilla source-code unit (two classes), from its size and change history."),
    ("wall-robot-navigation", "openml", 1497, "Which move a wall-following robot makes next, from its ultrasound sensor readings."),
    ("JapaneseVowels", "openml", 375, "Which speaker uttered a Japanese vowel, from cepstrum coefficients of one frame of the recording."),
    ("shuttle", "openml", 40685, "The operating state of the Space Shuttle's radiator subsystem, from sensor readings."),
    ("credit-default", "openml", 42477, "Whether a credit-card client in Taiwan defaults on next month's payment (1) or not (0)."),
    ("optdigits", "openml", 28, "Which handwritten digit an 8x8 image shows, from the pixel counts of its cells."),
    ("gas-drift", "openml", 1476, "Which gas a chemical sensor array is exposed to, from features of the sensor responses."),
    ("covertype", "sklearn", "covtype", "The forest cover type of a 30 m x 30 m cell of Roosevelt National Forest, from cartographic variables."),
    ("skin-segmentation", "openml", 1502, "Whether a pixel shows skin (two classes), from its blue, green and red values."),
    ("eeg-eye-state", "openml", 1471, "Whether a person's eyes are open or closed (two classes), from EEG electrode readings."),
    ("yeast", "openml", 181, "The cellular localisation site of a yeast protein, from sequence-derived scores."),
    ("wine-quality-white", "openml", 40498, "The quality score a taster gave a white wine, from its chemical measurements."),
    ("higgs", "openml", 23512, "Whether a simulated particle collision is a Higgs-boson signal or background (two classes)."),
    ("page-blocks", "openml", 30, "The kind of block (text, line, picture, ...) a region of a scanned page is, from its shape and pixel statistics."),
    ("jungle-chess", "openml", 41027, "The game-theoretic outcome of a two-piece Jungle Chess endgame for the side to move, from piece strengths and positions."),
    ("nomao", "openml", 1486, "Whether two listings of a place refer to the same place (two classes), from similarity scores of their fields."),
]
# Numeric columns that identify a row rather than measure anything; never used as features.
EXCLUDED_COLUMNS = {"mozilla4": ("id",), "JapaneseVowels": ("utterance", "frame")}
OK_LICENCES = ("public", "cc0", "cc by", "cc-by", "public domain", "unlimited reuse")


# ----------------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------------


@dataclass
class _Rows:
    """The parts of ``nets.Task`` that ``trained._fit`` and ``baselines.gradient_repair`` read."""

    X: np.ndarray
    Y: np.ndarray
    kind: str  # "binary": one output, class 1 when y > 0.5 | "class": argmax
    n_out: int


@dataclass
class RealData:
    name: str
    desc: str
    features: list[str]
    classes: list[str]
    X: np.ndarray  # (N, n_features), in the network's input convention
    y: np.ndarray  # (N,) class index
    pools: dict  # "fit" / "select" / "report" -> sorted row indices
    input_note: str = STANDARDISED
    licence: str = ""
    two_logits: bool = False  # a two-class network with one output per class (public models only)

    @property
    def kind(self) -> str:
        return "binary" if len(self.classes) == 2 and not self.two_logits else "class"

    @property
    def n_out(self) -> int:
        return 1 if self.kind == "binary" else len(self.classes)


def predict(net: MLP, X: np.ndarray, kind: str) -> np.ndarray:
    out = net.forward(X)
    return (out[:, 0] > 0.5).astype(int) if kind == "binary" else np.argmax(out, axis=1)


def _checksum(X: np.ndarray, y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype="<f8").tobytes() + np.ascontiguousarray(y, dtype="<i8").tobytes()).hexdigest()[:16]


def _pools(n: int, salt: int) -> dict:
    perm = np.random.default_rng([SEED0, 2, salt]).permutation(n)
    a, b = int(round(POOL_FRACTIONS[0] * n)), int(round((POOL_FRACTIONS[0] + POOL_FRACTIONS[1]) * n))
    return {"fit": np.sort(perm[:a]), "select": np.sort(perm[a:b]), "report": np.sort(perm[b:])}


def _fetch_raw(source: str, did):
    """(feature frame, label series, metadata) straight from the public source."""
    if source == "openml":
        from sklearn.datasets import fetch_openml

        # parser="liac-arff": scikit-learn's pandas ARFF parser fails on integer-named categories with pandas 3.
        b = fetch_openml(data_id=int(did), as_frame=True, parser="liac-arff", data_home=os.path.join(CACHE, "openml"))
        d = b.details
        return b.data, b.target, {"source": "openml", "openml_id": int(did), "openml_name": d.get("name"), "version": int(d.get("version")),
                                  "licence": str(d.get("licence")), "url": f"https://www.openml.org/d/{int(did)}",
                                  "target": d.get("default_target_attribute")}
    if source == "sklearn" and did == "covtype":
        from sklearn.datasets import fetch_covtype

        b = fetch_covtype(data_home=os.path.join(CACHE, "sklearn"), as_frame=True)
        return b.data, b.target, {"source": "sklearn.datasets.fetch_covtype", "openml_id": None, "openml_name": None, "version": None,
                                  "licence": "UCI Covertype: unlimited reuse with retention of the copyright notice (J. A. Blackard, Colorado State University)",
                                  "url": "https://archive.ics.uci.edu/dataset/31/covertype", "target": "Cover_Type"}
    raise ValueError(f"unknown source {source!r}")


def _prepare(cand: tuple, frozen: dict | None = None) -> dict:
    """Fetch and preprocess one dataset. With ``frozen`` (its registry entry) the recorded features, classes
    and standardisation constants are applied instead of being recomputed."""
    import pandas as pd

    name, source, did, desc = cand
    salt = zlib.crc32(name.encode())
    frame, target, meta = _fetch_raw(source, did)
    num = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    num = [c for c in num if frame[c].isna().mean() <= MAX_MISSING and c not in EXCLUDED_COLUMNS.get(name, ())]
    keep = (~frame[num].isna().any(axis=1) & ~target.isna()).to_numpy()
    X = frame.loc[keep, num].to_numpy(dtype=float)
    labels = target[keep].astype(str).to_numpy()
    vals, counts = np.unique(labels, return_counts=True)
    if frozen:
        classes = list(frozen["classes"])
    else:  # the most frequent classes (ties: label order), then listed in label order
        top = sorted(range(len(vals)), key=lambda i: (-counts[i], str(vals[i])))[:MAX_CLASSES]
        classes = sorted(str(vals[i]) for i in top)
    index = {c: i for i, c in enumerate(classes)}
    m = np.array([lab in index for lab in labels])
    X, y = X[m], np.array([index[lab] for lab in labels[m]], dtype=int)
    n_filtered = len(X)
    if len(X) > MAX_ROWS:
        sub = np.sort(np.random.default_rng([SEED0, 1, salt]).choice(len(X), size=MAX_ROWS, replace=False))
        X, y = X[sub], y[sub]
    pools = _pools(len(X), salt)
    fit = pools["fit"]
    if frozen:
        cols = [num.index(f) for f in frozen["features"]]
        mean, std = np.array(frozen["mean"]), np.array(frozen["std"])
    else:
        live = np.flatnonzero(X[fit].std(0) > 1e-9)
        if len(live) > MAX_FEATURES:
            from sklearn.feature_selection import mutual_info_classif

            rows = np.random.default_rng([SEED0, 4, salt]).choice(fit, size=min(4000, len(fit)), replace=False)
            mi = mutual_info_classif(X[rows][:, live], y[rows], random_state=0)
            live = np.sort(live[np.argsort(-mi, kind="stable")[:MAX_FEATURES]])
        cols = [int(c) for c in live]
        mean, std = X[fit][:, cols].mean(0), X[fit][:, cols].std(0)
    Z = np.round((X[:, cols] - mean) / std, 1) + 0.0
    share = np.bincount(y, minlength=len(classes)) / len(y)
    entry = {
        "name": name, **meta, "predicts": desc, "rows_raw": int(len(frame)), "rows_after_filter": int(n_filtered), "rows": int(len(Z)),
        "numeric_features_raw": len(num), "features": [str(num[c]) for c in cols], "classes_raw": int(len(vals)), "classes": classes,
        "class_share": [round(float(s), 4) for s in share], "mean": [float(v) for v in mean], "std": [float(v) for v in std],
        "pool_rows": {k: int(len(v)) for k, v in pools.items()}, "checksum": _checksum(Z, y),
    }
    return {"entry": entry, "X": Z, "y": y}


def _reference_accuracy(X: np.ndarray, y: np.ndarray, pools: dict, n_classes: int) -> float:
    """Select-pool accuracy of a fully trained network with 8 hidden neurons (dataset feasibility check)."""
    from .trained import _fit

    kind, n_out = ("binary", 1) if n_classes == 2 else ("class", n_classes)
    rng = np.random.default_rng([SEED0, 5])  # the pools are sorted and some datasets are ordered by class: sample, do not slice
    fit, sel = (rng.choice(pools[k], size=min(ROWS_CAP[k], len(pools[k])), replace=False) for k in ("fit", "select"))
    net = _fit(_Rows(X[fit], y[fit].astype(float) if kind == "binary" else y[fit], kind, n_out), [X.shape[1], 8, n_out],
               np.random.default_rng([SEED0, 3]), steps=FULL_STEPS)
    return float(np.mean(predict(net, X[sel], kind) == y[sel]))


def _try_candidate(cand: tuple) -> dict:
    try:
        out = _prepare(cand)
    except Exception as e:  # noqa: BLE001 - a candidate that cannot be fetched is reported as rejected
        return {"name": cand[0], "error": f"{type(e).__name__}: {str(e)[:160]}"}
    e = out["entry"]
    pools = _pools(len(out["y"]), zlib.crc32(cand[0].encode()))
    e["reference_accuracy"] = round(_reference_accuracy(out["X"], out["y"], pools, len(e["classes"])), 4)
    return out


def build_registry(procs: int = 4) -> None:
    """Fetch the candidates, apply the checks, freeze the first ``N_DATASETS`` and write the registry."""
    from multiprocessing import Pool

    os.makedirs(CACHE, exist_ok=True)
    with Pool(procs) as pool:
        jobs = [pool.apply_async(_try_candidate, (c,)) for c in CANDIDATES]
        results = []
        for cand, job in zip(CANDIDATES, jobs):
            try:
                results.append(job.get(timeout=420))
            except Exception as e:  # noqa: BLE001 - e.g. a download that does not finish
                results.append({"name": cand[0], "error": f"{type(e).__name__}: {str(e)[:120]}"})
    chosen, rejected = [], []
    for cand, out in zip(CANDIDATES, results):
        if "error" in out:
            rejected.append({"name": cand[0], "reason": out["error"]})
            continue
        e = out["entry"]
        reason = None
        if e["rows"] < MIN_ROWS:
            reason = f"{e['rows']} rows after dropping missing values and keeping {MAX_CLASSES} classes (< {MIN_ROWS})"
        elif not any(k in e["licence"].lower() for k in OK_LICENCES):
            reason = f"licence {e['licence']!r} not recognised as permissive"
        elif e["reference_accuracy"] < MIN_REFERENCE:
            reason = f"a fully trained 8-wide network reaches only {e['reference_accuracy']:.2f} on select (< {MIN_REFERENCE})"
        elif len(chosen) >= N_DATASETS:
            reason = f"not needed: {N_DATASETS} datasets with higher priority passed"
        if reason:
            rejected.append({"name": cand[0], "reason": reason})
        else:
            chosen.append(out)
    assert len(chosen) == N_DATASETS, f"only {len(chosen)} datasets passed"
    chosen.sort(key=lambda o: o["entry"]["name"].lower())
    for i, out in enumerate(chosen):
        out["entry"]["role"] = "heldout" if i % HELDOUT_EVERY == HELDOUT_EVERY - 1 else "train"
        np.savez_compressed(os.path.join(CACHE, out["entry"]["name"] + ".npz"), X=out["X"], y=out["y"])
    entries = [o["entry"] for o in chosen]
    reg = {
        "note": "Frozen dataset list of weightgym/realworld.py. Sorted alphabetically (case-insensitive); every third dataset is held out. "
                "A proposer trained on real-data networks must never see a held-out dataset.",
        "preprocessing": f"numeric columns only (columns with more than {MAX_MISSING:.0%} missing values dropped, then rows with missing values); "
                         f"the {MAX_CLASSES} most frequent classes when there are more; at most {MAX_ROWS} rows (fixed-seed sample); rows cut 40/30/30 "
                         f"into fit/select/report pools with a fixed seed; the {MAX_FEATURES} numeric features with the highest mutual information "
                         "with the label on the fit pool when there are more; standardised with fit-pool mean and std, rounded to 1 decimal.",
        "train": [e["name"] for e in entries if e["role"] == "train"],
        "heldout": [e["name"] for e in entries if e["role"] == "heldout"],
        "datasets": entries,
        "rejected_candidates": rejected,
    }
    with open(REGISTRY, "w") as f:
        json.dump(reg, f, indent=1)
    for e in entries:
        print(f"{e['role']:8s} {e['name']:26s} id={e['openml_id']} v={e['version']} licence={e['licence'][:12]!r} rows={e['rows']} "
              f"features={len(e['features'])}/{e['numeric_features_raw']} classes={len(e['classes'])}/{e['classes_raw']} "
              f"majority={max(e['class_share']):.2f} reference={e['reference_accuracy']:.2f}")
    for r in rejected:
        print("rejected", r["name"], "-", r["reason"])


@functools.lru_cache(maxsize=1)
def registry() -> dict:
    with open(REGISTRY) as f:
        return json.load(f)


def dataset_names(role: str) -> tuple[str, ...]:
    return tuple(registry()[role])


def cache_ready() -> bool:
    """True when every registry dataset is in the local cache (no network needed)."""
    return os.path.exists(REGISTRY) and all(os.path.exists(os.path.join(CACHE, e["name"] + ".npz")) for e in registry()["datasets"])


@functools.lru_cache(maxsize=None)
def load_dataset(name: str) -> RealData:
    if name in HF_DATASETS:
        return HF_DATASETS[name]()
    entry = next(e for e in registry()["datasets"] if e["name"] == name)
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        with np.load(path) as z:
            X, y = z["X"], z["y"]
    else:  # re-fetch from the public source with the frozen features, classes and constants
        out = _prepare(next(c for c in CANDIDATES if c[0] == name), frozen=entry)
        X, y = out["X"], out["y"]
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, X=X, y=y)
    if _checksum(X, y) != entry["checksum"]:
        raise RuntimeError(f"{name}: data does not match the checksum in {REGISTRY}; delete {path} and fetch again")
    return RealData(name, entry["predicts"], entry["features"], entry["classes"], X, y, _pools(len(y), zlib.crc32(name.encode())),
                    STANDARDISED, entry["licence"])


# ----------------------------------------------------------------------------------------------
# Items
# ----------------------------------------------------------------------------------------------


def _f1(v: float) -> str:
    s = f"{float(v):.1f}"
    s = s[:-2] if s.endswith(".0") else s
    return "0" if s == "-0" else s


def _render_weights(net: MLP, decimals: int) -> str:
    if decimals == 2:
        return render_weights(net)
    # Same layout as env.render_weights with more decimals (full-precision public models only).
    def f(v):
        s = f"{float(v):.{decimals}f}".rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s

    names = ["x"] + [f"h{l}" for l in range(1, net.n_layers)] + ["y"]
    lines = []
    for l, (W, b) in enumerate(zip(net.Ws, net.bs), start=1):
        lines.append(f"W{l} ({W.shape[0]} rows x {W.shape[1]} cols) and b{l}. "
                     f"Row i = incoming weights of {names[l]}[i]; column j = outgoing weights of {names[l - 1]}[j].")
        for i in range(W.shape[0]):
            lines.append(f"  W{l}[{i}] = [" + ", ".join(f(x) for x in W[i]) + f"]   b{l}[{i}] = {f(b[i])}")
    return "\n".join(lines)


class RealItem:
    """One network on one real dataset. Duck-types what ``llm.evaluate`` and ``eval.record`` use."""

    is_clean = False
    fault = "imperfect"
    variant = "weights+failures"

    def __init__(self, seed: int, dataset: str, net: MLP, rows: dict, examples: dict, meta: dict | None = None,
                 source: str = "real_sgd", decimals: int = 2):
        self.seed, self.family, self.dataset, self.net, self.source, self.decimals = int(seed), dataset, dataset, net, source, decimals
        self.rows = {k: np.asarray(v, dtype=int) for k, v in rows.items()}
        self.examples = {k: [int(i) for i in v] for k, v in examples.items()}  # dataset row indices, all in select
        self.meta = dict(meta or {})
        self.data = load_dataset(dataset)
        self.task = _Rows(self.data.X, self.data.y, self.data.kind, self.data.n_out)  # for baselines.gradient_repair
        self.acc_select, self.acc_report = self.accuracy(net, "select"), self.accuracy(net, "report")

    def accuracy(self, net: MLP, part: str) -> float:
        idx = self.rows[part]
        return float(np.mean(predict(net, self.data.X[idx], self.data.kind) == self.data.y[idx]))

    # --- prompt -------------------------------------------------------------------------------

    def _example(self, i: int) -> str:
        d = self.data
        out = self.net.forward(d.X[i][None, :])[0]
        xs = "[" + ", ".join(_f1(v) for v in d.X[i]) + "]"
        if d.kind == "binary":
            return f"x = {xs}: expected {int(d.y[i])}, network gives y = {fmt(out[0])}"
        return f"x = {xs}: expected answer {int(d.y[i])}, network gives y = {_vec(out)} (answer {int(np.argmax(out))})"

    def _specification(self) -> str:
        d = self.data
        feats = ", ".join(f"x{i} = {n}" for i, n in enumerate(d.features))
        if d.kind == "binary":
            outs = (f"1 output y. The network predicts class 1 ({d.classes[1]!r}) when y > 0.5 and class 0 ({d.classes[0]!r}) otherwise; "
                    "it counts as correct when that is the row's class.")
        else:
            names = ", ".join(f"{i} = {c!r}" for i, c in enumerate(d.classes))
            outs = (f"{len(d.classes)} outputs y[0]..y[{len(d.classes) - 1}], one per class ({names}). The network's answer is the index of "
                    "the largest output.")
        return (f"Task: {d.desc}\nInputs: {len(d.features)} numeric features, in this order: {feats}. {d.input_note}\nOutputs: {outs}\n"
                f"Current accuracy: {100 * self.acc_select:.1f}% of the {len(self.rows['select'])} labelled rows that proposals are tested on.")

    def prompt(self, direct: bool = False) -> str:
        if self.source == "real_sgd":
            origin = "The network was trained by gradient descent on the dataset described below but only partly learned it"
        else:
            origin = "The network is a public model that was trained on the dataset described below and is not perfect"
        s = [
            f"You are improving a small ReLU network by editing its weights. {origin}: it is currently right on about "
            f"{100 * self.acc_select:.0f}% of inputs. Nothing was deliberately broken, so there may be no single culprit.",
            f"## Specification\n{self._specification()}",
            f"## Architecture\n{_architecture(self.net)}",
            f"## Weights\n{_render_weights(self.net, self.decimals)}",
        ]
        if self.examples.get("wrong"):
            s.append("## Failing inputs\nSome of the inputs it gets wrong:\n" + "\n".join(self._example(i) for i in self.examples["wrong"]))
        if self.examples.get("right"):
            s.append("## Correct inputs\nSome of the inputs it gets right:\n" + "\n".join(self._example(i) for i in self.examples["right"]))
        s.append(
            "## Your task\n"
            f"Reply with a JSON list of at most {MAX_OPS} edit operations that makes the network more accurate on the "
            "specification. Each operation changes one row, one column or one scalar. Indices are 0-based. "
            "Allowed operations (examples):\n"
            f"{OPS_HELP}\n"
            "Your edit is a proposal: it will be tested with forward passes and kept only if it makes the network "
            "more accurate, so propose the change you think is most likely to help. It is scored by the share of the "
            "network's errors it removes (zero if it does not help); an invalid or over-budget answer is penalised. "
            "End your reply with the JSON list and nothing after it."
            + (DIRECT_SUFFIX if direct else "")
        )
        return "\n\n".join(s)

    def prompt_hash(self, direct: bool = False) -> str:
        return hashlib.sha256(self.prompt(direct).encode()).hexdigest()[:16]

    # --- reward and verification ----------------------------------------------------------------

    def score(self, text: str) -> dict:
        """Reward of one proposal, measured on the select rows only. Keys as ``ImperfectItem.score``."""
        res = {"reward": -1.0, "valid": False, "success": False, "recovery": None, "acc_after": None, "n_ops": None, "noop": False,
               "error": None, "ops": None, "gain": None}
        try:
            ops = parse_edit(text, self.net)
        except EditError as e:
            res["error"] = str(e)
            return res
        acc = self.accuracy(apply_edit(self.net, ops), "select")
        gain = acc - self.acc_select
        gap = gain / (1 - self.acc_select) if self.acc_select < 1 else min(0.0, gain)
        res.update(valid=True, ops=ops, n_ops=len(ops), noop=len(ops) == 0, acc_after=acc, gain=gain, recovery=gap,
                   reward=float(max(0.0, gap)), success=bool(gap >= 0.95))
        return res

    def verify_and_report(self, proposals: list[list[dict]]) -> dict:
        """Keep the proposal with the best accuracy on select, if it beats the unedited network there; return the
        accuracy change and the gap closed on the held-out report rows. Invalid proposals are skipped."""
        best_net, best_sel = self.net, self.acc_select
        for ops in proposals:
            try:
                cand = apply_edit(self.net, validate_edit(ops, self.net))
            except EditError:
                continue
            a = self.accuracy(cand, "select")
            if a > best_sel + 1e-9:
                best_net, best_sel = cand, a
        return self._report(best_net, best_sel)

    def _report(self, net: MLP, acc_sel: float) -> dict:
        gain = self.accuracy(net, "report") - self.acc_report
        return {"gain": gain, "gap": gain / (1 - self.acc_report) if self.acc_report < 1 else min(0.0, gain),
                "accepted": net is not self.net, "select_gain": acc_sel - self.acc_select}

    # --- storage --------------------------------------------------------------------------------

    def to_json(self) -> dict:
        return {"seed": self.seed, "dataset": self.dataset, "source": self.source, "decimals": self.decimals, "net": self.net.to_json(),
                "n_weights": self.net.n_weights, "widths": self.net.widths, "acc_select": self.acc_select, "acc_report": self.acc_report,
                "meta": self.meta, "examples": self.examples, "prompt_hash": self.prompt_hash(),
                "rows": {k: v.tolist() for k, v in self.rows.items()}}

    @staticmethod
    def from_json(d: dict) -> "RealItem":
        return RealItem(d["seed"], d["dataset"], MLP.from_json(d["net"]), d["rows"], d["examples"], d.get("meta"), d.get("source", "real_sgd"),
                        d.get("decimals", 2))


def _pick_examples(item_rows: np.ndarray, correct: np.ndarray, rng: np.random.Generator) -> dict:
    out = {}
    for key, mask, n in (("wrong", ~correct, N_WRONG), ("right", correct, N_RIGHT)):
        pool = item_rows[mask]
        out[key] = sorted(int(i) for i in rng.choice(pool, size=min(n, len(pool)), replace=False))
    return out


def make_real_item(seed: int, datasets: tuple[str, ...], max_tries: int = 400) -> RealItem:
    """An under-trained network on one of ``datasets``. Deterministic in ``seed`` (and the dataset list)."""
    from .trained import _fit

    rng = np.random.default_rng([int(seed), SALT])
    name = str(datasets[int(rng.integers(len(datasets)))])
    d = load_dataset(name)
    rows = {k: np.sort(rng.choice(d.pools[k], size=min(cap, len(d.pools[k])), replace=False)) for k, cap in ROWS_CAP.items()}
    yfit = d.y[rows["fit"]]
    fit = _Rows(d.X[rows["fit"]], yfit.astype(float) if d.kind == "binary" else yfit, d.kind, d.n_out)
    Xs, ys = d.X[rows["select"]], d.y[rows["select"]]
    ref_seed, full = int(rng.integers(1 << 62)), {}
    for _ in range(max_tries):
        hidden = [int(rng.integers(HIDDEN[0], HIDDEN[1] + 1))]
        if rng.random() < P_TWO_LAYERS:
            hidden.append(int(rng.integers(HIDDEN[0], HIDDEN[1] + 1)))
        steps = int(round(np.exp(rng.uniform(np.log(STEP_RANGE[0]), np.log(STEP_RANGE[1])))))
        net_seed = int(rng.integers(1 << 62))
        widths = [d.X.shape[1]] + hidden + [d.n_out]
        if sum((a + 1) * b for a, b in zip(widths[:-1], widths[1:])) > MAX_WEIGHTS:
            continue
        net = _fit(fit, widths, np.random.default_rng(net_seed), steps=steps).round_()
        pred = predict(net, Xs, d.kind)
        acc = float(np.mean(pred == ys))
        if not ACC_RANGE[0] <= acc <= ACC_RANGE[1]:
            continue
        if np.sort(np.bincount(pred, minlength=len(d.classes)))[-2] < MIN_SECOND_CLASS * len(pred):
            continue
        key = tuple(hidden)
        if key not in full:  # fully trained network of the same widths (same fit rows), accuracy on select
            full[key] = float(np.mean(predict(_fit(fit, widths, np.random.default_rng([ref_seed, *key]), steps=FULL_STEPS), Xs, d.kind) == ys))
        if acc > full[key] - MIN_GAP:
            continue
        ex = _pick_examples(rows["select"], pred == ys, rng)
        return RealItem(seed, name, net, rows, ex, {"steps": steps, "acc_full": full[key]})
    raise RuntimeError(f"could not train an under-trained network for seed {seed} ({name})")


def real_item_for(split: str, i: int) -> RealItem:
    seed0, _, role = SPLITS[split]
    return make_real_item(seed0 + i, dataset_names(role))


def real_train_item(i: int) -> RealItem:
    """The training stream: network ``i`` on one of the 16 training datasets (never a held-out one)."""
    return real_item_for("real_train", i)


def _gen_line(args: tuple) -> str:
    return json.dumps(real_item_for(*args).to_json())


def write_splits(names=("real_val", "real_test_seen", "real_test_unseen"), procs: int = 4) -> None:
    from multiprocessing import Pool

    for name in names:
        seed0, n, role = SPLITS[name]
        with Pool(procs) as pool:
            lines = pool.map(_gen_line, [(name, i) for i in range(n)], chunksize=4)
        with open(os.path.join(DATA_DIR, f"{name}.jsonl"), "w") as f:
            f.write(json.dumps({"split": name, "n": n, "seed0": seed0, "datasets": list(dataset_names(role)), "role": role}) + "\n")
            f.writelines(line + "\n" for line in lines)
        print(f"wrote {name}: {n} networks")


def read_split(name: str) -> tuple[dict, list[dict]]:
    """Header and raw records of a frozen split (no data needed)."""
    with open(os.path.join(DATA_DIR, f"{name}.jsonl")) as f:
        return json.loads(f.readline()), [json.loads(line) for line in f if line.strip()]


def load_real_split(name: str, n: int | None = None) -> list[RealItem]:
    recs = read_split(name)[1]
    return [RealItem.from_json(r) for r in (recs[:n] if n else recs)]


# ----------------------------------------------------------------------------------------------
# Propose-and-verify statistics and baselines (all numbers are on the report rows)
# ----------------------------------------------------------------------------------------------


def loop_stats(rows: list[dict], n_boot: int = 10000) -> dict:
    """Mean gap closed with a 95% bootstrap interval over networks; share improved / harmed by 2 points."""
    gap = np.clip(np.array([r["gap"] for r in rows], dtype=float), -1, 1)
    boot = gap[np.random.default_rng(0).integers(0, len(gap), size=(n_boot, len(gap)))].mean(1)
    gain = np.array([r["gain"] for r in rows], dtype=float)
    return {"n": len(rows), "gap_closed": float(gap.mean()), "gap_lo": float(np.quantile(boot, 0.025)), "gap_hi": float(np.quantile(boot, 0.975)),
            "gain": float(gain.mean()), "improved": float(np.mean(gain >= IMPROVED - 1e-12)), "harmed": float(np.mean(gain <= HARMED + 1e-12)),
            "accepted": float(np.mean([r["accepted"] for r in rows]))}


def loop_from_records(items: list[RealItem], records: list[dict], k: int) -> dict:
    """Propose-and-verify with the first ``k`` sampled answers per network from an ``llm.evaluate`` log."""
    by: dict[int, list] = {}
    for r in sorted(records, key=lambda r: r.get("sample", 0)):
        if r.get("ops") is not None and r.get("sample", 0) < k:
            by.setdefault(r["seed"], []).append(r["ops"])
    return loop_stats([it.verify_and_report(by.get(it.seed, [])) for it in items])


def gradient_reference(it: RealItem, steps: int = 300, lr: float = 0.02) -> dict:
    """Upper reference: Adam on the select rows from the shown network (no edit budget), kept if it helps there."""
    from .baselines import gradient_repair

    net = gradient_repair(it, steps=steps, lr=lr, idx=it.rows["select"])
    acc = it.accuracy(net, "select")
    return it._report(net, acc) if acc > it.acc_select + 1e-9 else it._report(it.net, it.acc_select)


def baselines(items: list[RealItem], ks=(1, 8, 32, 128)) -> dict:
    from .baselines import random_policy

    out = {}
    for k in ks:
        rng = np.random.default_rng(0)
        out[f"random x{k}"] = loop_stats([it.verify_and_report([json.loads(random_policy(it, rng)) for _ in range(k)]) for it in items])
    out["gradient on select (300 Adam steps)"] = loop_stats([gradient_reference(it) for it in items])
    return out


def run_baselines(splits=("real_val", "real_test_seen", "real_test_unseen"), out: str = "results/table_realworld_baselines.csv") -> None:
    from .eval import write_csv

    table = []
    for split in splits:
        for name, st in baselines(load_real_split(split)).items():
            table.append({"split": split, "proposer": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()}})
            print(f"{split:17s} {name:36s} gap closed {st['gap_closed']:+.3f} [{st['gap_lo']:+.3f}, {st['gap_hi']:+.3f}]  improved {st['improved']:.3f}  "
                  f"harmed {st['harmed']:.3f}  accepted {st['accepted']:.2f}")
    write_csv(os.path.join(ROOT, out), table)


def split_stats(splits=("real_val", "real_test_seen", "real_test_unseen")) -> None:
    for split in splits:
        items = load_real_split(split)
        nw = [it.net.n_weights for it in items]
        acc = [it.acc_select for it in items]
        accr = [it.acc_report for it in items]
        full = [it.meta["acc_full"] for it in items]
        plen = [len(it.prompt(True)) for it in items]
        two = np.mean([it.net.n_layers == 3 for it in items])
        print(f"{split}: n={len(items)} datasets={len({it.dataset for it in items})} weights {min(nw)}-{max(nw)} (median {int(np.median(nw))}) "
              f"two hidden layers {two:.2f}; accuracy on select {min(acc):.3f}-{max(acc):.3f} (mean {np.mean(acc):.3f}), on report "
              f"{min(accr):.3f}-{max(accr):.3f} (mean {np.mean(accr):.3f}); fully trained mean {np.mean(full):.3f}; prompt chars median "
              f"{int(np.median(plen))} max {max(plen)}")


# ----------------------------------------------------------------------------------------------
# Part B: small public MLPs from the Hugging Face Hub (case study)
#
# Only safetensors files are read (never pickles). A model qualifies when its card, config or published
# source code states a plain Linear+ReLU stack and the input preprocessing, and our forward pass then
# reproduces a sensible accuracy on the matching public dataset. Input standardisation is folded into the
# first layer, so the stored network takes raw feature values. The public weights are otherwise used as
# they are: rounded to 2 decimals only if accuracy on the dataset moves by at most 1 point, else kept at
# full precision and printed with 3 decimals. These networks were trained by their authors, so there is
# no fit pool: rows are cut 50/50 into select and report with a fixed seed.
# ----------------------------------------------------------------------------------------------

HF_PATH = os.path.join(DATA_DIR, "real_hf_models.jsonl")
HF_CACHE = os.path.join(CACHE, "hf")
IRIS_FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
IRIS_CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def _hf_file(repo: str, filename: str, repo_type: str = "model") -> str:
    from huggingface_hub import hf_hub_download

    d = os.path.join(HF_CACHE, ("dataset__" if repo_type == "dataset" else "") + repo.replace("/", "__"))
    path = os.path.join(d, filename)
    if not os.path.exists(path):
        os.makedirs(d, exist_ok=True)
        hf_hub_download(repo, filename, repo_type=repo_type, local_dir=d)
    return path


def _hf_tensors(repo: str) -> dict:
    from safetensors.numpy import load_file

    return {k: np.asarray(v, dtype=float) for k, v in load_file(_hf_file(repo, "model.safetensors")).items()}


def _hf_pools(n: int, salt: int) -> dict:
    perm = np.random.default_rng([HF_SEED0, salt]).permutation(n)
    return {"fit": np.array([], dtype=int), "select": np.sort(perm[: n // 2]), "report": np.sort(perm[n // 2 :])}


def _iris_table() -> tuple[np.ndarray, np.ndarray]:
    import csv

    with open(_hf_file("scikit-learn/iris", "Iris.csv", "dataset")) as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r[c]) for c in IRIS_FEATURES] for r in rows]), np.array([IRIS_CLASSES.index(r["Species"]) for r in rows])


def _hf_iris() -> RealData:
    X, y = _iris_table()
    return RealData("hf-iris", "The species of an iris flower, from the length and width of its sepals and petals.", IRIS_FEATURES, IRIS_CLASSES,
                    X, y, _hf_pools(len(y), 1), "Inputs are raw measurements in centimetres, given to 1 decimal.",
                    "CC0-1.0 (Hugging Face dataset scikit-learn/iris)")


TITANIC_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Title", "FamilySize", "IsAlone", "Embarked_C", "Embarked_Q", "Embarked_S"]


def _titanic_table() -> tuple[np.ndarray, np.ndarray]:
    """The 12 inputs of bernardozschaber/titanic-federated-dp-mlp, rebuilt from OpenML 'titanic' (id 40945, all
    1,309 passengers) with the constants of the model's own preprocessing.json. Unscaled values."""
    from sklearn.datasets import fetch_openml

    p = json.load(open(_hf_file("bernardozschaber/titanic-federated-dp-mlp", "preprocessing.json")))
    assert p["features"] == TITANIC_FEATURES
    b = fetch_openml(data_id=40945, as_frame=True, parser="liac-arff", data_home=os.path.join(CACHE, "openml"))
    df = b.data
    num = lambda c: df[c].astype(float).to_numpy()  # noqa: E731
    age, fare = num("age"), num("fare")
    sibsp, parch = num("sibsp"), num("parch")
    title = np.array([p["titles"].get((re.search(r" ([A-Za-z]+)\.", str(n)) or [None, ""])[1], 4) for n in df["name"]], dtype=float)
    emb = np.array([p["embarked_mode"] if e is None or e != e or str(e) in ("nan", "None") else str(e) for e in df["embarked"].astype(object)])
    fam = sibsp + parch + 1
    cols = {"Pclass": num("pclass"), "Sex": (df["sex"].astype(str).to_numpy() == "female").astype(float),
            "Age": np.where(np.isnan(age), p["age_median"], age), "SibSp": sibsp, "Parch": parch,
            "Fare": np.where(np.isnan(fare), p["fare_median"], fare), "Title": title, "FamilySize": fam, "IsAlone": (fam == 1).astype(float),
            "Embarked_C": (emb == "C").astype(float), "Embarked_Q": (emb == "Q").astype(float), "Embarked_S": (emb == "S").astype(float)}
    return np.stack([cols[f] for f in TITANIC_FEATURES], axis=1), b.target.astype(int).to_numpy()


def _hf_titanic() -> RealData:
    X, y = _titanic_table()
    return RealData("hf-titanic", "Whether a passenger of the Titanic survived (1) or not (0).", TITANIC_FEATURES, ["0", "1"],
                    np.round(X, 1) + 0.0, y, _hf_pools(len(y), 2),
                    "Inputs are unscaled values rounded to 1 decimal: Sex is 1 for female; Title is 0 Mr, 1 Miss, 2 Mrs, 3 Master, 4 other; "
                    "FamilySize = SibSp + Parch + 1; missing Age is 28.5 and missing Fare 14.5.",
                    "Public (OpenML titanic, id 40945)", two_logits=True)


HF_DATASETS = {"hf-iris": _hf_iris, "hf-titanic": _hf_titanic}


def _fold_scaling(net: MLP, mean: np.ndarray, std: np.ndarray) -> MLP:
    """W @ ((x - mean) / std) + b == (W / std) @ x + (b - W @ (mean / std))."""
    net = net.copy()
    net.bs[0] = net.bs[0] - net.Ws[0] @ (mean / std)
    net.Ws[0] = net.Ws[0] / std
    return net


def _stack(t: dict, names: list[str]) -> MLP:
    return MLP([t[f"{n}.weight"] for n in names], [t[f"{n}.bias"] for n in names])


def _iris_mlp(repo: str, scaling: str) -> tuple[MLP, dict]:
    """``scaling``: "card" = the statistics of the card's own training split, "all" = z-scores over all 150 rows
    (a guess, for a model that documents nothing), "raw" = no scaling."""
    raw = _stack(_hf_tensors(repo), ["fc1", "fc2"])
    X, y = _iris_table()
    z = _fold_scaling(raw, X.mean(0), X.std(0))
    extra = {"accuracy_raw_inputs": float(np.mean(np.argmax(raw.forward(X), 1) == y)),
             "accuracy_z_scored_inputs": float(np.mean(np.argmax(z.forward(X), 1) == y))}
    if scaling == "card":  # two stratified train_test_split calls with random_state=42, as in the card's code
        from sklearn.model_selection import train_test_split

        Xtf, Xte, ytf, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
        Xtr, _, _, _ = train_test_split(Xtf, ytf, test_size=0.1, stratify=ytf, random_state=42)
        net = _fold_scaling(raw, Xtr.mean(0), Xtr.std(0))
        extra["accuracy_on_card_test_split"] = float(np.mean(np.argmax(net.forward(Xte), 1) == yte))
        return net, extra
    return (z if scaling == "all" else raw), extra


def _titanic_mlp() -> tuple[MLP, dict]:
    repo = "bernardozschaber/titanic-federated-dp-mlp"
    p = json.load(open(_hf_file(repo, "preprocessing.json")))
    net = _fold_scaling(_stack(_hf_tensors(repo), ["net.0", "net.2", "net.4"]), np.array(p["scaler"]["mean"]), np.array(p["scaler"]["std"]))
    X, y = _titanic_table()
    return net, {"accuracy_unrounded_inputs": float(np.mean(np.argmax(net.forward(X), 1) == y))}


def _california_rmse() -> float:
    """Regression model: not usable for an item, but check that our forward pass reproduces the card's test RMSE."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    h = fetch_california_housing(data_home=os.path.join(CACHE, "sklearn"))
    Xtf, Xte, ytf, yte = train_test_split(h.data, h.target, test_size=0.25, random_state=42)
    Xtr, _, _, _ = train_test_split(Xtf, ytf, test_size=0.25, random_state=42)
    net = _fold_scaling(_stack(_hf_tensors("sadhaklal/mlp-california-housing"), ["fc1", "fc2", "fc3", "fc4"]), Xtr.mean(0), Xtr.std(0))
    return float(np.sqrt(np.mean((net.forward(Xte)[:, 0] - yte) ** 2)))


HF_MODELS = [  # usable candidates: repo, licence, dataset, builder, what the repository documents
    {"repo": "sadhaklal/mlp-iris", "licence": "none stated", "dataset": "hf-iris", "build": lambda: _iris_mlp("sadhaklal/mlp-iris", "card"),
     "activation": "ReLU (model card code)", "preprocessing": "standardised with training-split mean and std (model card); folded into W1, b1",
     "card_claim": "accuracy 0.9333 on its 15-row test split"},
    {"repo": "leoschneider/mlp-iris", "licence": "none stated", "dataset": "hf-iris", "build": lambda: _iris_mlp("leoschneider/mlp-iris", "all"),
     "activation": "INFERRED, not documented: the card is an empty template. ReLU assumed (tensor names fc1/fc2 as in the author's "
                   "futhark-mlp-iris, whose source states ReLU)",
     "preprocessing": "INFERRED, not documented: inputs z-scored with the mean and std of all 150 rows (raw inputs give a constant "
                      "prediction); folded into W1, b1",
     "card_claim": "none"},
    {"repo": "leoschneider/futhark-mlp-iris", "licence": "none stated", "dataset": "hf-iris",
     "build": lambda: _iris_mlp("leoschneider/futhark-mlp-iris", "raw"), "activation": "ReLU, identity output (model.fut in the repository)",
     "preprocessing": "none in model.fut; raw inputs", "card_claim": "none (no model card)"},
    {"repo": "bernardozschaber/titanic-federated-dp-mlp", "licence": "MIT", "dataset": "hf-titanic", "build": _titanic_mlp,
     "activation": "ReLU (inference.py)", "preprocessing": "12 engineered features standardised with preprocessing.json; folded into W1, b1",
     "card_claim": "accuracy 0.760 on 179 held-out passengers of the Kaggle train.csv (card says 986 parameters; the tensors hold 978)"},
]
HF_REJECTED = [
    {"repo": "sadhaklal/mlp-california-housing", "reason": "regression (predicts house value), and 5,601 weights; items are classification only"},
    {"repo": "jamieoliver/titanic-2410", "reason": "12-20-20-1 safetensors, MIT, but the card documents neither the activation nor the 12 input "
                                                   "features and their preprocessing (it only links a GitHub repository)"},
    {"repo": "pppereira3/NYC_SQF_ARR_MLP", "reason": "only a .pkl file (scikit-learn pickle); never unpickled"},
]
MIN_HF_ACCURACY = 0.6  # a reconstruction that scores below this on its own dataset is treated as not reproduced


def build_hf_items() -> None:
    lines, k = [], 0
    for spec in HF_MODELS:
        base = {"repo": spec["repo"], "url": f"https://huggingface.co/{spec['repo']}", "licence": spec["licence"], "activation": spec["activation"],
                "preprocessing": spec["preprocessing"], "card_claim": spec["card_claim"]}
        net, extra = spec["build"]()
        d = load_dataset(spec["dataset"])
        acc = lambda n: float(np.mean(predict(n, d.X, d.kind) == d.y))  # noqa: E731
        acc_full, acc_round = acc(net), acc(net.copy().round_())
        base.update(architecture="-".join(str(w) for w in net.widths), n_weights=net.n_weights, dataset=spec["dataset"], rows=int(len(d.y)),
                    accuracy_all_rows=acc_full, accuracy_all_rows_rounded_2dp=acc_round, **extra)
        if acc_full < MIN_HF_ACCURACY:
            lines.append({**base, "status": "rejected", "reason": f"forward pass with the documented pipeline gives accuracy {acc_full:.3f} on "
                          "its dataset (a constant prediction on raw inputs): no working pipeline is documented, not reproduced"})
            continue
        keep_round = abs(acc_round - acc_full) <= 0.01
        stored = net.copy().round_() if keep_round else net
        sel = d.pools["select"]
        rng = np.random.default_rng([HF_SEED0 + k, SALT])
        ex = _pick_examples(sel, predict(stored, d.X[sel], d.kind) == d.y[sel], rng)
        item = RealItem(HF_SEED0 + k, spec["dataset"], stored, d.pools, ex, {"repo": spec["repo"]}, source="hf:" + spec["repo"],
                        decimals=2 if keep_round else 3)
        k += 1
        lines.append({**base, "status": "usable", "stored_weights": "rounded to 2 decimals" if keep_round else "full precision, printed with 3 decimals",
                      "accuracy_select": item.acc_select, "accuracy_report": item.acc_report, "prompt_chars": len(item.prompt(True)),
                      "item": item.to_json()})
    for r in HF_REJECTED:
        extra = {}
        if r["repo"] == "sadhaklal/mlp-california-housing":
            try:
                extra = {"card_claim": "RMSE 0.5502 on its test split", "rmse_on_card_test_split": _california_rmse()}
            except Exception as e:  # noqa: BLE001 - the check is informative only
                extra = {"check_failed": f"{type(e).__name__}: {str(e)[:120]}"}
        lines.append({"repo": r["repo"], "url": f"https://huggingface.co/{r['repo']}", "status": "rejected", "reason": r["reason"], **extra})
    with open(HF_PATH, "w") as f:
        f.writelines(json.dumps(line) + "\n" for line in lines)
    for line in lines:
        print(json.dumps({k: v for k, v in line.items() if k != "item"}))


def load_hf_items() -> list[RealItem]:
    with open(HF_PATH) as f:
        return [RealItem.from_json(r["item"]) for r in map(json.loads, f) if r.get("status") == "usable"]


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "datasets":
        build_registry()
    elif cmd == "splits":
        write_splits()
    elif cmd == "baselines":
        run_baselines()
    elif cmd == "stats":
        split_stats()
    elif cmd == "hf":
        build_hf_items()
    elif cmd == "hf-baselines":
        for it in load_hf_items():
            st = baselines([it])
            print(it.source, f"select {it.acc_select:.3f} report {it.acc_report:.3f}", {k: round(v["gain"], 4) for k, v in st.items()})
    else:
        raise SystemExit(f"unknown command {cmd!r}")
