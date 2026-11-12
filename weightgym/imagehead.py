"""Real image classifiers from the Hugging Face Hub: is the final layer improvable by a small weight edit?

Whole networks are far too large to print, so only the **final classification layer** (the "head") of
each public model is treated as the editable network. The backbone is run once on CPU and its outputs,
the inputs of the final linear layer, are cached. The head is then an ordinary one-layer ``MLP``
(``Ws=[W]``, ``bs=[b]``, shape classes x features) and the usual edit DSL applies (at most 3 ops).

    select rows   images from the dataset's *training* split: proposals are verified here
    report rows   the dataset's official *test* split: the only place a change is reported

Caveat that matters when reading any number from this module: the published models were trained on the
training split, so ``select`` accuracy is usually far above test accuracy (the backbone has seen those
images). Where it is at or near 100% there is nothing left to select on.

    PYTHONPATH=. uv run python -m weightgym.imagehead extract     # download models + data, cache features
    PYTHONPATH=. uv run python -m weightgym.imagehead records     # write data/imagehead_models.jsonl
    PYTHONPATH=. uv run python -m weightgym.imagehead baselines   # non-LLM baselines -> results/
    PYTHONPATH=. uv run python -m weightgym.imagehead prompts     # prompt lengths

No LLM is called from this module. Weights are loaded from safetensors only.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib

import numpy as np

from .edits import MAX_OPS, EditError, apply_edit, parse_edit, validate_edit
from .env import DIRECT_SUFFIX, OPS_HELP, REPAIR_THRESHOLD, _architecture
from .nets import MLP

CACHE = os.path.join("cache", "imagehead")
RECORDS = os.path.join("data", "imagehead_models.jsonl")
BASELINE_TABLE = os.path.join("results", "table_imagehead_baselines.csv")
PROMPT_CAP = 40_000  # characters; above this only N_SHOWN_COLS columns of W1 are printed
N_SHOWN_COLS = 64
N_CONFUSIONS = 6
SEED0 = 60_000_000  # item seeds are SEED0 + crc32(hub id) % 1,000,000; no other split uses this range
DATA_SEED = 20261112  # fixes which images are subsampled, per dataset and split (same images for every model)
RANDOM_KS = (8, 32, 128)
FINETUNE_STEPS, FINETUNE_LR = 300, 1e-3

DATASETS = {
    "beans": dict(repo="AI-Lab-Makerere/beans", image="image", label="labels", select="train", report="test",
                  what="photographs of bean leaves (beans dataset)"),
    "cifar10": dict(repo="uoft-cs/cifar10", image="img", label="label", select="train", report="test",
                    what="CIFAR-10 images (32x32 colour photographs)"),
    "mnist": dict(repo="ylecun/mnist", image="image", label="label", select="train", report="test",
                  what="MNIST images (28x28 handwritten digits)"),
    "fashion_mnist": dict(repo="zalando-datasets/fashion_mnist", image="image", label="label", select="train", report="test",
                          what="Fashion-MNIST images (28x28 greyscale pictures of clothing)"),
}

# Chosen from Hub API searches on 2026-11-12 (image-classification models tagged with or named after these
# datasets). ``claimed`` is the accuracy the model card states and ``claimed_source`` says where and on what.
# n_select / n_report cap the number of images (None = the whole split); the caps exist for CPU time only.
MODELS = [
    dict(id="voloshka3/beans-deit-tiny-lab", dataset="beans", licence=None, claimed=0.9297,
         claimed_source="model card text: 'Test accuracy: 0.9297' (test split)", n_select=None, n_report=None),
    dict(id="Konradowsky/beans-deit-tiny", dataset="beans", licence="apache-2.0", claimed=0.9453,
         claimed_source="model card results table: test accuracy 0.9453 (test split, 128 images)", n_select=None, n_report=None),
    dict(id="Bocianelllo/beans-vit-tiny", dataset="beans", licence=None, claimed=0.9688,
         claimed_source="model card text: 'Test accuracy: 0.9688' (test split)", n_select=None, n_report=None),
    dict(id="Zharasbek/resnet18-bean-disease-classifier", dataset="beans", licence="apache-2.0", claimed=0.9453,
         claimed_source="model card table: Accuracy 0.9453 (test split)", n_select=None, n_report=None),
    dict(id="omaralshareef/beans-image-classifier", dataset="beans", licence="apache-2.0", claimed=0.8984,
         claimed_source="model-index metadata: 'Test Accuracy' 0.8984 (validation accuracy 0.9549 is also stated)", n_select=None, n_report=None),
    dict(id="pfpdostudent/deit-tiny-cifar10-lab", dataset="cifar10", licence=None, claimed=0.9435,
         claimed_source="model card text: 'Test accuracy: 0.9435'", n_select=5000, n_report=5000),
    dict(id="TestingProfile111/cifar10-vit-tiny", dataset="cifar10", licence=None, claimed=0.9613,
         claimed_source="model card text: 'Results (test split) accuracy: 0.9613'", n_select=5000, n_report=5000),
    dict(id="fancifulcrow/deit-tiny-patch16-224-finetuned-cifar10", dataset="cifar10", licence="apache-2.0", claimed=0.9262,
         claimed_source="auto-generated Trainer card: Accuracy 0.9262 on 'the evaluation set' (split not stated)", n_select=5000, n_report=5000),
    dict(id="abidlabs/mnist-vit-tiny-1min", dataset="mnist", licence="apache-2.0", claimed=0.9947,
         claimed_source="auto-generated Trainer card: Accuracy 0.9947 on 'the evaluation set' (split not stated)", n_select=5000, n_report=5000),
    dict(id="mmabas77/fashion-mnist-mobilenetv2", dataset="fashion_mnist", licence="other", claimed=0.9422,
         processor="google/mobilenet_v2_1.0_224",  # the repo has no preprocessor_config.json; this is its stated base model
         claimed_source="auto-generated Trainer card: Accuracy 0.9422 on 'the evaluation set' (split not stated)", n_select=5000, n_report=5000),
    dict(id="SamAdamDay/resnet18_cifar10", dataset="cifar10", licence="mit", claimed=0.9473, loader="timm",
         claimed_source="model card text and model-index: 'test accuracy of 94.73'", n_select=5000, n_report=5000),
    dict(id="dennishauser/mnist_basic", dataset="mnist", licence="apache-2.0", claimed=0.8809,
         claimed_source="model-index metadata and Trainer card: Accuracy 0.8809 on 'the evaluation set' (split not stated)", n_select=800, n_report=1200),
    dict(id="corralesvr/my_test_numbers_model", dataset="mnist", licence="apache-2.0", claimed=0.25,
         claimed_source="model-index metadata and Trainer card: Accuracy 0.25 on 'the evaluation set' (split not stated)", n_select=800, n_report=1200),
]


def _slug(hub_id: str) -> str:
    return hub_id.replace("/", "__")


def seed_for(hub_id: str) -> int:
    return SEED0 + zlib.crc32(hub_id.encode()) % 1_000_000


def _paths(hub_id: str) -> tuple[str, str]:
    base = os.path.join(CACHE, _slug(hub_id))
    return base + ".npz", base + ".json"


# ----------------------------------------------------------------------------------------------
# Feature extraction (the only part that needs torch, transformers/timm, datasets and the network)
# ----------------------------------------------------------------------------------------------


def _subset(dataset: str, kind: str, n: int, cap: int | None) -> np.ndarray:
    if cap is None or n <= cap:
        return np.arange(n)
    rng = np.random.default_rng([DATA_SEED, zlib.crc32(f"{dataset}/{kind}".encode())])
    return np.sort(rng.permutation(n)[:cap])


def _norm_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def extract(spec: dict, batch: int = 50, force: bool = False) -> None:
    """Run the backbone once and cache the final layer, its inputs and the labels for both row sets."""
    import time

    import datasets
    import torch
    from huggingface_hub import HfApi

    npz_path, meta_path = _paths(spec["id"])
    if os.path.exists(npz_path) and os.path.exists(meta_path) and not force:
        return
    os.makedirs(CACHE, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    ds = DATASETS[spec["dataset"]]
    sha = HfApi().model_info(spec["id"]).sha
    if spec.get("loader") == "timm":
        import timm

        model = timm.create_model(f"hf_hub:{spec['id']}@{sha}", pretrained=True)  # timm reads model.safetensors
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
        prep = lambda imgs: torch.stack([tf(im) for im in imgs])  # noqa: E731
        model_names, arch = None, f"timm {model.pretrained_cfg.get('architecture', type(model).__name__)}"
    else:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(spec["id"], revision=sha, use_safetensors=True)
        proc = AutoImageProcessor.from_pretrained(spec["processor"]) if spec.get("processor") else AutoImageProcessor.from_pretrained(spec["id"], revision=sha)
        prep = lambda imgs: proc(images=imgs, return_tensors="pt")["pixel_values"]  # noqa: E731
        model_names = [str(model.config.id2label[i]) for i in range(model.config.num_labels)]
        arch = type(model).__name__ + (f" ({model.config.architecture})" if getattr(model.config, "architecture", None) else "")
    model.eval()
    head = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    seen: dict = {}
    head.register_forward_hook(lambda m, i, o: seen.update(x=i[0].detach(), y=o.detach()))
    out: dict = {"W": head.weight.detach().numpy().astype(np.float64), "b": head.bias.detach().numpy().astype(np.float64)}
    meta = dict(id=spec["id"], sha=sha, dataset=spec["dataset"], dataset_repo=ds["repo"], architecture=arch,
                n_params=int(sum(p.numel() for p in model.parameters())), torch=torch.__version__)
    hook_gap = 0.0
    for kind in ("select", "report"):
        d = datasets.load_dataset(ds["repo"], split=ds[kind])
        ds_names = [str(s) for s in d.features[ds["label"]].names]
        if model_names and {_norm_name(s) for s in model_names} == {_norm_name(s) for s in ds_names}:
            pos = {_norm_name(s): k for k, s in enumerate(model_names)}
            to_model = np.array([pos[_norm_name(s)] for s in ds_names])  # dataset label -> model output index
            class_names, label_map = model_names, "by class name"
        else:
            to_model, class_names, label_map = np.arange(len(ds_names)), ds_names, "identity (model has no usable class names)"
        idx = _subset(spec["dataset"], kind, len(d), spec.get(f"n_{kind}"))
        sub = d.select(idx)
        F, P, t0 = [], [], time.time()
        for s in range(0, len(sub), batch):
            rows = sub[s : s + batch]
            with torch.inference_mode():
                res = model(prep([im.convert("RGB") for im in rows[ds["image"]]]))
            logits = res.logits if hasattr(res, "logits") else res
            hook_gap = max(hook_gap, float((logits - seen["y"]).abs().max()))
            F.append(seen["x"].reshape(len(logits), -1).numpy().astype(np.float32))
            P.append(logits.argmax(1).numpy())
        out[f"F_{kind}"], out[f"pred_{kind}"] = np.concatenate(F), np.concatenate(P).astype(np.int64)
        out[f"y_{kind}"], out[f"idx_{kind}"] = to_model[np.array(sub[ds["label"]])].astype(np.int64), idx
        meta.update({f"n_{kind}": int(len(idx)), f"n_{kind}_split": int(len(d)), f"split_{kind}": ds[kind],
                     f"seconds_{kind}": round(time.time() - t0, 1)})
    meta.update(class_names=class_names, label_map=label_map, hook_max_abs_gap=hook_gap,
                preprocessing=f"image processor of {spec['processor']} (the model repo ships none)" if spec.get("processor") else "the model repo's own image processor / timm data config")
    np.savez_compressed(npz_path, **out)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[extract] {spec['id']}: head {out['W'].shape}, select {meta['n_select']} in {meta['seconds_select']}s, "
          f"report {meta['n_report']} in {meta['seconds_report']}s", flush=True)


# ----------------------------------------------------------------------------------------------
# Items
# ----------------------------------------------------------------------------------------------


def _g(v: float) -> str:
    """Three significant figures."""
    s = f"{float(v):.3g}"
    return "0" if s in ("-0", "0") else s


def _vec(v) -> str:
    return "[" + ", ".join(_g(x) for x in np.atleast_1d(v)) + "]"


def _preds(net: MLP, F: np.ndarray) -> np.ndarray:
    """Head predictions. Single precision, like the model itself (the features are float32)."""
    return np.argmax(F @ net.Ws[0].astype(np.float32).T + net.bs[0].astype(np.float32), axis=1)


def _accuracy(net: MLP, F: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(_preds(net, F) == y))


class HeadItem:
    """The final layer of one Hub model. Duck-types what ``llm.evaluate`` and ``eval.record`` use."""

    is_clean = False
    fault = "imperfect"
    variant = "weights+failures"
    source = "huggingface_hub"

    def __init__(self, rec: dict, arrays: dict):
        self.rec, self.hub_id, self.family, self.seed = rec, rec["id"], rec["dataset"], int(rec["seed"])
        self.class_names = list(rec["class_names"])
        self.net = MLP([np.array(arrays["W"], dtype=float)], [np.array(arrays["b"], dtype=float)])
        self.F_select, self.y_select = arrays["F_select"], arrays["y_select"]
        self.F_report, self.y_report = arrays["F_report"], arrays["y_report"]
        self.pred_select_model, self.pred_report_model = arrays["pred_select"], arrays["pred_report"]
        self.acc_select = _accuracy(self.net, self.F_select, self.y_select)
        self.acc_report = _accuracy(self.net, self.F_report, self.y_report)
        self.claimed = rec.get("claimed_accuracy")
        self._prompt: dict[bool, str] = {}
        self._shown: np.ndarray | None = None  # column indices printed in the prompt (None until the prompt is built)
        self._restricted = False

    # ---- prompt -----------------------------------------------------------------------------

    @property
    def shown_columns(self) -> np.ndarray:
        """True indices of the columns of W1 the prompt prints (all of them when the matrix fits)."""
        if self._shown is None:
            self.prompt()
        return self._shown

    @property
    def restricted(self) -> bool:
        if self._shown is None:
            self.prompt()
        return self._restricted

    def _body(self, cols: np.ndarray | None) -> list[str]:
        net, names = self.net, self.class_names
        W, b = net.Ws[0], net.bs[0]
        C, Fdim = W.shape
        y, pred = self.y_select, _preds(net, self.F_select)
        n = len(y)
        logits = np.sort(self.F_select @ W.astype(np.float32).T + b.astype(np.float32), axis=1)
        margin = logits[:, -1] - logits[:, -2]
        s = [
            "You are improving a real, public image classifier by editing the weights of its final layer. The model is "
            f"`{self.hub_id}` from the Hugging Face Hub ({self.rec['architecture']}), which sorts {DATASETS[self.family]['what']} "
            f"into {C} classes. Everything before the final layer (the backbone) is frozen: for each image it produces a "
            f"vector x of {Fdim} features, and those features are the inputs here. The final layer computes y = W1 @ x + b1 "
            "and the predicted class is the index of the largest y. Nothing was deliberately broken, so there may be no "
            "single culprit.",
            "## Classes\n" + ", ".join(f"{i} = {nm}" for i, nm in enumerate(names)),
            f"## Architecture\n{_architecture(net)}\nx = the backbone's {Fdim} features for one image.",
        ]
        per = []
        for c in range(C):
            m = y == c
            per.append(f"  {c} {names[c]}: {100 * float(np.mean(pred[m] == c)) if m.any() else float('nan'):.1f}% ({int(np.sum(pred[m] == c))} of {int(m.sum())})")
        conf = np.zeros((C, C), dtype=int)
        np.add.at(conf, (y, pred), 1)
        np.fill_diagonal(conf, 0)
        order = np.argsort(-conf, axis=None, kind="stable")[:N_CONFUSIONS]
        pairs = [(int(i // C), int(i % C)) for i in order if conf.flat[i] > 0]
        s.append(
            f"## Current behaviour on {n} validation images\n"
            f"Accuracy: {100 * float(np.mean(pred == y)):.1f}% ({int(np.sum(pred == y))} of {n} correct).\n"
            "Per-class accuracy:\n" + "\n".join(per) + "\n"
            "Largest confusions (true class -> predicted class: number of images):\n"
            + ("\n".join(f"  {t} {names[t]} -> {p} {names[p]}: {conf[t, p]}" for t, p in pairs) if pairs else "  none")
            + f"\nScale: the largest y beats the runner-up by a median of {_g(np.median(margin))} "
            f"(10th percentile {_g(np.quantile(margin, 0.1))}); the mean feature magnitude |x[j]| is {_g(np.mean(np.abs(self.F_select)))}."
        )
        w = [f"b1 = {_vec(b)}", f"Row norms |W1[i]| = {_vec(np.linalg.norm(W, axis=1))}"]
        if cols is None:
            w.append(f"W1 ({C} rows x {Fdim} cols). Row i = incoming weights of y[i] (class i); column j = outgoing weights of feature x[j].")
            w += [f"  W1[{i}] = {_vec(W[i])}" for i in range(C)]
        else:
            w.append(
                f"W1 has {C} rows x {Fdim} cols, too many to print. Row i = incoming weights of y[i] (class i); column j = "
                f"outgoing weights of feature x[j]. Shown below, as column:weight pairs with the true column indices, are the "
                f"{len(cols)} columns whose weights vary most across the classes. The other {Fdim - len(cols)} columns are hidden "
                "and cannot be edited by index: a column operation or a single-weight edit must use one of the column indices "
                "shown. An operation on a whole row (scale or swap) still acts on every column of that row."
            )
            w += [f"  W1[{i}] = {{" + ", ".join(f"{int(j)}: {_g(W[i, j])}" for j in cols) + "}" for i in range(C)]
        s.append("## Weights\n" + "\n".join(w))
        return s

    def prompt(self, direct: bool = False) -> str:
        if direct not in self._prompt:
            C, Fdim = self.net.Ws[0].shape
            tail = (
                "## Your task\n"
                f"Reply with a JSON list of at most {MAX_OPS} edit operations that makes the classifier more accurate. Each "
                "operation changes one row, one column or one scalar. Indices are 0-based. Only the tensors W1 and b1 exist "
                f"(an added vector needs {Fdim} numbers for a row of W1 and {C} for a column). Allowed operations (the indices "
                "and values in these examples are only illustrations):\n"
                f"{OPS_HELP.replace('W2', 'W1').replace('b2', 'b1')}\n"
                "Your edit is a proposal: it will be tested with forward passes and kept only if it makes the network "
                "more accurate, so propose the change you think is most likely to help. It is scored by the share of the "
                "network's errors it removes (zero if it does not help); an invalid or over-budget answer is penalised. "
                "End your reply with the JSON list and nothing after it."
            )
            full = "\n\n".join(self._body(None) + [tail + DIRECT_SUFFIX])  # decide on the longer (direct) variant
            if len(full) <= PROMPT_CAP:
                self._shown, self._restricted, body = np.arange(Fdim), False, self._body(None)
            else:
                var = self.net.Ws[0].var(axis=0)
                self._shown = np.sort(np.argsort(-var, kind="stable")[:N_SHOWN_COLS])
                self._restricted, body = True, self._body(self._shown)
            for d in (False, True):
                self._prompt[d] = "\n\n".join(body + [tail + (DIRECT_SUFFIX if d else "")])
        return self._prompt[direct]

    def prompt_hash(self, direct: bool = False) -> str:
        return hashlib.sha256(self.prompt(direct).encode()).hexdigest()[:16]

    # ---- edits ------------------------------------------------------------------------------

    def validate(self, ops) -> list[dict]:
        """``edits.validate_edit`` plus the rule the prompt states: hidden columns cannot be addressed by index."""
        ops = validate_edit(ops, self.net)
        if self.restricted:
            ok = set(int(j) for j in self.shown_columns)
            for op in ops:
                if op["tensor"] != "W1":
                    continue
                cols = [op["index"][1]] if op["op"] == "set" else [] if op.get("axis") != "col" else [op[k] for k in ("index", "i", "j") if k in op]
                bad = [j for j in cols if j not in ok]
                if bad:
                    raise EditError(f"column {bad[0]} of W1 is hidden and cannot be edited by index")
        return ops

    def score(self, text: str) -> dict:
        """One proposal, judged on the ``select`` rows. reward = max(0, gap closed); -1 for an invalid answer."""
        res = {"reward": -1.0, "valid": False, "success": False, "recovery": None, "acc_after": None, "n_ops": None,
               "noop": False, "error": None, "ops": None, "gain": None}
        try:
            ops = self.validate(parse_edit(text, self.net))
        except EditError as e:
            res["error"] = str(e)
            return res
        acc_after = _accuracy(apply_edit(self.net, ops), self.F_select, self.y_select) if ops else self.acc_select
        gain = acc_after - self.acc_select
        gap = gain / (1 - self.acc_select) if self.acc_select < 1 else 0.0  # share of the select errors removed
        res.update(valid=True, ops=ops, n_ops=len(ops), noop=len(ops) == 0, acc_after=acc_after, gain=gain, recovery=gap,
                   reward=float(max(0.0, min(1.0, gap))), success=bool(gap >= REPAIR_THRESHOLD))
        return res

    def verify_and_report(self, proposals: list) -> dict:
        """Keep the valid proposal with the best ``select`` accuracy if it beats the unedited head there (first one
        wins ties); report the change on the ``report`` rows (the official test split) in percentage points."""
        best_net, best_sel, best_ops, n_valid = self.net, self.acc_select, None, 0
        for ops in proposals:
            try:
                ops = self.validate(ops)
            except EditError:
                continue
            n_valid += 1
            if not ops:
                continue
            cand = apply_edit(self.net, ops)
            a = _accuracy(cand, self.F_select, self.y_select)
            if a > best_sel + 1e-9:
                best_net, best_sel, best_ops = cand, a, ops
        return self._report(best_net, best_ops, n_valid=n_valid, n_proposals=len(proposals))

    def _report(self, net: MLP, ops=None, **extra) -> dict:
        sel = _accuracy(net, self.F_select, self.y_select)
        ok0, ok1 = _preds(self.net, self.F_report) == self.y_report, _preds(net, self.F_report) == self.y_report
        after = float(np.mean(ok1))
        return {"accepted": net is not self.net, "select_before": self.acc_select, "select_after": sel,
                "report_before": self.acc_report, "report_after": after, "change_points": 100 * (after - self.acc_report),
                "report_fixed": int(np.sum(~ok0 & ok1)), "report_broken": int(np.sum(ok0 & ~ok1)),  # test images that changed
                "claimed": self.claimed, "exceeds_claimed": None if self.claimed is None else bool(after > self.claimed + 1e-9),
                "ops": ops, **extra}


# ----------------------------------------------------------------------------------------------
# Records on disk and the loader
# ----------------------------------------------------------------------------------------------


def _load_cached(hub_id: str):
    npz_path, meta_path = _paths(hub_id)
    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        return None
    with np.load(npz_path) as z, open(meta_path) as f:
        return {k: z[k] for k in z.files}, json.load(f)


def write_records(path: str = RECORDS) -> list[dict]:
    """One record per cached model, with the checks: the head reproduces the model's own predictions, and
    printing 3 significant figures would not change accuracy by more than 0.1 point."""
    recs = []
    for spec in MODELS:
        got = _load_cached(spec["id"])
        if got is None:
            print(f"[records] {spec['id']}: no cache, skipped")
            continue
        a, meta = got
        net = MLP([a["W"].astype(float)], [a["b"].astype(float)])
        agree = float(np.mean(np.concatenate([_preds(net, a["F_select"]) == a["pred_select"], _preds(net, a["F_report"]) == a["pred_report"]])))
        sel, rep = _accuracy(net, a["F_select"], a["y_select"]), _accuracy(net, a["F_report"], a["y_report"])
        r3 = MLP([np.vectorize(lambda v: float(f"{v:.3g}"))(net.Ws[0])], [np.vectorize(lambda v: float(f"{v:.3g}"))(net.bs[0])])
        rec = {
            "id": spec["id"], "sha": meta["sha"], "seed": seed_for(spec["id"]), "licence": spec["licence"], "dataset": spec["dataset"],
            "dataset_repo": meta["dataset_repo"], "architecture": meta["architecture"], "n_params": meta["n_params"],
            "loader": spec.get("loader", "transformers.AutoModelForImageClassification") + " (safetensors)",
            "n_classes": int(a["W"].shape[0]), "feature_dim": int(a["W"].shape[1]), "head_weights": int(a["W"].size + a["b"].size),
            "class_names": meta["class_names"], "label_map": meta["label_map"], "preprocessing": meta.get("preprocessing"),
            "claimed_accuracy": spec["claimed"], "claimed_source": spec["claimed_source"],
            "select_split": meta["split_select"], "n_select": meta["n_select"], "n_select_split": meta["n_select_split"],
            "report_split": meta["split_report"], "n_report": meta["n_report"], "n_report_split": meta["n_report_split"],
            "select_accuracy": sel, "test_accuracy": rep,
            "model_test_accuracy": float(np.mean(a["pred_report"] == a["y_report"])),  # the model's own argmax
            "head_agreement": agree, "hook_max_abs_gap": meta["hook_max_abs_gap"],
            "test_accuracy_3sf": _accuracy(r3, a["F_report"], a["y_report"]), "select_accuracy_3sf": _accuracy(r3, a["F_select"], a["y_select"]),
        }
        assert agree >= 0.999, (spec["id"], agree)
        # The stored head keeps full precision; this only checks that the 3 figures the prompt prints describe it faithfully.
        rec["printing_3sf_within_0.1pt"] = bool(abs(rec["test_accuracy_3sf"] - rep) <= 0.001 + 1e-12 and abs(rec["select_accuracy_3sf"] - sel) <= 0.001 + 1e-12)
        if not rec["printing_3sf_within_0.1pt"]:
            print(f"[records] {spec['id']}: rounding to 3 significant figures changes accuracy by more than 0.1 point")
        recs.append(rec)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    return recs


def load_records(path: str = RECORDS) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_items(path: str = RECORDS, strict: bool = False) -> list[HeadItem]:
    """Items for every recorded model whose features are cached (``strict`` raises when one is missing)."""
    items = []
    for rec in load_records(path):
        got = _load_cached(rec["id"])
        if got is None:
            if strict:
                raise FileNotFoundError(f"no cached features for {rec['id']}; run `python -m weightgym.imagehead extract`")
            continue
        items.append(HeadItem(rec, got[0]))
    return items


# ----------------------------------------------------------------------------------------------
# Non-LLM baselines (all verified on ``select``, all reported on the test split)
# ----------------------------------------------------------------------------------------------


def random_valid_edit(item: HeadItem, rng: np.random.Generator) -> list[dict]:
    """One draw of ``baselines.random_policy``, redrawn until it is valid for this item."""
    from .baselines import random_policy

    while True:
        try:
            return item.validate(json.loads(random_policy(item, rng)))
        except EditError:
            continue


def best_bias_edit(item: HeadItem) -> list[dict]:
    """Exhaustive search for the best single bias change on ``select``: for every class, every threshold at which
    adding a constant to that class's bias changes a prediction. Returns [] when nothing beats the head."""
    W, b = item.net.Ws[0], item.net.bs[0]
    L = (item.F_select @ W.astype(np.float32).T + b.astype(np.float32)).astype(float)
    y, n = item.y_select, len(item.y_select)
    best = (int(round(item.acc_select * n)), 0.0, None)  # (correct, |delta|, op)
    for c in range(L.shape[1]):
        other = L.copy()
        other[:, c] = -np.inf
        alt = other.argmax(1)
        need = other.max(1) - L[:, c]  # the image is predicted c exactly when delta > need
        order = np.argsort(need, kind="stable")
        hit_c, hit_alt = (y == c)[order], (y == alt)[order]
        # k = number of images (in this order) predicted c
        correct = np.concatenate([[0], np.cumsum(hit_c)]) + np.concatenate([np.cumsum(hit_alt[::-1])[::-1], [0]])
        s = need[order]
        for k in np.flatnonzero(correct == correct.max()):
            lo = s[k - 1] if k > 0 else s[0] - 2.0
            hi = s[k] if k < n else s[-1] + 2.0
            if hi - lo < 1e-6:
                continue
            delta = float(0.5 * (lo + hi)) if not lo < 0 < hi else 0.0
            cand = (int(correct[k]), abs(delta), {"op": "add", "tensor": "b1", "index": int(c), "value": float(f"{delta:.4g}")})
            if delta != 0.0 and (cand[0] > best[0] or (cand[0] == best[0] and best[2] is not None and cand[1] < best[1])):
                best = cand
    return [best[2]] if best[2] is not None else []


def finetune_head(item: HeadItem, steps: int = FINETUNE_STEPS, lr: float = FINETUNE_LR) -> MLP:
    """Plain gradient descent on the head: full-batch Adam on cross-entropy over the ``select`` features, starting
    from the current weights. Not an edit (it changes every weight); it is the reference for what the data allow."""
    import torch

    torch.manual_seed(0)
    X, y = torch.tensor(item.F_select, dtype=torch.float32), torch.tensor(item.y_select)
    W = torch.tensor(item.net.Ws[0], dtype=torch.float32, requires_grad=True)
    b = torch.tensor(item.net.bs[0], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(X @ W.T + b, y).backward()
        opt.step()
    return MLP([W.detach().numpy().astype(float)], [b.detach().numpy().astype(float)])


def run_baselines(items: list[HeadItem] | None = None, out: str = BASELINE_TABLE) -> list[dict]:
    from .eval import write_csv

    rows = []
    for it in items or load_items():
        rng = np.random.default_rng([it.seed, 31337])
        draws = [random_valid_edit(it, rng) for _ in range(max(RANDOM_KS))]  # nested: K=8 is a prefix of K=32, ...
        results = {f"random x{k}": it.verify_and_report(draws[:k]) for k in RANDOM_KS}
        results["bias-only search"] = it.verify_and_report([best_bias_edit(it)])
        results[f"fine-tune head ({FINETUNE_STEPS} Adam steps)"] = {**it._report(finetune_head(it)), "accepted": "not verified"}
        for name, r in results.items():
            rows.append({
                "model": it.hub_id, "dataset": it.family, "head_weights": it.net.n_weights, "baseline": name,
                "accepted": r["accepted"], "select_before": round(r["select_before"], 4), "select_after": round(r["select_after"], 4),
                "test_before": round(r["report_before"], 4), "test_after": round(r["report_after"], 4),
                "change_points": round(r["change_points"], 2), "test_images_fixed": r["report_fixed"], "test_images_broken": r["report_broken"],
                "claimed": it.claimed, "exceeds_claimed": r["exceeds_claimed"],
                "n_select": len(it.y_select), "n_test": len(it.y_report), "ops": json.dumps(r["ops"]) if r.get("ops") and len(json.dumps(r["ops"])) < 300 else "",
            })
            print(f"{it.hub_id[:44]:44s} {name:34s} select {r['select_before']:.4f}->{r['select_after']:.4f}  test {r['report_before']:.4f}->"
                  f"{r['report_after']:.4f} ({r['change_points']:+.2f} pt, +{r['report_fixed']}/-{r['report_broken']} images) accepted={r['accepted']} exceeds_claimed={r['exceeds_claimed']}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_csv(out, rows)
    return rows


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "prompts"
    if cmd == "extract":  # extract [--force] [hub id ...]
        want = set(sys.argv[2:]) - {"--force"}
        for sp in MODELS:
            if not want or sp["id"] in want:
                try:
                    extract(sp, force="--force" in sys.argv)
                except Exception as e:  # noqa: BLE001 - one model failing must not stop the others
                    print(f"[extract] {sp['id']} FAILED: {e!r}", flush=True)
    elif cmd == "records":
        for r in write_records():
            print(f"{r['id'][:50]:50s} {r['dataset']:13s} head {r['n_classes']}x{r['feature_dim']} claimed {r['claimed_accuracy']:.4f} "
                  f"test {r['test_accuracy']:.4f} (n={r['n_report']}) select {r['select_accuracy']:.4f} (n={r['n_select']}) agree {r['head_agreement']:.4f}")
    elif cmd == "baselines":
        run_baselines()
    else:
        for it in load_items():
            print(f"{it.hub_id[:50]:50s} prompt {len(it.prompt())} chars (direct {len(it.prompt(True))}), "
                  f"{'restricted to ' + str(len(it.shown_columns)) + ' columns' if it.restricted else 'full matrix'}")
